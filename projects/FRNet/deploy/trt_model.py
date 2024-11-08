import os
import sys
from typing import List, Tuple

import numpy.typing as npt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

sys.path.append('/workspace/projects/FRNet')
from configs.deploy.frnet_tensorrt_dynamic import tensorrt_config


class TrtModel:

    def __init__(self,
                 onnx_path: str,
                 deploy: bool = True,
                 verbose: bool = False):

        self.logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.start = cuda.Event()
        self.end = cuda.Event()
        self.stream = cuda.Stream()
        if deploy:
            self.engine = self._deploy_model(onnx_path)
        else:
            self.engine = self._load_model(onnx_path)

    def _deploy_model(self, onnx_path: str) -> trt.ICudaEngine:
        runtime = trt.Runtime(self.logger)
        builder = trt.Builder(self.logger)

        # Network definition
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            pool=trt.MemoryPoolType.WORKSPACE, pool_size=1 << 32)

        # Optimization profile
        profile = builder.create_optimization_profile()
        for name, shapes in tensorrt_config.items():
            profile.set_shape(name, shapes['min_shape'], shapes['opt_shape'],
                              shapes['max_shape'])

        config.add_optimization_profile(profile)

        # Create an ONNX parser and parse the model
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse the ONNX file')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            else:
                print('Successfully parsed the ONNX file')

        # Build the TensorRT engine
        serialized_engine = builder.build_serialized_network(network, config)
        engine_path = os.path.join(os.path.dirname(onnx_path), 'frnet.engine')
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
            print(f'TensorRT engine saved to {engine_path}')

        return runtime.deserialize_cuda_engine(serialized_engine)

    def _load_model(self, onnx_path: str) -> trt.ICudaEngine:
        runtime = trt.Runtime(self.logger)
        engine_path = os.path.join(os.path.dirname(onnx_path), 'frnet.engine')
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()

        return runtime.deserialize_cuda_engine(serialized_engine)

    def _allocate_buffers(self, shapes_dict: dict) -> Tuple[dict]:
        tensors = {'input': {}, 'output': {}}

        def _allocate(self, tensors_io: dict, indices: List[int]) -> None:
            for i in indices:
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                shape = shapes_dict[tensor_name]
                if len(shape) > 1:
                    assert shape[-1] == self.engine.get_tensor_shape(tensor_name)[-1], \
                    f'Last dimension of shape {shape} does not match the engine\'s shape {self.engine.get_tensor_shape(tensor_name)}'
                size = trt.volume(shape) * np.array(1, dtype=dtype).itemsize
                device_ptr = cuda.mem_alloc(size)
                tensors_io[tensor_name] = {
                    'device_ptr': device_ptr,
                    'shape': shape
                }

        _allocate(self, tensors['input'], [0, 1, 2, 3])
        _allocate(self, tensors['output'], [4])

        return tensors

    def _transfer_input_to_device(self, batch_inputs_dict: dict,
                                  input_tensors: dict) -> None:
        input_data = [
            batch_inputs_dict['points'], batch_inputs_dict['coors'],
            batch_inputs_dict['voxel_coors'], batch_inputs_dict['inverse_map']
        ]
        input_buffers = [(value['device_ptr'], value['shape'])
                         for value in input_tensors.values()]

        for (device_input, shape), data in zip(input_buffers, input_data):
            np_data = np.array(data, dtype=data.numpy().dtype).reshape(shape)
            cuda.memcpy_htod_async(device_input, np_data, self.stream)
        self.stream.synchronize()

    def _inference(self, tensors: dict) -> None:
        with self.engine.create_execution_context() as context:
            for key, value in tensors['input'].items():
                context.set_input_shape(key, value['shape'])
                context.set_tensor_address(key, int(value['device_ptr']))

            for key, value in tensors['output'].items():
                context.set_tensor_address(key, int(value['device_ptr']))

            self.start.record(self.stream)
            context.execute_async_v3(stream_handle=self.stream.handle)
            self.end.record(self.stream)
            self.stream.synchronize()
            latency = self.end.time_since(self.start)
            print(f'Inference latency: {latency} ms')

    def _transfer_output_from_device(self,
                                     output_tensors: dict) -> npt.ArrayLike:
        results = []
        for key, value in output_tensors.items():
            np_output = np.empty(value['shape'], dtype=np.float32)
            cuda.memcpy_dtoh_async(np_output, value['device_ptr'], self.stream)
            results.append(np_output)
        self.stream.synchronize()
        return results[0]

    def inference(self, batch_inputs_dict: dict) -> npt.ArrayLike:
        shapes_dict = {
            'points': (batch_inputs_dict['points'].shape),
            'coors': (batch_inputs_dict['coors'].shape),
            'voxel_coors': (batch_inputs_dict['voxel_coors'].shape),
            'inverse_map': (batch_inputs_dict['inverse_map'].shape),
            'seg_logit': (batch_inputs_dict['points'].shape[0], 17)
        }
        tensors = self._allocate_buffers(shapes_dict)
        self._transfer_input_to_device(batch_inputs_dict, tensors['input'])
        self._inference(tensors)
        predictions = self._transfer_output_from_device(tensors['output'])

        return predictions
