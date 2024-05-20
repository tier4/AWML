import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Modifying TransFusion model')
    parser.add_argument('path', help='ONNX model path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = onnx.load(args.path)
    graph = gs.import_onnx(model)
    nodes = [node for node in graph.nodes if node.op == "TopK"]
    topk = nodes[0]
    k = graph.outputs[0].shape[2]
    topk.inputs[1] = gs.Constant("K", values=np.array([k], dtype=np.int64))
    topk.outputs[0].shape = [1, k]
    topk.outputs[
        0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
    topk.outputs[1].shape = [1, k]
    topk.outputs[1].dtype = np.int64
    graph.cleanup().toposort()
    output_path = (Path(args.path).parent / 'transfusion.onnx').as_posix()
    onnx.save_model(gs.export_onnx(graph), output_path)


if __name__ == '__main__':
    main()
