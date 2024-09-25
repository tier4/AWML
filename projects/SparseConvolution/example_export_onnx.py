from sparse_conv import SparseConvolution
from sparse_structure import SparseConvTensor
from torch import nn
import torch.onnx


class MyModule(nn.Module):

    def __init__(self, spatial_shape, batch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.spconv_layer = SparseConvolution(
            ndim=2,
            subm=False,
            transposed=False,
            in_channels=2,
            out_channels=5,
            fused_bn=False,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            inverse=False,
            indice_key="dummy_key",
            output_padding=0,
        )

        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def forward(self, features, indices):

        input = SparseConvTensor(features, indices, self.spatial_shape, self.batch_size)
        out_sp_tensor = self.spconv_layer(input)

        return out_sp_tensor.features, out_sp_tensor.indices


if __name__ == "__main__":

    features = torch.tensor(
        [
            [0.1, 0.1],
            [0.2, 0.2],
        ],
        dtype=torch.float32,
    ).cuda()  # n, point_features
    indices = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], dtype=torch.int32
    ).cuda()  # n, 4(batch, ind_x, ind_y, ind_z)

    input_sp_dict = {"features": features, "indices": indices}

    module = MyModule(spatial_shape=[5, 5], batch_size=1).cuda()

    module.eval()

    opset_version = 17
    onnx_name = "custom_spconv_3.onnx"

    torch.onnx.export(
        module,
        input_sp_dict,
        onnx_name,
        verbose=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["features", "indices"],
        output_names=["out_features", "out_indices"],
        dynamic_axes={
            "features": {0: "num_input_sparse_indices"},
            "indices": {0: "num_input_sparse_indices"},
            "out_features": {0: "num_output_sparse_indices"},
            "out_indices": {0: "num_output_sparse_indices"},
        },
        custom_opsets={"mydomain": opset_version},
    )

    print(f"generated {onnx_name}")
