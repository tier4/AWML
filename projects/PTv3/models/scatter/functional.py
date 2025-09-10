import torch
import torch_scatter
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes


class SegmentCSR(Function):

    @staticmethod
    def symbolic(
        g,
        src: torch.Tensor,
        indptr: torch.Tensor,
        reduce: str,
    ):

        output = g.op(
            "autoware::SegmentCSR",
            src,
            indptr,
            reduce_s=reduce,
            outputs=1,
        )
        src_shape = _get_tensor_sizes(src)
        if src_shape is not None and hasattr(output.type(), "with_sizes"):
            output_type = src.type().with_sizes([src_shape[0], src_shape[1]])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(
        ctx,
        src: torch.Tensor,
        indptr: torch.Tensor,
        reduce: str,
    ):
        return torch_scatter.segment_csr(src, indptr, reduce=reduce)


class Unique(Function):

    @staticmethod
    def symbolic(
        g,
        x: torch.Tensor,
    ):

        output = g.op(
            "autoware::CustomUnique",
            x,
            outputs=4,
        )
        x_shape = _get_tensor_sizes(x)
        if x_shape is not None and hasattr(output[0].type(), "with_sizes"):
            output_type = x.type().with_sizes([x_shape[0]])
            num_output_type = x.type().with_sizes([1])
            output[0].setType(output_type)
            output[1].setType(output_type)
            output[2].setType(output_type)
            output[3].setType(num_output_type)

        return output

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
    ):
        unique, inverse_indices, count = torch.unique(
            x,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        num_out = torch._shape_as_tensor(unique).to(x.device)[0]

        return unique, inverse_indices, count, num_out


class Argsort(Function):

    @staticmethod
    def symbolic(
        g,
        x: torch.Tensor,
    ):

        output = g.op(
            "autoware::Argsort",
            x,
            outputs=1,
        )
        x_shape = _get_tensor_sizes(x)
        if x_shape is not None and hasattr(output.type(), "with_sizes"):
            output_type = x.type().with_sizes(x_shape)
            output.setType(output_type)
        return output

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
    ):
        _, indices = torch.sort(x)
        return indices


segment_csr = SegmentCSR.apply
unique = Unique.apply
argsort = Argsort.apply
