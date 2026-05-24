import torch.nn.functional as F
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name="torch.nn.functional.layer_norm", backend="tensorrt"
)
@FUNCTION_REWRITER.register_rewriter(
    func_name="torch.nn.functional.layer_norm", backend="default"
)
def layer_norm__passthrough(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # Call the *original* op so the ONNX exporter sees aten::layer_norm
    # and emits a single LayerNormalization node at opset >= 17.
    ctx = FUNCTION_REWRITER.get_context()
    return ctx.origin_func(input, normalized_shape, weight, bias, eps)