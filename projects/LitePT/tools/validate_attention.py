"""
Validation script to compare Flash Attention and standard attention outputs.

This script verifies numerical consistency between Flash Attention implementation
and the standard attention implementation used for ONNX export.
"""

from __future__ import annotations

import argparse
import sys

import flash_attn
import torch
import torch.nn.functional as F


def standard_attention(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> torch.Tensor:
    """
    Standard attention implementation matching the ONNX export path.

    Args:
        qkv: Packed QKV tensor of shape (N, 3, H, head_dim)
        cu_seqlens: Cumulative sequence lengths
        max_seqlen: Maximum sequence length (patch_size)
        scale: Softmax scale factor

    Returns:
        Output tensor of shape (N, C)
    """
    N, _, H, head_dim = qkv.shape
    C = H * head_dim
    K = max_seqlen

    assert N % K == 0, f"Total tokens {N} must be divisible by patch_size {K}"

    # Reshape: (N, 3, H, head_dim) -> (N', K, 3, H, head_dim) -> (3, N', H, K, head_dim)
    q, k, v = qkv.reshape(-1, K, 3, H, head_dim).permute(2, 0, 3, 1, 4).unbind(dim=0)

    # Compute attention: (N', H, K, K)
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = F.softmax(attn, dim=-1)

    # Apply attention to values: (N', H, K, head_dim) -> (N, C)
    feat = (attn @ v).transpose(1, 2).reshape(-1, C)

    return feat


def flash_attention(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> torch.Tensor:
    """
    Flash Attention implementation used during training.

    Args:
        qkv: Packed QKV tensor of shape (N, 3, H, head_dim)
        cu_seqlens: Cumulative sequence lengths
        max_seqlen: Maximum sequence length (patch_size)
        scale: Softmax scale factor

    Returns:
        Output tensor of shape (N, C)
    """
    N, _, H, head_dim = qkv.shape
    C = H * head_dim

    feat = flash_attn.flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens,
        max_seqlen=max_seqlen,
        dropout_p=0.0,
        softmax_scale=scale,
    ).reshape(-1, C)

    return feat


def create_test_data(
    num_patches: int,
    patch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create random test data for attention comparison.

    Args:
        num_patches: Number of patches
        patch_size: Size of each patch (K)
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        dtype: Data type for tensors
        device: Device to create tensors on

    Returns:
        Tuple of (qkv tensor, cu_seqlens tensor)
    """
    N = num_patches * patch_size

    # Create random QKV tensor
    qkv = torch.randn(N, 3, num_heads, head_dim, dtype=dtype, device=device)

    # Create cumulative sequence lengths
    cu_seqlens = torch.arange(0, N + 1, patch_size, dtype=torch.int32, device=device)

    return qkv, cu_seqlens


def compare_outputs(
    out_flash: torch.Tensor,
    out_standard: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """
    Compare Flash Attention and standard attention outputs.

    Args:
        out_flash: Output from Flash Attention
        out_standard: Output from standard attention
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dictionary containing comparison metrics
    """
    # Convert to float32 for comparison
    out_flash_f32 = out_flash.float()
    out_standard_f32 = out_standard.float()

    # Compute metrics
    abs_diff = torch.abs(out_flash_f32 - out_standard_f32)
    rel_diff = abs_diff / (torch.abs(out_standard_f32) + 1e-8)

    metrics = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "allclose": torch.allclose(out_flash_f32, out_standard_f32, atol=atol, rtol=rtol),
    }

    return metrics


def run_validation(
    num_patches: int = 10,
    patch_size: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
    num_trials: int = 5,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
) -> bool:
    """
    Run validation comparing Flash Attention and standard attention.

    Args:
        num_patches: Number of patches to test
        patch_size: Size of each patch
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        num_trials: Number of random trials to run
        dtype: Data type for computation
        device: Device to run on
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        verbose: Whether to print detailed output

    Returns:
        True if all trials pass, False otherwise
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping validation.")
        return True

    scale = head_dim**-0.5
    all_passed = True

    if verbose:
        print("Validation Configuration:")
        print(f"  - num_patches: {num_patches}")
        print(f"  - patch_size: {patch_size}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - head_dim: {head_dim}")
        print(f"  - dtype: {dtype}")
        print(f"  - atol: {atol}, rtol: {rtol}")
        print(f"  - num_trials: {num_trials}")
        print()

    for trial in range(num_trials):
        # Create test data
        qkv, cu_seqlens = create_test_data(
            num_patches=num_patches,
            patch_size=patch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        # Run both implementations
        with torch.no_grad():
            out_flash = flash_attention(qkv, cu_seqlens, patch_size, scale)
            out_standard = standard_attention(qkv, cu_seqlens, patch_size, scale)

        # Compare outputs
        metrics = compare_outputs(out_flash, out_standard, atol=atol, rtol=rtol)

        if verbose:
            status = "PASS" if metrics["allclose"] else "FAIL"
            print(f"Trial {trial + 1}/{num_trials}: {status}")
            print(f"  - Max absolute difference: {metrics['max_abs_diff']:.6e}")
            print(f"  - Mean absolute difference: {metrics['mean_abs_diff']:.6e}")
            print(f"  - Max relative difference: {metrics['max_rel_diff']:.6e}")
            print(f"  - Mean relative difference: {metrics['mean_rel_diff']:.6e}")

        if not metrics["allclose"]:
            all_passed = False

    if verbose:
        print()
        if all_passed:
            print("All trials PASSED!")
        else:
            print("Some trials FAILED!")
            print("Note: Small numerical differences are expected between Flash Attention")
            print("and standard attention due to different computation order.")
            print("Consider increasing tolerances if differences are within acceptable range.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate Flash Attention vs Standard Attention numerical consistency"
    )
    parser.add_argument("--num-patches", type=int, default=10, help="Number of patches")
    parser.add_argument("--patch-size", type=int, default=1024, help="Patch size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension of each head")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of random trials")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for computation",
    )
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    passed = run_validation(
        num_patches=args.num_patches,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_trials=args.num_trials,
        dtype=dtype,
        atol=args.atol,
        rtol=args.rtol,
        verbose=not args.quiet,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
