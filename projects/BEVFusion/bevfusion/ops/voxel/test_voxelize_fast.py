"""Verify voxelize_fast_gpu is bit-identical to the C++ hard_voxelize op.

Run on a CUDA machine with the compiled voxel op:

    cd projects/BEVFusion && python -m bevfusion.ops.voxel.test_voxelize_fast

Checks, on synthetic clouds (incl. dense clusters that trigger the
max_num_points truncation), that the fast path reproduces the C++
deterministic op's coords / num_points / per-voxel mean (HardSimpleVFE input)
bit-for-bit. The voxel ROW order differs (sort-order vs first-appearance) and is
aligned away before comparison — it is irrelevant to the sparse encoder, which
indexes voxels by coords.
"""

import time

import torch
from bevfusion.ops.voxel.voxelize import voxelization, voxelize_fast_gpu


def _flat(coors, gx, gy):
    return coors[:, 2].long() * gy * gx + coors[:, 1].long() * gx + coors[:, 0].long()


def check(name, points, voxel_size, pcr, max_points, max_voxels):
    gx = round((pcr[3] - pcr[0]) / voxel_size[0])
    gy = round((pcr[4] - pcr[1]) / voxel_size[1])
    vc, cc, nc = voxelization(points, voxel_size, pcr, max_points, max_voxels, True)
    vf, cf, nf = voxelize_fast_gpu(points, voxel_size, pcr, max_points, max_voxels)

    oc, of = torch.argsort(_flat(cc, gx, gy)), torch.argsort(_flat(cf, gx, gy))
    coord_eq = torch.equal(_flat(cc, gx, gy)[oc], _flat(cf, gx, gy)[of])
    num_eq = torch.equal(nc[oc], nf[of])
    # per-voxel mean = HardSimpleVFE input; bit-identical iff kept point-sets match
    mc = vc.sum(1) / nc.view(-1, 1).float()
    mf = vf.sum(1) / nf.view(-1, 1).float()
    vfe_max = (mc[oc] - mf[of]).abs().max().item() if coord_eq else float("nan")
    ok = coord_eq and num_eq and vfe_max == 0.0

    # timing: per-frame voxelize, C++ op vs fast (30 runs after warmup)
    for _ in range(5):
        voxelization(points, voxel_size, pcr, max_points, max_voxels, True)
        voxelize_fast_gpu(points, voxel_size, pcr, max_points, max_voxels)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        voxelization(points, voxel_size, pcr, max_points, max_voxels, True)
    torch.cuda.synchronize()
    tc = (time.perf_counter() - t0) / 30 * 1000
    t0 = time.perf_counter()
    for _ in range(30):
        voxelize_fast_gpu(points, voxel_size, pcr, max_points, max_voxels)
    torch.cuda.synchronize()
    tf = (time.perf_counter() - t0) / 30 * 1000

    print(
        f"{name:8s}: M={cc.shape[0]:>7d} coord_eq={coord_eq} num_eq={num_eq} "
        f"vfe_maxdiff={vfe_max:.1e} | cpp={tc:.1f}ms fast={tf:.2f}ms ({tc / tf:.0f}x)  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def make_cloud(num_voxels, max_pts_per, vs, pcr, dev):
    """Synthetic cloud with a CONTROLLED voxel count (< max_voxels) and up to
    max_pts_per points per voxel (exercises the max_num_points truncation).
    Uniform-random points would yield M ~ N voxels (> max_voxels), which is not
    representative of a real LiDAR frame (M ~ 100k for a 120m sweep)."""
    gx = round((pcr[3] - pcr[0]) / vs[0])
    gy = round((pcr[4] - pcr[1]) / vs[1])
    gz = round((pcr[5] - pcr[2]) / vs[2])
    vst = torch.tensor(vs, device=dev)
    rmin = torch.tensor(pcr[:3], device=dev)
    cells = torch.stack(
        [
            torch.randint(0, gx, (num_voxels,), device=dev),
            torch.randint(0, gy, (num_voxels,), device=dev),
            torch.randint(0, gz, (num_voxels,), device=dev),
        ],
        dim=1,
    )
    counts = torch.randint(1, max_pts_per + 1, (num_voxels,), device=dev)
    rep = cells.repeat_interleave(counts, 0).float()
    jitter = torch.rand(rep.shape[0], 3, device=dev) * 0.999  # stay inside the cell
    xyz = (rep + jitter) * vst + rmin
    return torch.cat([xyz, torch.rand(rep.shape[0], 2, device=dev)], dim=1)


def main():
    assert torch.cuda.is_available(), "needs a GPU + the compiled voxel op"
    dev = "cuda"
    vs = [0.17, 0.17, 0.2]
    pcr = [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]
    mnp, maxv = 10, 120000
    torch.manual_seed(0)
    # ~90k voxels (< max_voxels=120k, like a real 120m LiDAR frame); the dense
    # case uses up to 15 points/voxel to exercise the max_num_points=10 truncation.
    ok = True
    ok &= check("sparse", make_cloud(90000, 3, vs, pcr, dev), vs, pcr, mnp, maxv)
    ok &= check("dense", make_cloud(90000, 15, vs, pcr, dev), vs, pcr, mnp, maxv)
    print("\nALL PASS" if ok else "\nFAILED")


if __name__ == "__main__":
    main()
