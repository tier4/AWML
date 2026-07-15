"""Verify voxelize_fast_gpu is bit-identical to the C++ hard_voxelize op.

Run as a unittest (skips when no CUDA / compiled op is available):

    cd projects/BEVFusion && python -m unittest bevfusion.ops.voxel.test_voxelize_fast

or directly, which also prints the speed benchmark:

    cd projects/BEVFusion && python -m bevfusion.ops.voxel.test_voxelize_fast

On synthetic clouds (incl. dense clusters that trigger the max_num_points
truncation) it checks that the fast path reproduces the C++ deterministic op's
coords / num_points / and the FULL per-voxel point tensors bit-for-bit. The
voxel ROW order differs (sort-order vs first-appearance) and is aligned away by
sorting both on the voxel id before comparison — it is irrelevant to the sparse
encoder, which indexes voxels by coords.
"""

import time
import unittest

import torch
from bevfusion.ops.voxel.voxelize import voxelization, voxelize_fast_gpu

VS = [0.17, 0.17, 0.2]
PCR = [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]
MNP, MAXV = 10, 120000


def _flat(coors, gx, gy):
    return coors[:, 2].long() * gy * gx + coors[:, 1].long() * gx + coors[:, 0].long()


def compare(points, voxel_size, pcr, max_points, max_voxels):
    """Return (coord_eq, num_eq, vox_max_abs_diff) of fast vs C++ op."""
    gx = round((pcr[3] - pcr[0]) / voxel_size[0])
    gy = round((pcr[4] - pcr[1]) / voxel_size[1])
    vc, cc, nc = voxelization(points, voxel_size, pcr, max_points, max_voxels, True)
    vf, cf, nf = voxelize_fast_gpu(points, voxel_size, pcr, max_points, max_voxels)

    # Align the two voxel orderings (sort-order vs first-appearance) by voxel id.
    oc, of = torch.argsort(_flat(cc, gx, gy)), torch.argsort(_flat(cf, gx, gy))
    coord_eq = torch.equal(_flat(cc, gx, gy)[oc], _flat(cf, gx, gy)[of])
    num_eq = torch.equal(nc[oc], nf[of])
    # Stronger than per-voxel means (which are permutation-invariant): compare
    # the FULL aligned voxel tensors. Both ops fill slots in original point
    # order, so the kept point sets AND their slot order must match exactly.
    vox_max = (vc[oc] - vf[of]).abs().max().item() if coord_eq else float("nan")
    return coord_eq, num_eq, vox_max


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


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU + the compiled voxel op")
class TestVoxelizeFast(unittest.TestCase):
    """voxelize_fast_gpu must be bit-identical to the deterministic C++ op."""

    def _check(self, num_voxels, max_pts_per):
        torch.manual_seed(0)
        pts = make_cloud(num_voxels, max_pts_per, VS, PCR, "cuda")
        coord_eq, num_eq, vox_max = compare(pts, VS, PCR, MNP, MAXV)
        self.assertTrue(coord_eq, "voxel coords differ from the C++ op")
        self.assertTrue(num_eq, "num_points_per_voxel differs from the C++ op")
        self.assertEqual(vox_max, 0.0, f"voxel features differ (max_abs_diff={vox_max})")

    def test_sparse(self):
        # ~90k voxels (< max_voxels), <= 3 pts/voxel: no max_num_points truncation.
        self._check(90000, 3)

    def test_dense(self):
        # up to 15 pts/voxel exercises the max_num_points=10 truncation (which
        # points are kept must match the C++ op exactly).
        self._check(90000, 15)


def _benchmark():
    """Print equivalence + per-frame voxelize timing (C++ op vs fast path)."""
    assert torch.cuda.is_available(), "needs a GPU + the compiled voxel op"
    dev = "cuda"
    for name, mpp in (("sparse", 3), ("dense", 15)):
        torch.manual_seed(0)
        pts = make_cloud(90000, mpp, VS, PCR, dev)
        coord_eq, num_eq, vox_max = compare(pts, VS, PCR, MNP, MAXV)
        for _ in range(5):
            voxelization(pts, VS, PCR, MNP, MAXV, True)
            voxelize_fast_gpu(pts, VS, PCR, MNP, MAXV)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            voxelization(pts, VS, PCR, MNP, MAXV, True)
        torch.cuda.synchronize()
        tc = (time.perf_counter() - t0) / 30 * 1000
        t0 = time.perf_counter()
        for _ in range(30):
            voxelize_fast_gpu(pts, VS, PCR, MNP, MAXV)
        torch.cuda.synchronize()
        tf = (time.perf_counter() - t0) / 30 * 1000
        ok = coord_eq and num_eq and vox_max == 0.0
        print(
            f"{name:8s}: coord_eq={coord_eq} num_eq={num_eq} vox_maxdiff={vox_max:.1e} | "
            f"cpp={tc:.1f}ms fast={tf:.2f}ms ({tc / tf:.0f}x)  {'PASS' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    _benchmark()
