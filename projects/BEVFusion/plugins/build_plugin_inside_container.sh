#!/usr/bin/env bash
# Build libautoware_tensorrt_plugins.so inside the BEVFusion Docker container.
# Run this script from inside the container (e.g. /workspace or any dir).
# Usage: bash projects/BEVFusion/plugins/build_plugin_inside_container.sh
# Local source (no clone): AUTOWARE_TENSORRT_PLUGINS_SRC=/path/to/perception/autoware_tensorrt_plugins
# Headers: must match the pip TensorRT runtime (libnvinfer.so.10). The script resolves them in
#   order: (1) TensorRT_INCLUDE_DIR env override, (2) pip/site-packages headers IF their
#   major.minor matches the runtime, (3) fetch matching public headers from NVIDIA/TensorRT OSS
#   (release/<major.minor>). It deliberately does NOT apt-install libnvinfer-dev — Ubuntu's
#   TensorRT is a different major.minor and causes an ABI/vtable segfault at plugin createPlugin.
#   Offline? Pre-fetch headers and pass TensorRT_INCLUDE_DIR=/path/to/TensorRT-<ver>/include.
# Result: libautoware_tensorrt_plugins.so is written to /opt/plugins/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/trt_plugin_build}"
SRC_DIR="${SRC_DIR:-/tmp/autoware_tensorrt_plugins_src}"
INSTALL_PLUGINS_DIR="${INSTALL_PLUGINS_DIR:-/opt/plugins}"
# AWML clones autoware_tensorrt_plugins from an AWML-maintained fork that bakes
# in the GetIndicePairsImplicitGemm do_sort-attribute change (required by the
# BEVFusion sparse ONNX export).
# Override the URL/ref via env vars when you want to track a different fork/branch
# (e.g. upstream autowarefoundation/autoware.universe main for an A/B build).
AUTOWARE_UNIVERSE_REPO="${AUTOWARE_UNIVERSE_REPO:-https://github.com/vividf/autoware.universe.git}"
# AUTOWARE_UNIVERSE_REF="${AUTOWARE_UNIVERSE_REF:-feat/spconv-do-sort-attribute}"
AUTOWARE_UNIVERSE_REF="${AUTOWARE_UNIVERSE_REF:-feat/implicit_gemm_int8}"

echo "[build_plugin] Script dir: $SCRIPT_DIR"
echo "[build_plugin] Build dir: $BUILD_DIR"
echo "[build_plugin] Source dir (clone): $SRC_DIR"
echo "[build_plugin] Install .so to: $INSTALL_PLUGINS_DIR"
echo "[build_plugin] Plugin repo: $AUTOWARE_UNIVERSE_REPO"
echo "[build_plugin] Plugin ref:  $AUTOWARE_UNIVERSE_REF"

# Resolve TensorRT from pip so CMake can find headers/libs
if python3 -c "import tensorrt" 2>/dev/null; then
  TRT_PIP_PATH="$(python3 -c "import tensorrt; print(tensorrt.__path__[0])")"
  echo "[build_plugin] TensorRT (pip): $TRT_PIP_PATH"
else
  echo "[build_plugin] ERROR: TensorRT not found (pip). Install with: pip install tensorrt-cu12"
  exit 1
fi

# Ensure LD_LIBRARY_PATH includes TensorRT libs from pip when we build/run
export LD_LIBRARY_PATH="${TRT_PIP_PATH}:${LD_LIBRARY_PATH}"

# Discover TensorRT lib directories/files from pip installation.
TRT_DISCOVERY="$(
python3 - <<'PY'
import glob
import os
import site

paths = []
for p in site.getsitepackages() + [site.getusersitepackages()]:
    if p and os.path.isdir(p):
        paths.append(p)

candidates = []
for base in paths:
    for rel in (
        "tensorrt",
        "tensorrt/lib",
        "tensorrt_libs",
        "nvidia/tensorrt",
        "nvidia/tensorrt/lib",
    ):
        c = os.path.join(base, rel)
        if os.path.isdir(c):
            candidates.append(c)

def pick(patterns):
    for d in candidates:
        for pat in patterns:
            matches = sorted(glob.glob(os.path.join(d, pat)))
            if matches:
                return matches[0]
    return ""

nv = pick(["libnvinfer.so*", "nvinfer.so*"])
onnx = pick(["libnvonnxparser.so*", "nvonnxparser.so*"])

include_candidates = []
for d in candidates:
    for rel in ("include", ""):
        t = os.path.join(d, rel, "NvInferRuntime.h")
        if os.path.isfile(t):
            include_candidates.append(os.path.dirname(t))
for d in ("/usr/include", "/usr/include/x86_64-linux-gnu"):
    t = os.path.join(d, "NvInferRuntime.h")
    if os.path.isfile(t):
        include_candidates.append(os.path.dirname(t))

print("HINTS=" + ";".join(dict.fromkeys(candidates)))
print("NVINFER=" + nv)
print("NVONNXPARSER=" + onnx)
print("INCLUDE=" + (include_candidates[0] if include_candidates else ""))
PY
)"
TRT_HINT_DIRS="$(echo "$TRT_DISCOVERY" | awk -F= '/^HINTS=/{print $2}')"
NVINFER_LIB="$(echo "$TRT_DISCOVERY" | awk -F= '/^NVINFER=/{print $2}')"
NVONNXPARSER_LIB="$(echo "$TRT_DISCOVERY" | awk -F= '/^NVONNXPARSER=/{print $2}')"
DISCOVERED_INCLUDE="$(echo "$TRT_DISCOVERY" | awk -F= '/^INCLUDE=/{print $2}')"
echo "[build_plugin] TensorRT hint dirs: ${TRT_HINT_DIRS:-<none>}"
echo "[build_plugin] NVINFER candidate: ${NVINFER_LIB:-<none>}"
echo "[build_plugin] NVONNXPARSER candidate: ${NVONNXPARSER_LIB:-<none>}"
echo "[build_plugin] Discovered include candidate: ${DISCOVERED_INCLUDE:-<none>}"

# ---------------------------------------------------------------------------
# Resolve TensorRT C++ headers that MATCH the pip runtime (libnvinfer.so.10).
#
# Why this matters: the pip wheel ships runtime libs but usually NO C++ headers,
# and Ubuntu's apt `libnvinfer-dev` is a DIFFERENT TensorRT major.minor (e.g. 11.x).
# Building the plugin against mismatched headers while linking the pip libs produces
# an IPluginCreatorV3One vtable/ABI mismatch that SEGFAULTS at plugin createPlugin
# time during ONNX parse. So we require headers whose major.minor == the runtime's,
# and we DO NOT apt-install libnvinfer-dev.
# ---------------------------------------------------------------------------
TRT_PY_VERSION="$(python3 -c 'import tensorrt as t; print(t.__version__)' 2>/dev/null)"
TRT_MM="$(echo "$TRT_PY_VERSION" | awk -F. 'NF>=2{print $1"."$2}')"
echo "[build_plugin] pip TensorRT runtime: ${TRT_PY_VERSION:-<unknown>} (require headers ${TRT_MM:-?})"

# Echo the major.minor recorded in a header dir's NvInferVersion.h, or empty.
# Note: a mismatched system header (e.g. Ubuntu's) may use non-numeric macros
# (TRT_*_ENTERPRISE) -> returns empty -> correctly rejected below.
_hdr_mm() {
  local h="$1/NvInferVersion.h" maj min
  [ -f "$h" ] || { echo ""; return; }
  maj="$(awk '/#define[ \t]+NV_TENSORRT_MAJOR/{print $3; exit}' "$h")"
  min="$(awk '/#define[ \t]+NV_TENSORRT_MINOR/{print $3; exit}' "$h")"
  case "$maj" in ''|*[!0-9]*) echo ""; return;; esac
  case "$min" in ''|*[!0-9]*) echo ""; return;; esac
  echo "${maj}.${min}"
}

TRT_INCLUDE_DIR=""
# (1) Explicit override wins (must contain NvInferRuntime.h).
if [ -n "${TensorRT_INCLUDE_DIR:-}" ] && [ -f "${TensorRT_INCLUDE_DIR}/NvInferRuntime.h" ]; then
  TRT_INCLUDE_DIR="$TensorRT_INCLUDE_DIR"
  echo "[build_plugin] Using TensorRT_INCLUDE_DIR override: $TRT_INCLUDE_DIR (mm=$(_hdr_mm "$TRT_INCLUDE_DIR"))"
fi
# (2) Discovered headers — only if their version matches the pip runtime.
if [ -z "$TRT_INCLUDE_DIR" ] && [ -n "$DISCOVERED_INCLUDE" ]; then
  cand_mm="$(_hdr_mm "$DISCOVERED_INCLUDE")"
  if [ -n "$TRT_MM" ] && [ "$cand_mm" = "$TRT_MM" ]; then
    TRT_INCLUDE_DIR="$DISCOVERED_INCLUDE"
    echo "[build_plugin] Using discovered TensorRT headers: $TRT_INCLUDE_DIR (mm=$cand_mm)"
  else
    echo "[build_plugin] Ignoring discovered headers $DISCOVERED_INCLUDE (version '${cand_mm:-?}' != runtime '${TRT_MM:-?}') to avoid an ABI mismatch."
  fi
fi
# (3) Fetch version-matched PUBLIC headers from TensorRT OSS (NOT apt).
if [ -z "$TRT_INCLUDE_DIR" ]; then
  if [ -z "$TRT_MM" ]; then
    echo "[build_plugin] ERROR: cannot determine the pip TensorRT version; set TensorRT_INCLUDE_DIR explicitly."
    exit 1
  fi
  OSS_DIR="${TRT_OSS_HEADERS_DIR:-/tmp/tensorrt_oss_headers}"
  if [ ! -f "$OSS_DIR/include/NvInferRuntime.h" ]; then
    echo "[build_plugin] Fetching TensorRT $TRT_MM public headers from NVIDIA/TensorRT (release/$TRT_MM)..."
    rm -rf "$OSS_DIR"
    git clone -q -b "release/$TRT_MM" --depth 1 --filter=blob:none --sparse \
      https://github.com/NVIDIA/TensorRT.git "$OSS_DIR" \
      && ( cd "$OSS_DIR" && git sparse-checkout set include >/dev/null 2>&1 ) || true
  fi
  oss_mm="$(_hdr_mm "$OSS_DIR/include")"
  if [ "$oss_mm" = "$TRT_MM" ]; then
    TRT_INCLUDE_DIR="$OSS_DIR/include"
    echo "[build_plugin] Using TensorRT OSS headers: $TRT_INCLUDE_DIR (mm=$oss_mm)"
  fi
fi

if [ -z "$TRT_INCLUDE_DIR" ]; then
  echo "[build_plugin] ERROR: could not obtain TensorRT C++ headers matching runtime ${TRT_MM:-?}."
  echo "[build_plugin]   - point TensorRT_INCLUDE_DIR at TensorRT-${TRT_PY_VERSION:-<ver>}/include, or"
  echo "[build_plugin]   - ensure network access to github.com/NVIDIA/TensorRT (release/${TRT_MM:-?})."
  echo "[build_plugin]   DO NOT use apt 'libnvinfer-dev': Ubuntu ships a different TensorRT major.minor,"
  echo "[build_plugin]   which links against the pip libs and segfaults at plugin createPlugin (ABI mismatch)."
  exit 1
fi
echo "[build_plugin] TensorRT include dir (final): $TRT_INCLUDE_DIR"

# Optional: use a bind-mounted `perception/autoware_tensorrt_plugins` tree and skip `git clone`.
# Example: export AUTOWARE_TENSORRT_PLUGINS_SRC=/workspace/autoware.universe/perception/autoware_tensorrt_plugins
AUTOWARE_TENSORRT_PLUGINS_SRC="${AUTOWARE_TENSORRT_PLUGINS_SRC:-}"
PLUGIN_SRC_DIR=""
if [ -n "$AUTOWARE_TENSORRT_PLUGINS_SRC" ]; then
  if [ -f "$AUTOWARE_TENSORRT_PLUGINS_SRC/src/implicit_gemm_plugin.cpp" ]; then
    PLUGIN_SRC_DIR="$(cd "$AUTOWARE_TENSORRT_PLUGINS_SRC" && pwd)"
    echo "[build_plugin] Using AUTOWARE_TENSORRT_PLUGINS_SRC=$PLUGIN_SRC_DIR (skipping git clone)"
  else
    echo "[build_plugin] ERROR: AUTOWARE_TENSORRT_PLUGINS_SRC=$AUTOWARE_TENSORRT_PLUGINS_SRC"
    echo "[build_plugin]        must contain src/implicit_gemm_plugin.cpp"
    exit 1
  fi
fi

# Clone only perception/autoware_tensorrt_plugins from the configured fork.
# The fork (default: vividf/autoware.universe @ feat/spconv-do-sort-attribute)
# already contains the `do_sort` attribute change; no source patching here.
# To A/B against stock upstream, set:
#   AUTOWARE_UNIVERSE_REPO=https://github.com/autowarefoundation/autoware.universe.git
#   AUTOWARE_UNIVERSE_REF=main
#
# Cache invalidation: if $SRC_DIR exists but was cloned from a different
# repo/ref (common after switching to an AWML fork), we must re-clone, otherwise
# we silently build stale upstream source. The repo/ref pair is recorded in
# $SRC_DIR/.awml_clone_meta and compared on every invocation.
if [ -z "$PLUGIN_SRC_DIR" ]; then
  CLONE_META_FILE="$SRC_DIR/.awml_clone_meta"
  EXPECTED_META="${AUTOWARE_UNIVERSE_REPO}@${AUTOWARE_UNIVERSE_REF}"
  NEEDS_CLONE=0
  if [ ! -d "$SRC_DIR/src" ] && [ ! -d "$SRC_DIR/perception/autoware_tensorrt_plugins/src" ]; then
    NEEDS_CLONE=1
  elif [ ! -f "$CLONE_META_FILE" ] || [ "$(cat "$CLONE_META_FILE" 2>/dev/null)" != "$EXPECTED_META" ]; then
    echo "[build_plugin] Cached clone at $SRC_DIR does not match $EXPECTED_META; forcing re-clone."
    NEEDS_CLONE=1
  fi

  if [ "$NEEDS_CLONE" = "1" ]; then
    echo "[build_plugin] Cloning autoware_tensorrt_plugins source from $AUTOWARE_UNIVERSE_REPO @ $AUTOWARE_UNIVERSE_REF ..."
    rm -rf "$SRC_DIR"
    git clone --depth 1 --branch "$AUTOWARE_UNIVERSE_REF" \
      --filter=blob:none --sparse \
      "$AUTOWARE_UNIVERSE_REPO" "$SRC_DIR"
    (cd "$SRC_DIR" && git sparse-checkout set perception/autoware_tensorrt_plugins)
    echo "$EXPECTED_META" > "$CLONE_META_FILE"
  fi
  if [ -d "$SRC_DIR/perception/autoware_tensorrt_plugins" ]; then
    PLUGIN_SRC_DIR="$SRC_DIR/perception/autoware_tensorrt_plugins"
  else
    PLUGIN_SRC_DIR="$SRC_DIR"
  fi
fi

if [ ! -f "$PLUGIN_SRC_DIR/src/implicit_gemm_plugin.cpp" ]; then
  echo "[build_plugin] ERROR: Clone failed or layout changed; $PLUGIN_SRC_DIR/src/implicit_gemm_plugin.cpp not found"
  exit 1
fi

# Sanity-check that the fork actually carries the do_sort attribute change.
# If the user overrode AUTOWARE_UNIVERSE_REPO/REF to stock upstream, this will
# warn but not fail (intended for A/B builds).
if grep -q "\"do_sort\"" "$PLUGIN_SRC_DIR/src/get_indices_pairs_implicit_gemm_plugin_creator.cpp" 2>/dev/null; then
  echo "[build_plugin] OK: cloned source exposes do_sort plugin attribute"
else
  echo "[build_plugin] WARNING: cloned source does NOT expose the do_sort attribute."
  echo "[build_plugin]          This is fine ONLY if you are intentionally A/B-testing against stock upstream."
  echo "[build_plugin]          For the BEVFusion sparse export, point AUTOWARE_UNIVERSE_REPO/REF at the AWML fork."
fi

# Configure and build with standalone CMakeLists (no ament/autoware_cmake)
mkdir -p "$BUILD_DIR"
cp "$SCRIPT_DIR/CMakeLists.standalone" "$BUILD_DIR/CMakeLists.txt"
cd "$BUILD_DIR"
rm -f CMakeCache.txt
cmake . \
  -DPLUGIN_SRC_DIR="$PLUGIN_SRC_DIR" \
  -DTensorRT_ROOT="$TRT_PIP_PATH" \
  -DTensorRT_EXTRA_HINT_DIRS="$TRT_HINT_DIRS" \
  -DTensorRT_INCLUDE_DIR="$TRT_INCLUDE_DIR" \
  -DNVINFER_LIBRARY="$NVINFER_LIB" \
  -DNVONNXPARSER_LIBRARY="$NVONNXPARSER_LIB" \
  -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

SO_NAME="libautoware_tensorrt_plugins.so"
if [ ! -f "$BUILD_DIR/$SO_NAME" ]; then
  echo "[build_plugin] ERROR: Build did not produce $SO_NAME"
  exit 1
fi

mkdir -p "$INSTALL_PLUGINS_DIR"
cp -a "$BUILD_DIR/$SO_NAME" "$INSTALL_PLUGINS_DIR/"
chmod 755 "$INSTALL_PLUGINS_DIR/$SO_NAME"
echo "[build_plugin] Installed: $INSTALL_PLUGINS_DIR/$SO_NAME"
ldconfig 2>/dev/null || true
echo "[build_plugin] Done. Set plugin_libraries=[\"$INSTALL_PLUGINS_DIR/$SO_NAME\"] in deploy_config.py or export DEPLOY_TENSORRT_PLUGIN_LIBS=$INSTALL_PLUGINS_DIR/$SO_NAME"
echo "[build_plugin] Verify: python3 projects/BEVFusion/deploy/check_trt_spconv_plugins.py --plugin-so $INSTALL_PLUGINS_DIR/$SO_NAME"
