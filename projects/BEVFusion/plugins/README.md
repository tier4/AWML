# Place TensorRT Plugin `.so` Here

Put your TensorRT custom plugin shared library in this directory **before** building the BEVFusion Docker image.

## Why you need this (ImplicitGemm / "Plugin not found")

The BEVFusion ONNX model contains **sparse convolution** nodes (`ImplicitGemm`, `GetIndicePairsImplicitGemm`). TensorRT does not ship these; they are provided by a **custom plugin** that must be loaded before parsing ONNX. If you see:

```text
Plugin not found, are the plugin name, version, and namespace correct?
operator: ImplicitGemm (checkFallbackPluginImporter)
```

then the container is **missing the plugin .so** or it is not being loaded (see config below).

## Where to get the plugin .so

The plugin is **not** built by `spconv_cpp` (that repo only builds `libspconv.so` and cumm). The TensorRT plugin that registers `ImplicitGemm` and `GetIndicePairsImplicitGemm` is built from **Autoware**'s package:

- **Repository:** [autoware_universe / perception / autoware_tensorrt_plugins](https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_tensorrt_plugins)
- **Build requirements:** TensorRT (e.g. 10.x), CUDA, and **spconv** (cumm + libspconv) installed so `find_package(spconv)` succeeds.

**Option A – Build from Autoware (recommended for version match)**  
In an environment with the same TensorRT/CUDA as your BEVFusion image (e.g. TensorRT 10.8, CUDA 12):

1. Clone `autoware_universe`, install cumm + spconv (e.g. from spconv_cpp .deb or build).
2. Build the `autoware_tensorrt_plugins` package (e.g. with colcon).
3. Copy the built shared library (e.g. `libautoware_tensorrt_plugins.so` from the install tree) into this directory:  
   `AWML/projects/BEVFusion/plugins/`
4. Rebuild the BEVFusion Docker image so the Dockerfile `COPY projects/BEVFusion/plugins/ /opt/plugins/` includes the .so.

**Option B – Copy from an Autoware Docker image**  
If you have an Autoware image that already builds perception (e.g. `universe-sensing-perception-devel-cuda`):

```bash
# Find the plugin in the Autoware container (path may vary by install layout)
docker run --rm autoware:universe-sensing-perception-devel-cuda find /opt -name "*tensorrt*plugin*.so" 2>/dev/null
# Copy it out and put the file into projects/BEVFusion/plugins/
docker create --name tmp-autoware autoware:universe-sensing-perception-devel-cuda
docker cp tmp-autoware:/path/to/libautoware_tensorrt_plugins.so ./projects/BEVFusion/plugins/
docker rm tmp-autoware
```

Then rebuild the BEVFusion image.

**Option C – Build inside the BEVFusion container**  
You can compile the plugin **inside** your existing BEVFusion Docker (no host build, no second image). The image already has TensorRT (pip), CUDA, and spconv/cumm from the Dockerfile’s spconv_cpp build.

```bash
# Enter your BEVFusion container
docker run -it --rm --gpus all -v $PWD:/workspace -w /workspace awml-bevfusion:full bash

# Build the plugin (clones Autoware plugin source, builds with CMake, installs to /opt/plugins/)
bash projects/BEVFusion/plugins/build_plugin_inside_container.sh

# Then set config or env and run export again (see “Config” below)
```

- Script location: `projects/BEVFusion/plugins/build_plugin_inside_container.sh`
- It clones `autoware_universe` (sparse checkout, plugin only), builds with a standalone CMakeLists (no ament/ROS2), and copies `libautoware_tensorrt_plugins.so` to `/opt/plugins/`.
- Optional env vars: `BUILD_DIR`, `SRC_DIR`, `INSTALL_PLUGINS_DIR`, `AUTOWARE_UNIVERSE_REF` (default `main`).
- If build fails on `find_package(spconv)` / `find_package(cumm)`, ensure the image was built with `BUILD_SPCONV_CPP=true` (Dockerfile default) so the spconv .deb is installed.

## Expected filename and paths

- In this folder (host): `libautoware_tensorrt_plugins.so`
- After Docker build (in container): `/opt/plugins/libautoware_tensorrt_plugins.so`

## Config: tell AWML to load the plugin

- **File:** `deployment/projects/bevfusion/config/deploy_config.py`
- Set `plugin_libraries` so the exporter loads the .so **before** parsing ONNX:

```python
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 32,
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)
```

Alternatively, set the env var (no image rebuild needed if the .so is already in the container):

```bash
export DEPLOY_TENSORRT_PLUGIN_LIBS=/opt/plugins/libautoware_tensorrt_plugins.so
```

## Quick check inside container

```bash
python projects/BEVFusion/deploy/check_trt_spconv_plugins.py \
  --plugin-so /opt/plugins/libautoware_tensorrt_plugins.so
```

You should see `ImplicitGemm` and `GetIndicePairsImplicitGemm` in the plugin creator list.
