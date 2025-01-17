# Custom Hooks

This folder consists of custom hooks that can be used by both detection2d and detection3d in `autoware-ml`. Hooks are actions that will be triggered at certain points during running cycles in a mmengine runner. Please refer to this [official turorial](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html) for more details about hooks.

## Usage

1. Please import the folder `autoware_ml.hooks` in a config file, for example:
```python
custom_imports = dict(imports=["projects.CenterPoint.models"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.hooks"]
```

2. Please add hooks to `custom_hooks` in a config file, for example:
```python
custom_hooks = [
    dict(type="MomentumInfoHook"),
]
```

## For development
- To create a new custom hook, please follow the steps below:
  1. Create a new hook class in this folder
  2. Make sure it is a subclass of `mmengine.hooks.Hook` and wrap it with `@HOOKS.register_module()`
  3. Register it with `__init__.py` in this folder
  4. **Add information about the new hook in this README**

- Please take the following notes when creating a new custom hook:
  - Make sure every hook as independent as possible
  - Make sure the priority/order of hooks is correct, for example, a new hook should be in a higher priority than `LoggerHook` if `LoggerHook` needs to log information from the new hook

## Supported hooks
- MomentumInfoHook
    - Log momentum information for every training iteration
- PytorchTrainingProfilerHook
    - Profile cuda memory and runtime information in pytorch every `interval` training iteration
- PytorchTestingProfilerHook
    - Profile cuda memory and runtime information in pytorch every `interval` testing iteration
- PytorchValidationProfilerHook
    - Profile cuda memory and runtime information in pytorch every `interval` validation iteration
- LoggerHook
    - Custom `LoggerHook` to overwrite the default `LoggerHook` to log information to tensorboard during testing by setting `logging_inference_to_tensorboard` to `True`
