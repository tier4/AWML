# Docs for configs
## detection3d
### Use for Tensorboard

- Add backend for Tensorboard to config

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
```
