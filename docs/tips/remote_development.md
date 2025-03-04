# Remote development

VS Code gives a convinient way for remote development using devcontainer configuration files. It is suggested for `AWML` users to setup configuration by themselves based on current environment.

## Configuration

Before you run container, you might need to configure environment yourself. This consists of:

* [devcontainer.json](../../.devcontainer/devcontainer.json):
  - `"BASE_IMAGE": "autoware-ml"` can be exchanged with your current project image.
  - `"--volume=${env:HOME}/autoware_data:/home/autoware/autoware_data"` can be exchanged with you external storage path. The new user inside Docker container is `autoware`, therefore you might need to change symlink within `/workspace/data` inside Docker container thought.
  - `"extensions":` list can be suited for your personal preferences.
* [Dockerfile](../../Dockerfile) can be extended with you preferable tools.

## Run container

### VS Code

Using the [Visual Studio Code](https://code.visualstudio.com/) with the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, you can develop Autoware in the containerized environment with ease.

Get the Visual Studio Code's [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
And reopen the workspace in the container by selecting `Remote-Containers: Reopen in Container` from the Command Palette (`F1`).

### Terminal

If you already run container and prefer to use system terminal rather than itegrated one in VS Code, you can log to current container using provided script:

```sh
./.devcontainer/enter.sh
```

## Debug

Debugging train loops can be perforemd as its described in [Python debugging in VS Code](https://code.visualstudio.com/docs/python/debugging).
For extra parameter you can update `"args": []` in [launch.json](../../.vscode/launch.json).
