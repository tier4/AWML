# Flow of contribution
## 1. Report issue

If you want to add/fix some code, please make the issue at first.
Please comment and show any issues that need to be fixed before creating a PR.

## 2. Implementation

You should fork from autoware-ml to own repository and make new branch.
Please check [AWML design document](/docs/design/autoware_ml_design.md) before implementation.

## 3. Formatting

- We recommend some tools as below.
- Install [black](https://github.com/psf/black)

```sh
pip install black
```

- Install [isort](https://github.com/PyCQA/isort)

```sh
pip install isort
```

- Install pre-commit

```sh
pip install pre-commit
```

- Formatting by manual command

```sh
# To use:
pre-commit run -a

# runs every time you commit in git
pre-commit install  # ()
```

- If you use VSCode, you can use [tasks of VSCode](https://github.com/tier4/AWML/blob/main/.vscode/tasks.json).
  - "Ctrl+shift+P" -> Select "Tasks: Run Task" -> Select "Pre-commit: Run"
- In addition to it, we recommend VSCode extension
  - [black-formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  - [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)

## 4. Add document

If you add some feature, you must add the document like `README.md`.
The target of document is as below.

- `/docs/*`: Design documents for developers

Design documents aims for developers.
So please comment and show "why we should do" for documents.

- `/tools/*`: Process documents for engineer users

Process documents aims for engineer users.
So please comment and show "how we should do" for documents assuming that users know basic command linux around machine learning.
You can assume the user can fix the bug in the tools on their own.

- `/pipelines/*`: Process documents for non-engineer users

Process documents aims for non-engineer users.
So please comment and show "how we should do" for documents assuming that users do not know basic linux command.

## 5. Test by CI/CD

For now, integration test is done on local environment.

## 6. Make PR

Please make PR and write the contents of it in English.
When you make the PR, you check [PR rule](/docs/contribution/contribution_flow/pr_rule.md) and the list of "Use case for contribution" in [contribution docs](/docs/contribution/contribution.md).

## 7. Fix from review

When you get comments for PR, you should fix it.
