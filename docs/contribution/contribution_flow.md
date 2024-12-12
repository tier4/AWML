# Docs for contribution flow
## Flow of contribution
### 1. Report issue

If you want to add/fix some code, please make the issue at first.
You should comment and show the issue and show what you want to fix  before implementation.

### 2. Implementation

You should fork from autoware-ml to own repository and make new branch.

### 3. Formatting

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

- If you use VSCode, you can use [tasks of VSCode](https://github.com/tier4/autoware-ml/blob/main/.vscode/tasks.json).
  - "Ctrl+shift+P" -> Select "Tasks: Run Task" -> Select "Pre-commit: Run"
- In addition to it, we recommend VSCode extension
  - [black-formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  - [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)

### 4. Test by CI/CD

For now, integration test is done on local environment.
See [test_integration](/tools/test_integration) for internal user.

### 5. Make PR

Please make PR and write the contents of it in English.
When you make the PR, you check the list as below "Use case for contribute".

### 6. Fix from review

When you get comments for PR, you should fix it.

## PR rule
### The philosophy for PR in `autoware-ml`

- Reduce maintenance costs as much as possible

Basically, we do not merge unused code as a library because the code itself becomes technical debt.
Because we have few resource to develop `autoware-ml`, we merge the PR only if the code quality reach enough to be maintained.
If you make the PR of new feature that is experimental code or lack of maintainability, we will not merge.
If you make prototype code but you think it is useful feature for `autoware-ml`, please make fork repository and let us by issues and please explain in detail why it has to be in `autoware-ml`.
After considering whether to integration and prioritizing with other developing items, we will integrate to `autoware-ml`.

- Architecture design is more important than code itself

Regarding PR review, we review the architecture design more than code itself.
While it is easy to fix something like variable name or function composition, it is difficult to change the architecture of software.
So we often comment as "I don't think we need the option for the tool.", "I recommend you separate the tools for A and B.", or "I recommend to use config format of MMLab library instead of many args."
Of course, please keep the code itself as clean as possible at the point you make a PR.

- For core libraries and tools

When you change core libraries or core tools, we will review PRs strictly.
Please follow the PR template and write the changing point as much as possible.
As you judge how core it is, you can refer [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority).

- Independent software as much as possible

It is very costly to delete a part of feature from software that has various features.
If tools are separated for each feature, it's easy to erase with just one tool when you no longer use it.

- Separate PR

If you want to change from core library to your tools, you should separate PR.
At first, you should make PR for core library.
You check the pipelines that already exist for your changes.
This PR is reviewed carefully.
After that, you should make PR for your tools.
This PR is reviewed casually.

### PR title

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) can generate categorized changelogs, for example using [git-cliff](https://github.com/orhun/git-cliff).

```
feat(autoware_ml): add loss function
```

If your change breaks some interfaces, use the ! (breaking changes) mark as follows:

```
feat(autoware_ml)!: change function name from function_a() to function_b()
```

You can use the following definition.

- feat
- fix
- chore
- ! (breaking changes)

### Document

If you add some feature, you must add the document like `README.md`.
The target of document is as below.

- `/docs/*`: Design documents for developers

Design documents aims for developers.
So you should write "why we should do" for documents.

- `/tools/*`: Process documents for engineer users

Process documents aims for engineer users.
So you should write "how we should do" for documents assuming that users know basic command linux around machine learning.
You can assume the user can fix the bug in the tools on their own.

- `/pipelines/*`: Process documents for non-engineer users

Process documents aims for non-engineer users.
So you should write "how we should do" for documents assuming that users do not know basic linux command.
