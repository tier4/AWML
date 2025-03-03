# PR rule
## The philosophy for PR in `AWML`
### Reduce maintenance costs as much as possible

Basically, we do not merge unused code as a library because the code itself becomes technical debt.
Because we have few resource to develop `AWML`, we merge the PR only if the code quality reach enough to be maintained.
If you make the PR of new feature that is experimental code or lack of maintainability, we will not merge.
If you make prototype code but you think it is useful feature for `AWML`, please make fork repository and let us by issues and please explain in detail why it has to be in `AWML`.
After considering whether to integration and prioritizing with other developing items, we will integrate to `AWML`.

### Architecture design is more important than code itself

Regarding PR review, we review the architecture design more than code itself.
While it is easy to fix something like variable name or function composition, it is difficult to change the architecture of software.
So we often comment as "I don't think we need the option for the tool.", "I recommend you separate the tools for A and B.", or "I recommend to use config format of MMLab library instead of many args."
Of course, please keep the code itself as clean as possible at the point you make a PR.

### For core libraries and tools

When you change core libraries or core tools, we will review PRs strictly.
Please follow the PR template and write the changing point as much as possible.
As you judge how core it is, you can refer [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority).

### Independent software as much as possible

It is very costly to delete a part of feature from software that has various features.
If tools are separated for each feature, it's easy to erase with just one tool when you no longer use it.

### Separate PR

If you want to change from core library to your tools, you should separate PR.
At first, you should make PR for core library.
You check the pipelines that already exist for your changes.
The PR related to broad user like core library is reviewed carefully.
After that, you should make PR for your tools.
The PR related to limited user is reviewed casually.

## PR title

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
