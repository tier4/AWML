# Contribution docs
## Contribute flow
### 1. Report issue

If you want to add/fix some code, comment and show that you want to fix the issue before implementation.

### 2. Implementation

You should fork from autoware-ml to own repository and make new branch.

### 3. Test on integration test

For now, integration test is done on local environment.
See [test_intefration](/tools/test_intefration) for internal user.

### 4. Make PR

When you make the PR, you check the list as below "Use case for contribute".

## PR rule
### title

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) can generate categorized changelogs, for example using [git-cliff](https://github.com/orhun/git-cliff).

```
feat(autoware-ml): add loss function
```

If your change breaks some interfaces, use the ! (breaking changes) mark as follows:

```
feat(autoware-ml)!: change function name from function_a() to function_b()
```

You can use the following definition.

- feat
- fix
- chore
- ! (breaking changes)

## Use case for contribute
### Add/Fix functions to `autoware_ml`

If you want to add/fix functions to use for many projects, you should commit to `autoware_ml/*`.
It is library used for many projects and need to maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner for autoware-ml
- [ ] Performing test for function
- [ ] Check/Add/Update unit test
- [ ] Update [release note](/docs/release_note.md)

### Fix code in tools

If you want to add/fix tools to use for many projects, you should commit to `tools/*`.
It is library used for many projects and need to maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner for tools
- [ ] Performing test for tools
- [ ] Update [release note](/docs/release_note.md)

### Fix code in projects

You can fix code in a project more casually than fixing codes with `autoware_ml/*` because the area of ​​influence is small.

For PR review list with code owner for the project
- [ ] Write the result log for the model
- [ ] Update [release note](/docs/release_note.md)

### Add a new model

If you want to add a new model, you should add codes to `/projects/{model_name}/`.
For PR review list with code owner for autoware-ml

- [ ] Write why you add a new model
- [ ] Add code for `/projects/{model_name}`
- [ ] Add `/projects/{model_name}/README.md`
- [ ] Write the result log for new model
- [ ] Update [release note](/docs/release_note.md)

### Add/Fix config file in a project

If you want to add/fix config file in a project, you should add/fix `projects/{model_name}/configs/{dataset_name}/*.py`.

For PR review list with code owner for the project
- [ ] Performing test with the model
- [ ] Update `projects/{model_name}/README.md` adding evaluation result for new config.
- [ ] Delete unused config file
- [ ] Update [release note](/docs/release_note.md)
