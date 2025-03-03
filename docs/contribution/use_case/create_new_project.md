# Add a new project / a new tool
## Choice 1. Merge to original MMLab libraries

Note that if you want to new algorithm, basically please make PR for [original MMLab libraries](https://github.com/open-mmlab).
After merged by it for MMLab libraries like [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [mmdetection](https://github.com/open-mmlab/mmdetection), we update the version of dependency of MMLab libraries and make our configs in `/projects` for TIER IV products.
If you want to add a config to T4dataset or scripts like onnx deploy for models of MMLab's model, you should add codes to `/projects/{model_name}/`.

For PR review list with code owner

- [ ] Write why you add a new model
- [ ] Add code for `/projects/{model_name}`
- [ ] Add `/projects/{model_name}/README.md`
- [ ] Write the result log for new model

We would you like to write summary of the new model considering the case when some engineers want to catch up.
If someone want to catch up this model, it is best situation that they need to see only github and do not need to search the information in Jira ticket jungle, Confluence ocean, and Slack universe.

## Choice 2. Make on your repository

As another way, which we recommend for especially researcher, you can make a new algorithm or a new tool on your repository.
The repository [mm-project-template](https://github.com/scepter914/mm-project-template) is one example of template repository.
You can start from this template and you can add code of `/tools/*` and `/projects/*` from `AWML` to use for your a new algorithm or a new tool.
We are glad if you want to contribute to `AWML` and the PR to add for the document of [community_support](/docs/tips/community_support.md).
We hope it promotes the community of robotics ML researchers and engineers.

For PR review list with code owner

- [ ] Add your algorithm or your tool for [community_support](/docs/tips/community_support.md)
