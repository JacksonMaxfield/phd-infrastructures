# PhD Infrastructures

Collection of generalized compute resources

## Projects

### Speakerbox

For setting up the storage, compute, and images needed for training and evaluating
a [`speakerbox`](https://github.com/CouncilDataProject/speakerbox) model.

## Development

### Tox / Pre-Commit

The only real tests and CI that runs on this repo is pre-commit which can be
ran with `tox`.

Will check Python formatting, linting, types, and terraform files.

### Terraform

You will need to move into the terraform directory of the project you want to
manage before you change any configuration or run any commands.

Remember, the basic commands are:

-   `terraform init` for initializing the module (installing all resources)
-   `terraform plan` for planning the changes
-   `terraform apply` for applying the changes
-   `terraform destroy` for destroying any resources (buckets aren't removed)
