FedLess
================================

![CI](https://github.com/andreas-grafberger/thesis-code/workflows/Lint%20and%20Test/badge.svg)
[![codecov](https://codecov.io/gh/andreas-grafberger/thesis-code/branch/main/graph/badge.svg?token=Z5SRPU9AAI)](https://codecov.io/gh/andreas-grafberger/thesis-code)

## Installation

Requires Python 3.8 (other Python 3 version might work as well)

```bash
# (Optional) Create and activate virtual environment
virtualenv .venv
source .venv/bin/activate

# Install development dependencies
pip install ".[dev]"
```

## Development

Bash scripts are checked in CI via [ShellCheck](https://github.com/koalaman/shellcheck). For python tests and linting
use the commands below.

```bash
# Run tests
pytest

# Lint
black .
```

## Deployment

Various scripts require an installed and fully
configured [AWS Command Line Interface](https://aws.amazon.com/cli/?nc1=h_ls). Install and configure it if you want to
deploy functions or images with AWS.  
If possible, we configure and deploy functions (including AWS Lambda)
with the [Serverless Framework](https://www.serverless.com/framework/docs/getting-started/). When using Serverless with
AWS make sure you have everything set up according
to [this guide](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/).

```bash
# Install Serverless Framework (MacOS/Linux only)
curl -o- -L https://slss.io/install | bash
```

If functions are not deployed using Serverless, install the required SDK, e.g., gcloud SDK for google cloud functions, OpenWhisk CLI for OpenWhisk functions, etc.
All functions are located in the *functions* directory, and maybe except those using Serverless, have a *deploy.sh,* script.
Just keep in mind that whenever we use a custom Docker image, you likely have to upload the Python wheel for FedLess to e.g., a custom s3 bucket and change the URL inside Dockerfiles or bash scripts to point to your new file on s3.
For these tasks, just look inside the *scripts* directory, where almost everything should already exist and only should need small adjustments (like changing the base path to your s3 bucket and so on).
The basic workflow is mostly: build FedLess after your changes, upload the wheel to s3, build a new Docker image (scripts for that also contained in *scripts* directory), deploy the function.
Regarding the folder names in the *functions* directory: *client* refers to FedKeeper clients, *client-indep* to FedLess clients and *client-indep-secure* 
clients are FedLess functions with security enabled.

## Running experiments
For this, you can, e.g., take a look at the bash scripts inside *experiments/system/*.
You basically have to create a new YAML config file first with information about all functions, OpenWhisk cluster, and so on.
There already exist scripts in the *scripts* folder to deploy a parameter server in a k8 cluster. The same is true for the file server.
Just bear in mind that all bash scripts were not directly designed to be used by other people, so you might have to adjust individual URLs, links to repositories, etc.
To run FedLess functions with enabled security, you basically have to create a custom Cognito user pool with the required app clients, but this is a bit more involved.
The Yaml files and everything in config files conforms to custom Pydantic models located directly inside the code.
Please keep in mind that except for MNIST, you have to host the dataset files yourself. Scripts to download the datasets, create the splits,
deploy the file and parameter servers, etc., also exist in this repository. Just one caveat: the originally required file for the Shakespeare dataset hosted by project Gutenberg
is not reachable directly in Germany due to legal reasons. So you have to host this file yourself and change the URL inside the leaf repository's code. I hosted it on s3 and created a fork of the repository. You can just do so yourself and change the URL to the GitHub repository in the bash script in this directory from my fork to yours.






