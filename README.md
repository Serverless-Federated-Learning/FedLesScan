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


