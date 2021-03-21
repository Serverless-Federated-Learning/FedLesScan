import asyncio
import functools
from typing import Optional, Type

import click
import pydantic
import yaml
from pydantic import ValidationError


def parse_yaml_file(path, model: Optional[Type[pydantic.BaseModel]] = None):
    with open(path) as f:
        file_dict = yaml.safe_load(f)
    if not model:
        return file_dict
    try:
        return model.parse_obj(file_dict)
    except (KeyError, ValidationError) as e:
        raise click.ClickException(str(e))


def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))

    return inner
