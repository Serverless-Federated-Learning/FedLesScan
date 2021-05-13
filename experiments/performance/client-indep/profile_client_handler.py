import cProfile
import logging
import pstats
import time
from pathlib import Path

import click

from fedless.client import fedless_mongodb_handler
from fedless.models import MongodbConnectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--out", type=click.Path(), required=True)
def run(out):
    start_time_ns = time.time_ns()
    # cProfile.run()
    profiler = cProfile.Profile()
    profiler.enable()
    mongo = MongodbConnectionConfig(host="138.246.235.163")
    fedless_mongodb_handler(
        session_id="6d725adc-7e03-41d8-94a4-c135fd613adb",
        round_id=14,
        client_id="66076af9-6a15-44f4-b1f7-798224195831",
        database=mongo,
    )
    print(f"Execution took {((time.time_ns() - start_time_ns) / 10 ** 9)} seconds")
    profiler.disable()
    (
        pstats.Stats(profiler)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_callees("client.py:.*(fedless_mongodb_handler)")
        .dump_stats(out)
    )


if __name__ == "__main__":
    run()
