import cProfile
import logging
import pstats
import time
from pathlib import Path

import click

from fedless.client import fedless_mongodb_handler
from fedless.models import MongodbConnectionConfig, DatasetLoaderConfig
from fedless.data import DatasetLoaderBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--out", type=click.Path(), required=True)
def start(out):
    DatasetLoaderBuilder.from_config(
        DatasetLoaderConfig.parse_raw(
            """
    {
    "type" : "mnist",
    "params" : {
        "type" : "mnist",
        "indices" : [
            34434,
            34435,
            34445,
            34446,
            34454,
            34459,
            34472,
            34477,
            34485,
            34487,
            34501,
            34503,
            34506,
            34516,
            34518,
            34528,
            34560,
            34564,
            34568,
            34576,
            34583,
            34599,
            34600,
            34609,
            34618,
            34633,
            34647,
            34661,
            34666,
            34670,
            34673,
            34676,
            34677,
            34687,
            34689,
            34690,
            34702,
            34704,
            34732,
            34751,
            34772,
            34774,
            34807,
            34825,
            34835,
            34837,
            34861,
            34880,
            34881,
            34890,
            34899,
            34909,
            34914,
            34916,
            34925,
            34934,
            34935,
            34939,
            34941,
            34951,
            34959,
            34970,
            34972,
            34981,
            34982,
            34997,
            34998,
            35032,
            35053,
            35081,
            35101,
            35102,
            35121,
            35130,
            35150,
            35176,
            35177,
            35196,
            35199,
            35203,
            35223,
            35225,
            35255,
            35260,
            35277,
            35291,
            35306,
            35316,
            35321,
            35334,
            35337,
            35351,
            35361,
            35371,
            35372,
            35382,
            35391,
            35416,
            35417,
            35446,
            56660,
            56668,
            56683,
            56688,
            56689,
            56696,
            56714,
            56717,
            56728,
            56734,
            56737,
            56744,
            56750,
            56757,
            56764,
            56768,
            56772,
            56779,
            56783,
            56810,
            56831,
            56834,
            56836,
            56851,
            56865,
            56870,
            56882,
            56902,
            56903,
            56910,
            56912,
            56945,
            56953,
            56955,
            56960,
            56965,
            56971,
            56980,
            56986,
            56991,
            57005,
            57014,
            57022,
            57037,
            57046,
            57049,
            57057,
            57058,
            57069,
            57094,
            57095,
            57101,
            57109,
            57160,
            57169,
            57170,
            57190,
            57192,
            57193,
            57197,
            57203,
            57220,
            57222,
            57223,
            57243,
            57250,
            57266,
            57268,
            57272,
            57283,
            57287,
            57301,
            57303,
            57305,
            57310,
            57317,
            57322,
            57339,
            57340,
            57362,
            57378,
            57381,
            57385,
            57400,
            57410,
            57412,
            57418,
            57430,
            57433,
            57436,
            57437,
            57440,
            57453,
            57454,
            57460,
            57464,
            57473,
            57505,
            57516,
            57517,
            47714,
            47716,
            47722,
            47736,
            47739,
            47740,
            47752,
            47756,
            47786,
            47791,
            47797,
            47812,
            47815,
            47816,
            47820,
            47822,
            47835,
            47855,
            47860,
            47867,
            47887,
            47902,
            47914,
            47917,
            47923,
            47953,
            47957,
            47976,
            47997,
            48012,
            48023,
            48025,
            48026,
            48039,
            48059,
            48111,
            48121,
            48128,
            48136,
            48142,
            48167,
            48168,
            48173,
            48176,
            48180,
            48185,
            48187,
            48196,
            48200,
            48207,
            48210,
            48218,
            48227,
            48263,
            48267,
            48268,
            48274,
            48284,
            48287,
            48295,
            48299,
            48322,
            48337,
            48350,
            48364,
            48371,
            48403,
            48412,
            48414,
            48419,
            48421,
            48434,
            48438,
            48439,
            48459,
            48479,
            48488,
            48511,
            48523,
            48526,
            48529,
            48532,
            48542,
            48547,
            48554,
            48574,
            48578,
            48601,
            48602,
            48604,
            48607,
            48612,
            48613,
            48627,
            48631,
            48638,
            48641,
            48661,
            48670,
            48676,
            6727,
            6732,
            6742,
            6767,
            6789,
            6796,
            6799,
            6800,
            6807,
            6828,
            6837,
            6854,
            6869,
            6874,
            6876,
            6895,
            6903,
            6907,
            6909,
            6927,
            6930,
            6954,
            6963,
            6971,
            6975,
            6982,
            7008,
            7015,
            7049,
            7063,
            7064,
            7086,
            7095,
            7096,
            7102,
            7104,
            7109,
            7111,
            7126,
            7163,
            7179,
            7195,
            7202,
            7206,
            7212,
            7220,
            7224,
            7230,
            7268,
            7273,
            7289,
            7291,
            7294,
            7298,
            7341,
            7342,
            7361,
            7379,
            7385,
            7389,
            7398,
            7401,
            7434,
            7452,
            7456,
            7460,
            7490,
            7493,
            7503,
            7504,
            7516,
            7522,
            7543,
            7544,
            7549,
            7559,
            7562,
            7564,
            7571,
            7577,
            7580,
            7581,
            7593,
            7598,
            7641,
            7661,
            7662,
            7684,
            7700,
            7711,
            7715,
            7717,
            7718,
            7724,
            7741,
            7744,
            7759,
            7785,
            7815,
            7829,
            35200,
            35204,
            35205,
            35206,
            35210,
            35211,
            35213,
            35219,
            35222,
            35238,
            35241,
            35280,
            35284,
            35297,
            35305,
            35318,
            35322,
            35342,
            35377,
            35385,
            35411,
            35419,
            35424,
            35426,
            35433,
            35439,
            35452,
            35468,
            35470,
            35497,
            35517,
            35537,
            35540,
            35564,
            35566,
            35575,
            35578,
            35579,
            35588,
            35601,
            35607,
            35614,
            35616,
            35627,
            35664,
            35682,
            35683,
            35684,
            35687,
            35709,
            35714,
            35718,
            35726,
            35729,
            35734,
            35736,
            35750,
            35753,
            35768,
            35791,
            35795,
            35808,
            35810,
            35817,
            35825,
            35828,
            35844,
            35846,
            35851,
            35861,
            35868,
            35883,
            35886,
            35888,
            35910,
            35917,
            35918,
            35935,
            35948,
            35955,
            35962,
            35968,
            35988,
            35993,
            36002,
            36014,
            36036,
            36038,
            36049,
            36061,
            36063,
            36067,
            36068,
            36070,
            36079,
            36101,
            36113,
            36125,
            36129,
            36144,
            9608,
            9611,
            9640,
            9641,
            9650,
            9651,
            9661,
            9686,
            9694,
            9726,
            9732,
            9749,
            9762,
            9764,
            9769,
            9782,
            9789,
            9792,
            9794,
            9799,
            9805,
            9811,
            9832,
            9839,
            9844,
            9847,
            9852,
            9861,
            9866,
            9873,
            9900,
            9918,
            9924,
            9925,
            9935,
            9936,
            9937,
            9944,
            9948,
            9952,
            9956,
            9970,
            9982,
            9996,
            10001,
            10012,
            10021,
            10041,
            10054,
            10057,
            10061,
            10064,
            10066,
            10079,
            10115,
            10124,
            10126,
            10151,
            10187,
            10189,
            10197,
            10201,
            10217,
            10222,
            10233,
            10250,
            10274,
            10281,
            10319,
            10339,
            10358,
            10359,
            10379,
            10395,
            10398,
            10422,
            10426,
            10444,
            10446,
            10447,
            10482,
            10486,
            10490,
            10499,
            10513,
            10515,
            10551,
            10559,
            10561,
            10572,
            10575,
            10584,
            10592,
            10595,
            10596,
            10611,
            10619,
            10628,
            10632,
            10649
        ],
        "split" : "train",
        "proxies" : null
    }
}
    """
        )
    ).load()
    start_time_ns = time.time_ns()
    # cProfile.run()
    profiler = cProfile.Profile()
    profiler.enable()
    mongo = MongodbConnectionConfig(host="138.246.233.217")
    fedless_mongodb_handler(
        session_id="bddbfb84-47d7-444b-8d18-aacd8a31578b",
        round_id=0,
        client_id="a648db91-b0e3-4e25-98f8-0ead490a2e1a",
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
    start()
