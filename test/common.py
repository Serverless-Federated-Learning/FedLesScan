from pathlib import Path


def resource_folder_path() -> Path:
    """
    Get path to test resource directory
    :return: Path
    """
    return Path(__file__).parent / "res"


def get_error_function(error: Exception.__class__, *arguments):
    # noinspection PyUnusedLocal
    def f(*args, **kwargs):
        raise error(*arguments)

    return f
