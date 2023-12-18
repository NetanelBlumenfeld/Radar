import datetime
import os


def ensure_path_exists(path):
    """
    Checks if a given path exists, and if not, creates it.

    Parameters:
    path (str): The path to be checked and potentially created.

    Returns:
    None
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except PermissionError:
        print(
            f"Permission denied: Cannot create directory at '{path}'. Please check your permissions or choose a different location."
        )
    except Exception as e:
        print(f"An error occurred while creating directory: {e}")


def get_time_in_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")
