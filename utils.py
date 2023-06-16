from pathlib import Path

# Constants
ASSETS_DIR = Path("assets")

SEARCH_RESULTS_DIR = Path("search_results")

RESULTS_LOG_FILE_PATH = SEARCH_RESULTS_DIR / "results_log.csv"


CSV_RESULTS_DIR = SEARCH_RESULTS_DIR / "csv"


def take_input_as_bool(prompt: str) -> bool:
    """
    Take user input as a boolean value.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        bool: The boolean value entered by the user.
    """
    while True:
        try:
            return {"y": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input, please enter 'y' or 'n'.")


def delete_all_in_dir(path):
    for item in path.rglob("*"):
        if item.is_file():
            item.unlink()  # Removes the file
        elif item.is_dir():
            delete_all_in_dir(item)  # Recursively deletes the contents of the directory
            item.rmdir()  # Removes the directory itself
