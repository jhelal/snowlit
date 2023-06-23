from enum import Enum
from pathlib import Path

# Constants
ASSETS_DIR = Path("assets")

SEARCH_RESULTS_DIR = Path("search_results")

RESULTS_LOG_FILE_PATH = SEARCH_RESULTS_DIR / "results_log.csv"

COMPARE_RESULTS_DIR = Path("compare_results")


# Enums
class QueryStatus(Enum):
    NEW = "New"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


# Helper functions
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


def save_plot_as_image(plt, image_path: Path):
    if image_path.exists():
        return

    # Save the plot as an image at the unique path
    plt.savefig(image_path.absolute(), dpi=600)

    # Close the plot to free up memory
    plt.close()

    print("plot generated at: ", image_path)
