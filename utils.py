from pathlib import Path

# Constants
ASSETS_DIR = Path("assets")

PLOTS_DIR = ASSETS_DIR / "plots"

CSV_RESULTS_DIR = ASSETS_DIR / "csv"


def save_plot_as_image(plt, image_name):
    # Define the directory path

    # Join the directory path and the image name to form the full path
    image_path = PLOTS_DIR / image_name
    if image_path.exists():
        return

    # Save the plot as an image at the unique path
    plt.savefig(image_path.absolute(), dpi=600)

    # Close the plot to free up memory
    plt.close()

    print("plot generated at: ", image_path)
