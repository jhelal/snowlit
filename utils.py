import os


def save_plot_as_image(plt, image_name):
    # Define the directory path
    dir_path = "C:/Users/james/PycharmProjects/lcareview/plots/"

    # Join the directory path and the image name to form the full path
    full_path = os.path.join(dir_path, image_name)

    # Use get_unique_filename to avoid overwriting existing files
    unique_full_path = get_unique_filename(full_path)

    # Save the plot as an image at the unique path
    plt.savefig(unique_full_path, dpi=600)

    # Close the plot to free up memory
    plt.close()

    # Return the full path of the saved image
    return unique_full_path


def get_unique_filename(filepath):
    """
    Returns a unique file name by appending a number suffix to the given file name.

    Args:
        filepath (str): The original file path

    Returns:
        str: The modified file path if a file with the original name already exists, else the original file path
    """
    filename, extension = os.path.splitext(filepath)
    counter = 2

    while os.path.exists(filepath):
        filepath = f"{filename}_{counter}{extension}"
        counter += 1

    return filepath
