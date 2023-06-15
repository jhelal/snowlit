from pybliometrics.scopus import utils
from scopus import search, backward_snowballing, forward_snowballing
from plots import (
    plot_abstracts_wordcloud,
    plot_cumulative_documents_by_year,
    plot_documents_by_affiliation,
    plot_documents_by_author,
    plot_documents_by_continent,
    plot_documents_by_country,
    plot_documents_by_source,
    plot_documents_by_type,
    plot_documents_by_year,
    plot_documents_by_year_and_source,
    plot_titles_wordcloud,
    plot_top_cited_documents,
    plot_total_documents,
)
from ppt_generation import generate_ppt_from_plots
import pandas as pd
from utils import PLOTS_DIR, CSV_RESULTS_DIR
import warnings


def init():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    init_pybliometrics()


def init_pybliometrics():
    """
    Initialize pybliometrics by creating a config file if it doesn't exist.
    """

    if utils.CONFIG_FILE.exists():
        return

    utils.create_config()


def generate_plot(df):
    plot_total_documents(df)
    plot_documents_by_year(df)
    plot_cumulative_documents_by_year(df)
    plot_documents_by_source(df)
    plot_documents_by_year_and_source(df)
    plot_documents_by_affiliation(df)
    plot_documents_by_country(df)
    plot_documents_by_continent(df)
    plot_documents_by_type(df)
    plot_documents_by_author(df)
    plot_top_cited_documents(df)
    plot_abstracts_wordcloud(df)
    plot_titles_wordcloud(df)


def main(
    query,
    *,
    forward_snowball=True,
    backward_snowball=False,
    generate_visualization=True,
):
    init()

    results_path = CSV_RESULTS_DIR / "scopus_results.csv"
    forward_snowball_path = CSV_RESULTS_DIR / "forward_snowball_results.csv"
    backward_snowball_path = CSV_RESULTS_DIR / "backward_snowball_results.csv"

    print("\nPerforming query search...")
    if results_path.exists():
        df = pd.read_csv(results_path)
    else:
        df = search(query)

    # Export the results to a CSV file
    if not results_path.exists():
        df.to_csv(results_path, index=False)
        print("New search performed and results exported to ", results_path)
    else:
        print("results for query already exists, using existing results...")

    if not forward_snowball and not backward_snowball:
        return

    if forward_snowball:
        print("\n\nPerforming forward snowballing...")
        if not forward_snowball_path.exists():
            # Perform forward snowballing
            forward_snowballing(df[:10], forward_snowball_path)
            print(
                "Forward snowballing results exported to forward_snowballing_results.csv"
            )
        else:
            print("Forward snowballing results already exist, skipping...")

    if backward_snowball:
        print("\n\nPerforming backward snowballing...")
        if not backward_snowball_path.exists():
            # Perform backward snowballing
            backward_snowballing(df[:10], backward_snowball_path)
            print(
                "Backward snowballing results exported to backward_snowballing_results.csv"
            )
        else:
            print("Backward snowballing results already exist, skipping...")

    if generate_visualization:
        print("\n\nGenerating plots...")

        # Generate plots
        generate_plot(df)

        # Generate PowerPoint presentation
        generate_ppt_from_plots()


query = (
    "(TITLE-ABS-KEY((LCA OR life cycle assessment OR life cycle analysis OR whole life carbon assessment OR "
    "embodied) AND structur* AND (building* OR infrastructure)))"
)
main(query, forward_snowball=True, backward_snowball=True, generate_visualization=True)
