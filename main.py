from datetime import datetime
from enum import Enum
from pathlib import Path
from pybliometrics.scopus import utils
from scopus import search, backward_snowballing, forward_snowballing


from plots import Plotter
from ppt_generation import generate_ppt_from_plots
import pandas as pd
from utils import (
    RESULTS_LOG_FILE_PATH,
    SEARCH_RESULTS_DIR,
    delete_all_in_dir,
    take_input_as_bool,
)


class QueryStatus(Enum):
    NEW = "New"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


class SnowLit:
    def __init__(self, query: str, **options) -> None:
        self.query: str = query

        SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.log: pd.DataFrame = self.get_log()
        self.init_query()

        self.run_query(**options)

    def get_log(self):
        if not RESULTS_LOG_FILE_PATH.exists():
            self.log = pd.DataFrame(
                columns=[
                    "id",
                    "query_name",
                    "query",
                    "results_directory",
                    "last_run_timestamp",
                    "status",
                ]
            )
            self.save_log()

        return pd.read_csv(RESULTS_LOG_FILE_PATH)

    def save_log(self):
        self.log.to_csv(RESULTS_LOG_FILE_PATH, index=False)

    def init_query(self):
        existing_row = self.log[self.log["query"] == self.query]
        if not existing_row.empty:
            # The query exists in the DataFrame, return the row
            self.results_dir = Path(existing_row["results_directory"].values[0])
            force = take_input_as_bool(
                "\nQuery Results already exists, do you want regenerate all assets? (y/n): "
            )

            if force:
                delete_all_in_dir(self.results_dir)

        else:
            # The query does not exist, so we create a new directory and a new row in the DataFrame
            query_name = input(
                "\nEnter a name for your query (this will be used in results directory name): "
            ).strip()

            query_id = len(self.log) + 1

            # Create a new directory for the results
            self.results_dir = SEARCH_RESULTS_DIR / f"{query_id}_{query_name}"
            self.results_dir.mkdir(parents=True, exist_ok=True)

            new_row = {
                "id": query_id,  # Assigning the next available ID
                "query_name": query_name,
                "query": self.query,
                "results_directory": self.results_dir.absolute(),
                "last_run_timestamp": datetime.now(),
                "status": QueryStatus.NEW,  # Status is set as 'New' for a new query
            }

            self.log = self.log.append(new_row, ignore_index=True)
            self.save_log()

        # Create a new directory for the plots
        (self.results_dir / "plots").mkdir(parents=True, exist_ok=True)

    def update_status(self, status):
        self.log.loc[self.log["query"] == self.query, "status"] = status
        self.log.loc[
            self.log["query"] == self.query, "last_run_timestamp"
        ] = datetime.now()

        self.save_log()

    def run_query(
        self,
        *,
        forward_snowball=True,
        backward_snowball=False,
        generate_visualization=True,
        **kwargs,
    ):
        """
        Run the query and perform snowballing if required.

        Args:
            forward_snowball (bool, optional): Whether to perform forward snowballing. Defaults to True.
            backward_snowball (bool, optional): Whether to perform backward snowballing. Defaults to False.
            generate_visualization (bool, optional): Whether to generate plots and PowerPoint presentation. Defaults to True.
            force (bool, optional): Whether to force the query to run even if the results already exist. Defaults to False.


        """

        self.update_status(QueryStatus.RUNNING)

        csv_dir = self.results_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        results_path = csv_dir / "scopus_results.csv"
        forward_snowball_path = csv_dir / "forward_snowball_results.csv"
        backward_snowball_path = csv_dir / "backward_snowball_results.csv"

        print("\nPerforming query search...")
        if not results_path.exists():
            df = search(self.query)
            df.to_csv(results_path, index=False)

            print("New search performed and results exported to ", results_path)
        else:
            print("results for query already exists, using existing results...")
            df = pd.read_csv(results_path)

        if generate_visualization:
            print("\n\nGenerating plots for scopus results...")

            # Generate plots in the plots directory
            self.generate_plots(df)

            # Generate PowerPoint presentation
            generate_ppt_from_plots(self.results_dir)

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

        self.update_status(QueryStatus.COMPLETED)

    def generate_plots(self, df):
        plotter = Plotter(self.results_dir)
        plotter.plot_total_documents(df)
        plotter.plot_documents_by_year(df)
        plotter.plot_cumulative_documents_by_year(df)
        plotter.plot_documents_by_source(df)
        plotter.plot_documents_by_year_and_source(df)
        plotter.plot_documents_by_affiliation(df)
        plotter.plot_documents_by_country(df)
        plotter.plot_documents_by_continent(df)
        plotter.plot_documents_by_type(df)
        plotter.plot_documents_by_author(df)
        plotter.plot_top_cited_documents(df)
        plotter.plot_abstracts_wordcloud(df)
        plotter.plot_titles_wordcloud(df)


def init_pybliometrics():
    """
    Initialize pybliometrics by creating a config file if it doesn't exist.
    """

    if utils.CONFIG_FILE.exists():
        return

    utils.create_config()


def main():
    init_pybliometrics()

    query = input("\n\nEnter your query: ").strip()

    forward_snowball = take_input_as_bool("\nPerform forward snowballing? (y/n): ")
    backward_snowball = take_input_as_bool("\nPerform backward snowballing? (y/n): ")
    generate_visualization = take_input_as_bool(
        "\nGenerate plots and PowerPoint presentation? (y/n): "
    )

    SnowLit(
        query,
        forward_snowball=forward_snowball,
        backward_snowball=backward_snowball,
        generate_visualization=generate_visualization,
    )


if __name__ == "__main__":
    main()
