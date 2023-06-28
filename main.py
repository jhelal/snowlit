from pybliometrics.scopus import utils
from search_logs.search_log_file_service import SearchLogFileService
from scopus import search, backward_snowballing, forward_snowballing


from plots import Plotter
from ppt_generation import generate_ppt_from_plots
import pandas as pd
from utils import QueryStatus, delete_all_in_dir, take_input_as_bool


class SnowLit:
    def __init__(self, query: str, **options) -> None:
        self.query: str = query

        # Initialize the SearchLogManager
        self.log_service: SearchLogFileService = SearchLogFileService()

        # Initialize the query
        self.log = self.init_query()

        # Run the query
        self.run_query(**options)

    def init_query(self):
        log = self.log_service.get_log_by_query(self.query)

        if log:
            # The query exists in the data source
            delete_existing_assets = take_input_as_bool(
                "\nQuery Results already exists, do you want regenerate all assets? (y/n): "
            )

            if delete_existing_assets:
                delete_all_in_dir(log.get_results_directory())

        else:
            # The query does not exist, so we create a new directory and a new row in the DataFrame
            query_name = input(
                "\nEnter a name for your query (this will be used in results directory name): "
            ).strip()

            log = self.log_service.add_new_search_log(self.query, query_name)

        # Create a new directory for the plots
        (log.get_results_directory() / "plots").mkdir(parents=True, exist_ok=True)

        return log

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

        self.log_service.update_query_status(self.log.id, QueryStatus.RUNNING)

        results_path = self.log.get_results_file_path()
        forward_snowball_path = self.log.get_forward_snowball_results_file_path()
        backward_snowball_path = self.log.get_backward_snowball_results_file_path()

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
            generate_ppt_from_plots(self.query, self.log.get_results_directory())

        if forward_snowball:
            print("\n\nPerforming forward snowballing...")
            if not forward_snowball_path.exists():
                # Perform forward snowballing
                forward_snowballing(df, forward_snowball_path)
                print(
                    "Forward snowballing results exported to forward_snowballing_results.csv"
                )
            else:
                print("Forward snowballing results already exist, skipping...")

        if backward_snowball:
            print("\n\nPerforming backward snowballing...")
            if not backward_snowball_path.exists():
                # Perform backward snowballing
                backward_snowballing(df, backward_snowball_path)
                print(
                    "Backward snowballing results exported to backward_snowballing_results.csv"
                )
            else:
                print("Backward snowballing results already exist, skipping...")

        self.log_service.update_query_status(self.log.id, QueryStatus.COMPLETED)

    def generate_plots(self, df):
        plotter = Plotter(self.log.get_results_directory())
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
