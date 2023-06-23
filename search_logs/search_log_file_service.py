from datetime import datetime
import pandas as pd
from search_logs.search_log_model import SearchLog
from utils import RESULTS_LOG_FILE_PATH, SEARCH_RESULTS_DIR, QueryStatus


class SearchLogFileService:
    """
    This class is responsible for managing the search logs file.

    Methods
    -------
    calculate_mean(data)
        Calculate the mean of a list of numbers.
    calculate_median(data)
        Calculate the median of a list of numbers.
    """

    _logs_df = None
    _logs_dict = None

    def __init__(self) -> None:
        # Make sure the results directory exists
        SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Make sure the logs file exists
        self._make_logs_file()

        # load the logs from the file
        self.load_logs()

    def reload_logs(self) -> None:
        """
        Reload the logs from the file, useful when the file needs to be updated externally
        """

        self.load_logs(cache=False)

    def load_logs(self, cache: bool = True) -> None:
        """
        Loads the search logs from the file

        :param cache: if True, the logs will be cached in the service
        :param cache: bool
        """
        if self._logs_df is not None and cache:
            return

        self._logs_df = pd.read_csv(RESULTS_LOG_FILE_PATH)

    def get_all_logs(self, cache: bool = True) -> list[SearchLog]:
        """
        Get all the search logs in the form of a list of SearchLog objects

        :param cache: if True, the logs will be cached in the service
        :param cache: bool

        :return: list of SearchLog objects
        :rtype: list[SearchLog]
        """
        # make sure the logs are loaded
        self.load_logs(cache=cache)

        return self._logs_df.apply(SearchLog.from_df_row, axis=1).tolist()

    def get_log(self, id: int) -> SearchLog:
        """
        Get a search log by its id

        :param id: the id of the search log
        :param id: int

        :return: the search log
        :rtype: SearchLog

        :raises ValueError: if the log does not exist

        """
        return self._get_log_by("id", id)

    def _get_log_by(self, key: str, value) -> SearchLog:
        """
        Get a search log by a key and value

        :param key: the key to search by
        :param key: str

        :param value: the value to search by
        :param value: any

        :return: the search log
        :rtype: SearchLog

        """
        result = self._logs_df.loc[self._logs_df[key] == value]
        if result.empty:
            return None
            # raise ValueError(f"Search log with {key} {value} does not exist")

        return SearchLog.from_df_row(result.iloc[0])

    def get_log_by_query(self, query: str) -> SearchLog:
        """
        Get a search log by its query

        :param query: the query of the search log
        :param query: str

        :return: the search log
        :rtype: SearchLog

        :raises ValueError: if the log does not exist

        """

        return self._get_log_by("query", query)

    def _make_logs_file(self) -> None:
        """
        Create the empty logs file with CSV headers if it does not exist
        """
        if RESULTS_LOG_FILE_PATH.exists():
            return

        self._logs_df = pd.DataFrame(
            columns=[
                "id",
                "query_name",
                "query",
                "results_directory",
                "last_run_timestamp",
                "status",
            ],
        )

        self.save_logs_to_file()

    def add_new_search_log(self, query, query_name) -> SearchLog:
        """
        Insert a new search log to the file

        :param query: the query of the search log
        :param query: str

        :param query_name: the name of the query
        :param query_name: str

        :return: saved search log
        :rtype: SearchLog


        Note: log will not be checked for duplicates, the search log id will be updated to the next available id
        """
        id = self.get_next_id()

        search_log = SearchLog(
            id=id,
            query=query,
            query_name=query_name,
            results_directory=SEARCH_RESULTS_DIR / f"{id}_{query_name}",
            status=QueryStatus.NEW,
            last_run_timestamp=datetime.now(),
        )

        self.logs_df = self._logs_df.append(search_log.to_dict(), ignore_index=True)
        self.save_logs_to_file()
        return search_log

    def save_logs_to_file(self) -> None:
        """
        Save the logs to the file
        """
        self._logs_df.to_csv(RESULTS_LOG_FILE_PATH, index=False)
        self.reload_logs()

    def update_query_status(self, search_id: int, status: QueryStatus) -> None:
        """
        Update the status of a search log by its id

        :param search_id: the id of the search log
        :param search_id: int

        :param status: the new status
        :param status: str
        """
        predicate = self._logs_df["id"] == search_id

        self._logs_df.loc[predicate, "status"] = status.value
        self._logs_df.loc[predicate, "last_run_timestamp"] = datetime.now()

        self.save_logs_to_file()

    def get_next_id(self):
        return len(self._logs_df) + 1
