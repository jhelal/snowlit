import csv
from datetime import datetime
from search_logs.search_log_model import SearchLog
from utils import RESULTS_LOG_FILE_PATH, SEARCH_RESULTS_DIR, QueryStatus


class SearchLogFileService:
    logs: list[SearchLog] = []

    def __init__(self) -> None:
        # Make sure the results directory exists
        SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Make sure the logs file exists
        self.save_logs_to_file()

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
        if self.logs and cache:
            return

        with open(RESULTS_LOG_FILE_PATH, "r") as f:
            reader = csv.DictReader(f)

            self.logs = []
            for _log in reader:
                log = SearchLog.from_dict(_log)
                self.logs.append(log)

        self.logs.sort(key=lambda x: x.id)

    def get_log(self, id: int) -> SearchLog:
        """
        Get a search log by its id

        :param id: the id of the search log
        :param id: int

        :return: the search log
        :rtype: SearchLog


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
        for log in self.logs:
            if getattr(log, key) == value:
                return log

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

        self.logs.append(search_log)
        self.save_logs_to_file()
        return search_log

    def save_logs_to_file(self) -> None:
        """
        Save the logs to the file
        """

        with open(RESULTS_LOG_FILE_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SearchLog.KEYS)

            # Write the header
            writer.writeheader()

            # Write the data
            for log in self.logs:
                writer.writerow(log.to_dict())

        # self.reload_logs()

    def update_query_status(self, search_id: int, status: QueryStatus) -> None:
        """
        Update the status of a search log by its id

        :param search_id: the id of the search log
        :param search_id: int

        :param status: the new status
        :param status: str
        """
        log = self.get_log(search_id)

        log.status = status.value
        log.last_run_timestamp = datetime.now()

        self.save_logs_to_file()

    def get_next_id(self):
        if not self.logs:
            return 1

        return self.logs[-1].id + 1
