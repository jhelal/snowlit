from datetime import datetime
import pandas as pd
from search_logs.search_log_model import SearchLog
from utils import RESULTS_LOG_FILE_PATH, SEARCH_RESULTS_DIR


class SearchLogManager:
    def __init__(self) -> None:
        SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.logs_df: pd.DataFrame = self.get_logs_df()

    def get_logs_df(self):
        if not RESULTS_LOG_FILE_PATH.exists():
            self.logs_df = pd.DataFrame(
                columns=[
                    "id",
                    "query_name",
                    "query",
                    "results_directory",
                    "last_run_timestamp",
                    "status",
                ]
            )
            self.save_logs_file()

        return pd.read_csv(RESULTS_LOG_FILE_PATH)

    def add_row(self, quey_log: SearchLog):
        self.logs_df = self.logs_df.append(quey_log.to_dict(), ignore_index=True)
        self.save_logs_file()

    def save_logs_file(self):
        self.logs_df.to_csv(RESULTS_LOG_FILE_PATH, index=False)

    def update_query_status(self, query_id, status):
        self.logs_df.loc[self.logs_df["id"] == query_id, "status"] = status
        self.logs_df.loc[
            self.logs_df["id"] == query_id, "last_run_timestamp"
        ] = datetime.now()

        self.save_logs_file()

    def get_query(self, id):
        return self.logs_df.loc[self.logs_df["id"] == id]
