from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from utils import SEARCH_RESULTS_DIR


@dataclass
class SearchLog:
    id: int
    query_name: str
    query: str
    last_run_timestamp: datetime
    status: str

    KEYS = [
        "id",
        "query_name",
        "query",
        "results_directory",
        "last_run_timestamp",
        "status",
    ]

    def get_results_directory(self) -> Path:
        return SEARCH_RESULTS_DIR / self.get_results_directory_name()

    def get_results_directory_name(self) -> str:
        return f"{self.id}_{self.query_name}"

    def get_csv_directory(self) -> Path:
        path = self.get_results_directory() / "csv"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_results_file_path(self) -> Path:
        return self.get_csv_directory() / "scopus_results.csv"

    def get_forward_snowball_results_file_path(self) -> Path:
        return self.get_csv_directory() / "forward_snowball_results.csv"

    def get_backward_snowball_results_file_path(self) -> Path:
        return self.get_csv_directory() / "backward_snowball_results.csv"

    @classmethod
    def from_df_row(cls, row) -> "SearchLog":
        return cls(
            id=row["id"],
            query_name=row["query_name"],
            query=row["query"],
            # results_directory=Path(row["results_directory"]),
            last_run_timestamp=row["last_run_timestamp"],
            status=row["status"],
        )

    @classmethod
    def from_dict(cls, row) -> "SearchLog":
        return cls(
            id=int(row["id"]),
            query_name=row["query_name"],
            query=row["query"],
            # results_directory=Path(row["results_directory"]),
            last_run_timestamp=datetime.fromisoformat(row["last_run_timestamp"]),
            status=row["status"],
        )

    def to_dict(self):
        return {
            "id": self.id,
            "query_name": self.query_name,
            "query": self.query,
            "results_directory": self.get_results_directory().absolute(),
            "last_run_timestamp": self.last_run_timestamp.isoformat(),
            "status": self.status,
        }
