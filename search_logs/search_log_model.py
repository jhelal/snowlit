from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SearchLog:
    id: int
    query_name: str
    query: str
    results_directory: Path
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

    @classmethod
    def from_df_row(cls, row) -> "SearchLog":
        return cls(
            id=row["id"],
            query_name=row["query_name"],
            query=row["query"],
            results_directory=Path(row["results_directory"]),
            last_run_timestamp=row["last_run_timestamp"],
            status=row["status"],
        )

    @classmethod
    def from_dict(cls, row) -> "SearchLog":
        return cls(
            id=int(row["id"]),
            query_name=row["query_name"],
            query=row["query"],
            results_directory=Path(row["results_directory"]),
            last_run_timestamp=datetime.fromisoformat(row["last_run_timestamp"]),
            status=row["status"],
        )

    def to_dict(self):
        return {
            "id": self.id,
            "query_name": self.query_name,
            "query": self.query,
            "results_directory": self.results_directory.absolute(),
            "last_run_timestamp": self.last_run_timestamp.isoformat(),
            "status": self.status,
        }
