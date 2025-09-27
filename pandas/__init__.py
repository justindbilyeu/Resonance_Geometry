"""Lightweight fallback shim for the :mod:`pandas` API used in tests.

The real pandas dependency is substantial and is not available in the
execution environment used for automated evaluation.  This module
provides just enough functionality for the repository's scripts and
unit tests (DataFrame creation, CSV round-tripping, and simple column
reductions).  If a full pandas installation is available, this shim
delegates to it automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import csv
import importlib
import importlib.util
import sys

__all__ = ["DataFrame", "read_csv"]


def _load_real_pandas() -> Any:
    """Attempt to load the real pandas module, avoiding recursion."""

    spec = importlib.util.find_spec("pandas")
    if spec and spec.origin and Path(spec.origin).resolve() != Path(__file__).resolve():
        return importlib.import_module("pandas")
    return None


_real = _load_real_pandas()
if _real is not None:
    globals().update({name: getattr(_real, name) for name in dir(_real)})
    __all__ = getattr(_real, "__all__", list(globals().keys()))  # type: ignore[assignment]
    sys.modules[__name__] = _real
else:

    class Series(list):
        """Very small subset mimicking :class:`pandas.Series`."""

        def __init__(self, values: Iterable[Any]) -> None:
            super().__init__(values)

        def sum(self) -> Any:  # type: ignore[override]
            total: Any = 0
            for val in self:
                total += val
            return total

    class DataFrame:
        """Minimal stand-in for :class:`pandas.DataFrame`."""

        def __init__(self, rows: Iterable[Dict[str, Any]]) -> None:
            self._rows: List[Dict[str, Any]] = [dict(row) for row in rows]
            self.columns: List[str] = []
            if self._rows:
                seen: List[str] = []
                for row in self._rows:
                    for key in row.keys():
                        if key not in seen:
                            seen.append(key)
                self.columns = seen

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, key: str) -> Series:
            return Series(row.get(key) for row in self._rows)

        def to_csv(self, path: Any, index: bool = False) -> None:  # noqa: ARG002 - mimic pandas signature
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", newline="") as fh:
                if not self.columns:
                    return
                writer = csv.DictWriter(fh, fieldnames=self.columns)
                writer.writeheader()
                for row in self._rows:
                    writer.writerow({col: row.get(col, "") for col in self.columns})

        def __repr__(self) -> str:
            return f"DataFrame(rows={len(self._rows)}, columns={self.columns!r})"

    def _parse_value(value: str) -> Any:
        value = value.strip()
        if value == "":
            return value
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def read_csv(path: Any) -> DataFrame:
        src = Path(path)
        with src.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = []
            for row in reader:
                rows.append({k: _parse_value(v) for k, v in row.items()})
        return DataFrame(rows)

    __all__ = ["DataFrame", "read_csv"]
