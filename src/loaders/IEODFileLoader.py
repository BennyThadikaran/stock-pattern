from pathlib import Path
from datetime import datetime
from typing import Optional
from .AbstractLoader import AbstractLoader
from .utils import csv_loader
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class IEODFileLoader(AbstractLoader):
    """
    A class to load intraday data from CSV files.

    Parameters:
    :param config: User config
    :type config: dict
    :param timeframe: See timeframes property for list of acceptable strings
    :type timeframe: str
    :param end_date: End date upto which date must be returned
    :type end_date: Optional[datetime]
    :param period: Number of lines to return from end_date or end of file
    """

    timeframes = {
        "1": "1min",
        "5": "5min",
        "10": "10min",
        "15": "15min",
        "25": "25min",
        "30": "30min",
        "60": "1h",
        "75": "75min",
        "125": "125min",
        "2h": "2h",
        "4h": "4h",
    }

    def __init__(
        self,
        config: dict,
        tf: Optional[str] = None,
        end_date: Optional[datetime] = None,
        period: int = 160,
    ):

        # No need to call close method on this class
        self.closed = True

        # Default timeframe is 1 min.
        self.default_tf = str(config.get("DEFAULT_TF", "1"))
        self.end_date = end_date

        if self.default_tf not in self.timeframes:
            valid_values = ", ".join(self.timeframes.keys())

            raise ValueError(
                f"`DEFAULT_TF` in config must be one of {valid_values}"
            )

        if tf is None:
            tf = self.default_tf

        if not tf in self.timeframes:
            valid_values = ", ".join(self.timeframes.keys())

            raise ValueError(f"Timeframe must be one of {valid_values}")

        self.tf = tf
        self.offset_str = self.timeframes[tf]
        self.period = period

        self.data_path = Path(config["DATA_PATH"]).expanduser()

        self.ohlc_dict = dict(
            Open="first",
            High="max",
            Low="min",
            Close="last",
            Volume="sum",
        )

    def get(self, symbol: str) -> Optional[pd.DataFrame]:

        file = self.data_path / f"{symbol.lower()}.csv"

        if not file.exists():
            logger.warning(f"File not found: {file}")
            return

        try:
            df = csv_loader(
                file,
                period=self.period,
                end_date=self.end_date,
            )
        except (IndexError, ValueError):
            return

        if self.tf == self.default_tf or df.empty:
            return df

        df = (
            df.resample(self.offset_str, origin="start")
            .agg(self.ohlc_dict)
            .dropna()
        )

        assert isinstance(df, pd.DataFrame)

        return df

    def close(self):
        """Not required as nothing to close"""
        pass
