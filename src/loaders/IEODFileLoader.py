from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
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
        self.is_24_7 = config.get("24_7", False)
        self.end_date = end_date

        valid_values = ", ".join(self.timeframes.keys())

        if self.default_tf not in self.timeframes:
            raise ValueError(
                f"`DEFAULT_TF` in config must be one of {valid_values}"
            )

        if tf is None:
            tf = self.default_tf
        elif tf not in self.timeframes:
            raise ValueError(f"Timeframe must be one of {valid_values}")

        self.tf = tf
        self.offset_str = self.timeframes[tf]
        self.period = self._get_period(period)

        self.data_path = Path(config["DATA_PATH"]).expanduser()

        self.ohlc_dict = dict(
            Open="first",
            High="max",
            Low="min",
            Close="last",
            Volume="sum",
        )

    def _get_period(self, period: int) -> int:
        if "h" in self.tf:
            tf = int(self.tf[:-1]) * 60
        else:
            tf = int(self.tf)

        if "h" in self.default_tf:
            default_tf = int(self.default_tf[:-1]) * 60
        else:
            default_tf = int(self.default_tf)

        if tf == default_tf:
            return period

        if tf < default_tf:
            raise ValueError("Timeframe cannot be less than default timeframe.")

        if tf % default_tf != 0:
            raise ValueError(
                f"Resampling {default_tf}min to {tf}min wont be accurate."
            )

        return tf // default_tf * period

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

        if not self.is_24_7 and self.tf in ("25", "75", "125"):
            return self.resample_df(df, self.offset_str, self.ohlc_dict)

        df = (
            df.resample(self.offset_str, origin="start")
            .agg(self.ohlc_dict)
            .dropna()
        )

        assert isinstance(df, pd.DataFrame)

        return df

    @staticmethod
    def resample_df(
        df: pd.DataFrame,
        target_tf: str,
        ohlc_dict: Dict[str, str],
    ):
        """
        Resample 25, 75 and 125 mins
        """
        lst = []
        dt = None

        while dt is None or dt <= df.index[-1]:
            dt = df.index[0] if dt is None else dt + pd.Timedelta(days=1)

            if dt not in df.index:
                continue

            slice_df = df.loc[
                dt : dt.replace(
                    hour=23,
                    minute=59,
                    second=59,
                    microsecond=10**6 - 1,
                )
            ]

            lst.append(
                slice_df.resample(target_tf, origin="start").agg(ohlc_dict)
            )

        return pd.concat(lst).dropna()

    def close(self):
        """Not required as nothing to close"""
        pass
