import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from fast_csv_loader import csv_loader
except ModuleNotFoundError:
    exit("fast-csv-loader not found. Run `pip install fast-csv-loader`")

from .AbstractLoader import AbstractLoader

logger = logging.getLogger(__name__)

csv_loader = lru_cache(maxsize=6)(csv_loader)


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
        self.start_time = config.get("EXCHANGE_START_TIME", None)
        self.end_date = end_date

        valid_values = ", ".join(self.timeframes.keys())

        if not self.is_24_7 and self.start_time is None:
            raise ValueError(
                "Add `START_TIME` to config with the market start time in format `HH:MM`"
            )

        if self.default_tf not in self.timeframes:
            raise ValueError(f"`DEFAULT_TF` in config must be one of {valid_values}")

        if tf is None:
            tf = str(self.default_tf)
        elif tf not in self.timeframes:
            raise ValueError(f"Timeframe must be one of {valid_values}")

        self.tf = tf
        self.offset_str = self.timeframes[tf]
        self.period = self._get_period(period)
        self.date_format = config.get("DATE_FORMAT", None)

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
                date_format=self.date_format,
            )
        except IndexError:
            return
        except Exception as e:
            # Any other error log it with the symbol name
            logger.warning(f"{symbol}: Error loading file - {e!r}")
            return

        if self.tf == self.default_tf or df.empty:
            return df

        if not self.is_24_7:
            hour, minute = self.start_time.split(":")
            start_ts = df.index[0].replace(hour=int(hour), minute=int(minute))

            return self._resample_df(df, self.offset_str, self.ohlc_dict, start_ts)

        df = (
            df.resample(self.offset_str, origin="start_day")
            .agg(self.ohlc_dict)
            .dropna()
        )

        assert isinstance(df, pd.DataFrame)

        return df

    def close(self):
        """Not required as nothing to close"""
        pass

    def _tf_minutes(self, tf) -> int:
        """
        Convert timeframe string to minutes
            2h -> 120
            30 -> 30
        """
        return int(tf[:-1]) * 60 if "h" in tf else int(tf)

    def _get_period(self, period: int) -> int:
        tf = self._tf_minutes(self.tf)

        default_tf = self._tf_minutes(self.default_tf)

        if tf == default_tf:
            return period

        if tf < default_tf:
            raise ValueError("Timeframe cannot be less than default timeframe.")

        if tf % default_tf != 0:
            raise ValueError(f"Resampling {default_tf}min to {tf}min wont be accurate.")

        return tf // default_tf * period

    @staticmethod
    def _resample_df(
        df: pd.DataFrame,
        target_tf: str,
        ohlc_dict: Dict[str, str],
        start_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Resample 25, 75 and 125 mins
        """
        lst = []
        dt = None

        while dt is None or dt <= df.index[-1]:
            dt = start_ts if dt is None else dt + pd.Timedelta(days=1)

            if dt not in df.index:
                continue

            end_dt = dt.replace(
                hour=23,
                minute=59,
                second=59,
                microsecond=10**6 - 1,
            )

            slice_df = df.loc[(df.index >= dt) & (df.index <= end_dt)]

            lst.append(slice_df.resample(target_tf, origin=dt).agg(ohlc_dict))

        return pd.concat(lst).dropna()
