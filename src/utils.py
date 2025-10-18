import logging
from typing import Any, NamedTuple, Optional, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Point(NamedTuple):
    x: pd.Timestamp
    y: float


class Coordinate(NamedTuple):
    start: Point
    end: Point


class Line(NamedTuple):
    line: Coordinate
    slope: float
    y_int: float


T = TypeVar("T")

is_silent = None

ascii_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

fib_ser = pd.Series((0.236, 0.382, 0.5, 0.618, 0.707, 0.786, 0.886, 1))


def get_relative_clusters(levels, reference_key) -> dict:
    """
    Find price levels relatively close to the price associated with the reference key.

    Parameters:
        price_levels (dict): A dictionary of price levels with keys as level names and values as prices (float).
        reference_key (str): The key in the dictionary to use as the reference point.
    """
    reference_price = levels[reference_key]

    mean_dev = np.mean(
        tuple(
            abs(price - reference_price)
            for price in levels.values()
            if price != reference_price
        )
    )

    # Find price levels that fall within the mean deviation range
    return {
        level: price
        for level, price in levels.items()
        if abs(price - reference_price) <= mean_dev
    }


def getY(slope, yintercept, x_value) -> float:
    """
    Returns the value of the Y-axis (Price) at the given X-axis value

    Useful for calculating the price on a trendline at the specified date.
    """
    # y = mx + b
    # where m is slope, b is yintercept
    return slope * x_value + yintercept


def make_serializable(obj: T) -> T:
    """Convert pandas.Timestamp and numpy.Float32 objects in obj
    to serializable native types"""

    def serialize(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, np.generic)):
            # Convert Pandas Timestamp to Python datetime or NumPy item
            return obj.isoformat() if isinstance(obj, pd.Timestamp) else obj.item()
        elif isinstance(obj, (list, tuple)):
            # Recursively convert lists and tuples
            return tuple(serialize(item) for item in obj)
        elif isinstance(obj, dict):
            # Recursively convert dictionaries
            return {key: serialize(value) for key, value in obj.items()}
        return obj

    return serialize(obj)


def has_time_component(datetime_index: pd.DatetimeIndex) -> bool:
    """Return True if any value in DatetimeIndex has time component
    other than `00:00:00` (Midnight hour)
    Ex Datetime(2023, 12, 10)           ->  `00:00:00` (Defaults to midnight)
       Datetime(2023, 12, 10, 0, 0)     ->  `00:00:00` (Midnight time)
       Datetime(2023, 12, 10, 12, 10)   ->  `12:10:00`
    """
    return any(datetime_index.to_series().dt.time != pd.Timestamp("00:00:00").time())


def get_atr(high: pd.Series, low: pd.Series, close: pd.Series, window=15) -> pd.Series:
    # Calculate true range
    tr = pd.DataFrame(index=high.index)
    tr["h-l"] = high - low
    tr["h-pc"] = abs(high - close.shift(1))
    tr["l-pc"] = abs(low - close.shift(1))
    tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)

    return tr.tr.rolling(window=window).mean()


def is_triangle(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    avgBarLength: float,
) -> Optional[str]:
    r"""
          A
         /\        C
        /  \      /\    E
       /    \    /  \  /\
      /      \  /    \/  F
     /        \/      D
    /         B            Symmetric

        A
       /\      C
      /  \    /\    E
     /    \  /  \  /\ 
    /      \/    \/  F
           B     D         Descending

         A       C     E
        /\      /\    /\
       /  \    /  \  /  \
      /    \  /    \/    F
     /      \/     D
    /        B             Ascending
    """
    is_ac_straight_line = abs(a - c) <= avgBarLength
    is_ce_straight_line = abs(c - e) <= avgBarLength

    if is_ac_straight_line and is_ce_straight_line and b < d < f < e:
        return "Ascending"

    is_bd_straight_line = abs(b - d) <= avgBarLength

    if is_bd_straight_line and a > c > e > f and f >= d:
        return "Descending"

    if a > c > e and b < d < f and e > f:
        return "Symmetric"

    return None


def is_hns(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    avgBarLength: float,
) -> bool:
    r"""
    Head and Shoulders
                C
                /\
        A      /  \      E
        /\    /    \    /\
       /  \  /      \  /  \
      /    \/________\/____\F__Neckline
     /      B         D     \
    /                        \
    """
    shoulder_height_threshold = round(avgBarLength * 0.6, 2)

    return (
        c > max(a, e)
        and max(b, d) < min(a, e)
        and f < e
        and abs(b - d) < avgBarLength
        and abs(c - e) > shoulder_height_threshold
    )


def is_reverse_hns(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    avgBarLength: float,
) -> bool:
    r"""
    Reverse Head and Shoulders
    \
     \                  /
      \   _B_______D___/___
       \  /\      /\  /F   Neckline
        \/  \    /  \/
        A    \  /    E
              \/
              C
    """
    shoulder_height_threshold = round(avgBarLength * 0.6, 2)

    return (
        c < min(a, e)
        and min(b, d) > max(a, e)
        and f > e
        and abs(b - d) < avgBarLength
        and abs(c - e) > shoulder_height_threshold
    )


def is_double_top(
    a: float,
    b: float,
    c: float,
    d: float,
    aVol: int,
    cVol: int,
    avgBarLength: float,
    atr: float,
) -> bool:
    r"""
    Double Top
          A     C
         /\    /\
        /  \  /  \
       /    \/    D
      /      B
     /
    /
    """
    return (
        c - b < atr * 4
        and abs(a - c) <= avgBarLength * 0.5
        and cVol < aVol
        and b < min(a, c)
        and b < d < c
    )


def is_double_bottom(
    a: float,
    b: float,
    c: float,
    d: float,
    aVol: int,
    cVol: int,
    avgBarLength: float,
    atr: float,
) -> bool:
    r"""
    Double Bottom
      \
       \
        \      B
         \    /\    D
          \  /  \  /
           \/    \/
            A     C
    """

    return (
        b - c < atr * 4
        and abs(a - c) <= avgBarLength * 0.5
        and cVol < aVol
        and b > max(a, c)
        and b > d > c
    )


def is_bearish_vcp(
    a: float, b: float, c: float, d: float, e: float, avgBarLength: float
) -> bool:
    r"""
    Volatilty Contraction pattern
          B
         /\      D
        /  \    /\
       /    \  /  \
      /      \/    E
     A       C

    B is highest point in pattern
    D is second highest after B
    """
    if c < a and abs(a - c) >= avgBarLength * 0.5:
        return False

    return (
        abs(a - c) <= avgBarLength
        and abs(b - d) >= avgBarLength * 0.8
        and b > max(a, c, d, e)
        and d > max(a, c, e)
        and e > c
    )


def is_bullish_vcp(
    a: float, b: float, c: float, d: float, e: float, avgBarLength: float
) -> bool:
    r"""
    Volatilty Contraction pattern

       A        C
         \      /\    E
          \    /  \  /
           \  /    \/
            \/      D
             B

    B is lowest point in pattern
    D is second lowest after B
    """
    if c > a and abs(a - c) >= avgBarLength * 0.5:
        return False

    return (
        abs(a - c) <= avgBarLength
        and abs(b - d) >= avgBarLength * 0.8
        and b < min(a, c, d, e)
        and d < min(a, c, e)
        and e < c
    )


def get_max_min(
    df: pd.DataFrame, barsLeft=6, barsRight=6, pivot_type="both"
) -> pd.DataFrame:
    window = barsLeft + 1 + barsRight

    local_max_dt = []
    local_min_dt = []
    cols = ["P", "V"]

    for win in df.rolling(window):
        if win.shape[0] < window:
            continue

        idx = win.index[barsLeft + 1]  # center candle

        if win.High.idxmax() == idx:
            local_max_dt.append(idx)

        if win.Low.idxmin() == idx:
            local_min_dt.append(idx)

    maxima = pd.DataFrame(df.loc[local_max_dt, ["High", "Volume"]])
    maxima.columns = cols

    minima = pd.DataFrame(df.loc[local_min_dt, ["Low", "Volume"]])
    minima.columns = cols

    if pivot_type == "high":
        return maxima

    if pivot_type == "low":
        return minima

    return pd.concat([maxima, minima], axis=0).sort_index()


def get_next_index(index: pd.DatetimeIndex, idx: pd.Timestamp) -> int:
    pos = index.get_loc(idx)

    if isinstance(pos, slice):
        return pos.stop

    if isinstance(pos, int):
        return pos + 1

    raise TypeError("Expected Integer")


def get_prev_index(index: pd.DatetimeIndex, idx: pd.Timestamp) -> int:
    pos = index.get_loc(idx)

    if isinstance(pos, slice):
        return pos.stop

    if isinstance(pos, int):
        return pos - 1

    raise TypeError("Expected Integer")


def generate_trend_line(
    series: pd.Series, date1: pd.Timestamp, date2: pd.Timestamp
) -> Line:
    """Return the end coordinates for a trendline along with slope and y-intercept
    Input: Pandas series with a pandas.DatetimeIndex, and two dates:
           The two dates are used to determine two "prices" from the series

    Output: tuple(tuple(coord, coord), slope, y-intercept)
    source: https://github.com/matplotlib/mplfinance/blob/master/examples/scratch_pad/trend_line_extrapolation.ipynb

    """
    index = series.index

    p1 = float(series[date1])
    p2 = float(series[date2])

    d1 = index.get_loc(date1)
    d2 = index.get_loc(date2)

    lastIdx = index[-1]
    lastIdxPos = index.get_loc(lastIdx)

    assert isinstance(lastIdx, pd.Timestamp)
    assert isinstance(d1, int)
    assert isinstance(d2, int)
    assert isinstance(lastIdxPos, int)
    assert isinstance(p1, float)
    assert isinstance(p2, float)

    # b = y - mx
    # where m is slope,
    # b is y-intercept
    # slope m = change in y / change in x
    m = (p2 - p1) / (d2 - d1)

    yintercept = p1 - m * d1  # b = y - mx

    return Line(
        line=Coordinate(
            start=Point(x=date1, y=m * d1 + yintercept),  # y = mx + b
            end=Point(x=lastIdx, y=m * lastIdxPos + yintercept),
        ),
        slope=m,
        y_int=yintercept,
    )


def find_bullish_flag(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Find Bullish High Pole and Flag pattern.

    Returns None if no patterns found.
    """

    if len(df) < 50:
        return

    lastIdx = df.index[-1]

    recent_high_idx = df.High.iloc[-7:].idxmax()

    # Last candle is the weekly high
    if recent_high_idx == lastIdx or len(df.loc[recent_high_idx:]) < config.get(
        "FLAG_MAX_BARS", 5
    ):
        return

    monthly_high = df.High.iloc[-30:].max()
    three_month_high = df.High.iloc[-90:].max()

    recent_high = df.at[recent_high_idx, "High"]
    recent_low = df.loc[recent_high_idx:, "Low"].min()

    # A new high formed in the last 7 days exeeds the 30 and 90 day high
    if recent_high >= monthly_high and recent_high >= three_month_high:
        sma20_ser = df.Close.rolling(20).mean()
        sma50_ser = df.Close.rolling(50).mean()

        close = df.at[df.index[-1], "Close"]
        sma20 = sma20_ser.iloc[-1]
        sma50 = sma50_ser.iloc[-1]

        last_pivot_idx = pivots.index[-1]
        last_pivot = pivots.at[last_pivot_idx, "P"]

        if isinstance(last_pivot, pd.Series):
            last_pivot = last_pivot.max()

        fib_50 = last_pivot + (recent_high - last_pivot) / 2

        flag_df = df.loc[recent_high_idx:, ["High", "Low"]]
        flag_df["RANGE"] = flag_df.High - flag_df.Low

        # flag_start_range = flag_df.RANGE.iloc[1]
        # flag_max_range = flag_df.RANGE.iloc[2:].max()

        if (
            sma20 < sma50 * 1.08 or recent_low < fib_50
            # or flag_max_range > flag_start_range
        ):
            return

        return dict(
            sym=sym,
            pattern="FLAGU",
            start=last_pivot_idx,
            end=lastIdx,
            df_start=df.index[0],
            df_end=lastIdx,
            points=dict(
                A=(last_pivot_idx, last_pivot),
                B=(recent_high_idx, recent_high),
                C=(lastIdx, close),
            ),
            extra_points=dict(direction=(recent_high_idx, recent_high)),
        )


def find_bearish_flag(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Find Bearish High Pole and Flag pattern.

    Returns None if no patterns found.
    """

    if len(df) < 50:
        return

    lastIdx = df.index[-1]

    recent_low_idx = df.Low.iloc[-7:].idxmin()

    if recent_low_idx == lastIdx or len(df.loc[recent_low_idx:]) < config.get(
        "FLAG_MAX_BARS", 5
    ):
        return

    monthly_low = df.Low.iloc[-30:].min()
    three_month_low = df.Low.iloc[-90:].min()

    recent_low = df.at[recent_low_idx, "Low"]
    recent_high = df.loc[recent_low_idx:, "High"].max()

    # A new Low formed in the last 7 days exeeds the 30 and 90 day Low
    if recent_low <= monthly_low and recent_low <= three_month_low:
        sma20_ser = df.Close.rolling(20).mean()
        sma50_ser = df.Close.rolling(50).mean()

        close = df.at[df.index[-1], "Close"]

        sma20 = sma20_ser.iloc[-1]
        sma50 = sma50_ser.iloc[-1]

        last_pivot_idx = pivots.index[-1]
        last_pivot = pivots.at[last_pivot_idx, "P"]

        if isinstance(last_pivot, pd.Series):
            last_pivot = last_pivot.min()

        fib_50 = last_pivot - (last_pivot - recent_low) / 2

        flag_df = df.loc[recent_low_idx:, ["High", "Low"]]
        flag_df["RANGE"] = flag_df.High - flag_df.Low

        # flag_start_range = flag_df.RANGE.iloc[1]
        # flag_max_range = flag_df.RANGE.iloc[2:].max()

        if (
            sma20 > sma50 * 0.92 or recent_high > fib_50
            # or flag_max_range > flag_start_range
        ):
            return

        return dict(
            sym=sym,
            pattern="FLAGD",
            start=last_pivot_idx,
            end=lastIdx,
            df_start=df.index[0],
            df_end=lastIdx,
            points=dict(
                A=(last_pivot_idx, last_pivot),
                B=(recent_low_idx, recent_low),
                C=(lastIdx, close),
            ),
            extra_points=dict(direction=(recent_low_idx, recent_low)),
        )


def find_bullish_vcp(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Volatilty Contraction Pattern Bullish.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    a_idx = pivots["P"].idxmax()

    a = pivots.at[a_idx, "P"]

    assert isinstance(a_idx, pd.Timestamp)

    e_idx = df.index[-1]
    e = df.at[e_idx, "Close"]

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        b_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()

        b = pivots.at[b_idx, "P"]

        pos_after_b = get_next_index(pivots.index, b_idx)

        if pos_after_b >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[pos_after_b] :, "P"].idxmin()
        d = pivots.at[d_idx, "P"]

        c_idx = pivots.loc[b_idx:d_idx, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

            if isinstance(d, pd.Series):
                d = pivots.at[d_idx, "P"].min()

        if is_bullish_vcp(a, b, c, d, e, avgBarLength):
            # check if Level C has been breached after it was formed
            if (
                c_idx != df.loc[c_idx:, "Close"].idxmax()
                or d_idx != df.loc[d_idx:, "Close"].idxmin()
            ):
                # Level C is breached, current pattern is not valid
                # check if C is the last pivot formed
                if pivots.index[-1] == c_idx or pivots.index[-1] == d_idx:
                    break

                # continue search for patterns
                a_idx, a = c_idx, c
                continue

            logger.debug(f"{sym} - VCPU")

            return dict(
                sym=sym,
                pattern="VCPU",
                start=a_idx,
                end=e_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                points=dict(
                    A=(a_idx, a),
                    B=(b_idx, b),
                    C=(c_idx, c),
                    D=(d_idx, d),
                    E=(e_idx, e),
                ),
                extra_points=dict(direction=(c_idx, c)),
            )

        a_idx, a = c_idx, c


def find_bearish_vcp(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Volatilty Contraction Pattern Bearish.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    a_idx = pivots["P"].idxmin()
    a = pivots.at[a_idx, "P"]

    assert isinstance(a_idx, pd.Timestamp)

    e_idx = df.index[-1]
    e = df.at[e_idx, "Close"]

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        b_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()

        b = pivots.at[b_idx, "P"]

        pos_after_b = get_next_index(pivots.index, b_idx)

        if pos_after_b >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[pos_after_b] :, "P"].idxmax()
        d = pivots.at[d_idx, "P"]

        c_idx = pivots.loc[b_idx:d_idx, "P"].idxmin()
        c = pivots.at[c_idx, "P"]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

            if isinstance(d, pd.Series):
                d = pivots.at[d_idx, "P"].max()

        if is_bearish_vcp(a, b, c, d, e, avgBarLength):
            if (
                d_idx != df.loc[d_idx:, "Close"].idxmax()
                or c_idx != df.loc[c_idx:, "Close"].idxmin()
            ):
                # check that the pattern is well formed
                if pivots.index[-1] == d_idx or pivots.index[-1] == c_idx:
                    break

                a_idx, a = c_idx, c
                continue

            logger.debug(f"{sym} - VCPD")

            return dict(
                sym=sym,
                pattern="VCPD",
                start=a_idx,
                end=e_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                points=dict(
                    A=(a_idx, a),
                    B=(b_idx, b),
                    C=(c_idx, c),
                    D=(d_idx, d),
                    E=(e_idx, e),
                ),
                extra_points=dict(direction=(c_idx, c)),
            )

        # We assign pivot level C to be the new A
        # This may not be the lowest pivot, so additional checks are required.
        a_idx, a = c_idx, c


def find_double_bottom(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Double bottom.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]

    a_idx = pivots["P"].idxmin()
    a = pivots.loc[a_idx, "P"]
    aVol = pivots.loc[a_idx, "V"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    atr_ser = get_atr(df.High, df.Low, df.Close)

    assert isinstance(a_idx, pd.Timestamp)

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        c = pivots.at[c_idx, "P"]
        cVol = pivots.at[c_idx, "V"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        atr = atr_ser.at[c_idx]

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(aVol, pd.Series):
                aVol = pivots.at[a_idx, "V"].iloc[0]

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

            if isinstance(cVol, pd.Series):
                cVol = pivots.at[c_idx, "V"].iloc[0]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        if is_double_bottom(a, b, c, d, aVol, cVol, avgBarLength, atr):
            if (
                a == df.at[a_idx, "High"]
                or b == df.at[b_idx, "Low"]
                or c == df.at[c_idx, "High"]
            ):
                # check that the pattern is well formed
                a_idx, a, aVol = c_idx, c, cVol
                continue

            # check if Level C has been breached after it was formed
            if (
                c_idx != df.loc[c_idx:, "Close"].idxmin()
                or b_idx != df.loc[b_idx:, "Close"].idxmax()
            ):
                a_idx, a, aVol = c_idx, c, cVol
                continue

            if df.loc[c_idx:, "Close"].max() > b:
                a_idx, a, aVol = c_idx, c, cVol
                continue

            logger.debug(f"{sym} - DBOT")

            return dict(
                sym=sym,
                pattern="DBOT",
                start=a_idx,
                end=d_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                points=dict(A=(a_idx, a), B=(b_idx, b), C=(c_idx, c), D=(d_idx, d)),
                extra_points=dict(direction=(b_idx, b)),
            )

        a_idx, a, aVol = c_idx, c, cVol


def find_double_top(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Double Top.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]

    a_idx = pivots["P"].idxmax()
    a = pivots.loc[a_idx, "P"]
    aVol = pivots.loc[a_idx, "V"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    atr_ser = get_atr(df.High, df.Low, df.Close)

    assert isinstance(a_idx, pd.Timestamp)

    while True:
        idx = get_next_index(pivots.index, a_idx)

        if idx >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[idx] :, "P"].idxmax()
        c = pivots.loc[c_idx, "P"]
        cVol = pivots.loc[c_idx, "V"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        atr = atr_ser.at[c_idx]

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(aVol, pd.Series):
                aVol = pivots.at[a_idx, "V"].iloc[0]

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

            if isinstance(cVol, pd.Series):
                cVol = pivots.at[c_idx, "V"].iloc[0]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        if is_double_top(a, b, c, d, aVol, cVol, avgBarLength, atr):
            if (
                a == df.at[a_idx, "Low"]
                or b == df.at[b_idx, "High"]
                or c == df.at[c_idx, "Low"]
            ):
                a_idx, a, aVol = c_idx, c, cVol
                continue

            # check if Level C has been breached after it was formed
            if (
                c_idx != df.loc[c_idx:, "Close"].idxmax()
                or b_idx != df.loc[b_idx:, "Close"].idxmin()
            ):
                # Level C is breached, current pattern is not valid
                a_idx, a, aVol = c_idx, c, cVol
                continue

            logger.debug(f"{sym} - DTOP")

            return dict(
                sym=sym,
                pattern="DTOP",
                start=a_idx,
                end=d_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                points=dict(A=(a_idx, a), B=(b_idx, b), C=(c_idx, c), D=(d_idx, d)),
                extra_points=dict(direction=(b_idx, b)),
            )

        a_idx, a, aVol = c_idx, c, cVol


def find_triangles(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Triangles - Symmetric, Ascending, Descending.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    a_idx = pivots["P"].idxmax()
    a = pivots.loc[a_idx, "P"]

    f_idx = df.index[-1]
    f = df.at[f_idx, "Close"]

    while True:
        b_idx = pivots.loc[a_idx:, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        assert isinstance(a_idx, pd.Timestamp)

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        # A is already the lowest point
        if a_idx == b_idx:
            a_idx = pivots.index[pos_after_a]
            a = pivots.at[a_idx, "P"]
            continue

        pos_after_b = get_next_index(pivots.index, b_idx)

        if pos_after_b >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_b] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        pos_after_c = get_next_index(pivots.index, c_idx)

        if pos_after_c >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[pos_after_c] :, "P"].idxmin()
        d = pivots.at[d_idx, "P"]

        pos_after_d = get_next_index(pivots.index, d_idx)

        if pos_after_d >= pivot_len:
            break

        e_idx = pivots.loc[pivots.index[pos_after_d] :, "P"].idxmax()
        e = pivots.at[e_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

            if isinstance(d, pd.Series):
                d = pivots.at[d_idx, "P"].min()

            if isinstance(e, pd.Series):
                e = pivots.at[e_idx, "P"].max()

        df_slice = df.loc[a_idx:d_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        triangle = is_triangle(a, b, c, d, e, f, avgBarLength)

        if triangle is not None:
            # Check if A is indeed the pivot high
            if (
                a == df.at[a_idx, "Low"]
                or b == df.at[b_idx, "High"]
                or c == df.at[c_idx, "Low"]
                or d == df.at[d_idx, "High"]
                or e == df.at[e_idx, "Low"]
            ):
                a_idx, a = c_idx, c
                continue

            upper_duration = (f_idx - a_idx).days
            lower_duration = (f_idx - b_idx).days

            if (
                max(upper_duration, lower_duration)
                / min(upper_duration, lower_duration)
                > 1.8
            ):
                # Ensure a 2:1 ratio in start duration of upper and lower lines
                a_idx, a = c_idx, c
                continue

            upper = generate_trend_line(df.High, a_idx, c_idx)
            lower = generate_trend_line(df.Low, b_idx, d_idx)

            # If trendlines have intersected, pattern has played out
            if upper.line.end.y < lower.line.end.y:
                break

            if triangle == "Ascending" and (upper.slope > 0.1 and lower.slope < 0.2):
                break

            if triangle == "Descending" and (lower.slope < -0.1 and upper.slope > -0.2):
                break

            if triangle == "Symmetric" and (upper.slope > -0.2 and lower.slope < 0.2):
                break

            # Check if trendlines have been breached
            pos = df.reset_index().index

            # calculate the y-axis price for every point on the slope
            upper_slope = upper.slope * pos + upper.y_int
            lower_slope = lower.slope * pos + lower.y_int

            # Check if close has violated the upper or lower trendline
            if (df.Close > upper_slope).any() or (df.Close < lower_slope).any():
                break

            logger.debug(f"{sym} - {triangle}")

            return dict(
                sym=sym,
                pattern="TRNG",
                alt_name=triangle,
                start=a_idx,
                end=f_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                slope_upper=upper.slope,
                slope_lower=lower.slope,
                points=dict(
                    A=(a_idx, a),
                    B=(b_idx, b),
                    C=(c_idx, c),
                    D=(d_idx, d),
                    E=(e_idx, e),
                    F=(f_idx, f),
                ),
                extra_points=dict(
                    upper_start=upper.line.start,
                    upper_end=upper.line.end,
                    lower_start=lower.line.start,
                    lower_end=lower.line.end,
                ),
            )

        a_idx, c = c_idx, c


def find_hns(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Head and Shoulders - Bearish

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    f_idx = df.index[-1]
    f = df.at[f_idx, "Close"]

    c_idx = pivots["P"].idxmax()
    c = pivots.at[c_idx, "P"]

    assert isinstance(c_idx, pd.Timestamp)

    while True:
        pos = get_prev_index(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_before_c = pivots.index[pos]

        a_idx = pivots.loc[:idx_before_c, "P"].idxmax()
        a = pivots.at[a_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        pos = get_next_index(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_after_c = pivots.index[pos]

        e_idx = pivots.loc[idx_after_c:, "P"].idxmax()
        e = pivots.at[e_idx, "P"]

        d_idx = pivots.loc[c_idx:e_idx, "P"].idxmin()
        d = pivots.at[d_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

            if isinstance(d, pd.Series):
                d = pivots.at[d_idx, "P"].min()

            if isinstance(e, pd.Series):
                e = pivots.at[e_idx, "P"].max()

        df_slice = df.loc[b_idx:d_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        if is_hns(a, b, c, d, e, f, avgBarLength):
            if (
                a == df.at[a_idx, "Low"]
                or b == df.at[b_idx, "High"]
                or c == df.at[c_idx, "Low"]
                or d == df.at[d_idx, "High"]
                or e == df.at[e_idx, "Low"]
            ):
                # Make sure pattern is well formed and
                # pivots are correctly anchored to highs and lows
                c_idx, c = e_idx, e
                continue

            neckline_price = min(b, d)
            lowest_after_e = df.loc[e_idx:, "Low"].min()

            if (
                lowest_after_e < neckline_price
                and abs(lowest_after_e - neckline_price) > avgBarLength
            ):
                # check if neckline was breached after pattern formation
                c_idx, c = e_idx, e
                continue

            # bd is the line coordinate for points B and D
            tline = generate_trend_line(df.Low, b_idx, d_idx)

            # Get the y coordinate of the trendline at the end of the chart
            # With the given slope(m) and y-intercept(b) as y_int,
            # Get the x coordinate (index position of last date in DataFrame)
            # and calculate value of y coordinate using y = mx + b
            x = df.index.get_loc(f_idx)

            assert isinstance(x, int)

            y = tline.slope * x + tline.y_int

            # if the close price is below the neckline (trendline), skip
            if f < y:
                c_idx, c = e_idx, e
                continue

            logger.debug(f"{sym} - HNSD")

            return dict(
                sym=sym,
                pattern="HNSD",
                start=a_idx,
                end=f_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                slope=tline.slope,
                y_intercept=tline.y_int,
                points=dict(
                    A=(a_idx, a),
                    B=(b_idx, b),
                    C=(c_idx, c),
                    D=(d_idx, d),
                    E=(e_idx, e),
                    F=(f_idx, f),
                ),
                extra_points=dict(direction=(b_idx, b)),
            )

        c_idx, c = e_idx, e


def find_reverse_hns(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Find Head and Shoulders - Bullish

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    f_idx = df.index[-1]
    f = df.at[f_idx, "Close"]

    c_idx = pivots["P"].idxmin()
    c = pivots.at[c_idx, "P"]

    assert isinstance(c_idx, pd.Timestamp)

    while True:
        pos = get_prev_index(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_before_c = pivots.index[pos]

        a_idx = pivots.loc[:idx_before_c, "P"].idxmin()
        a = pivots.at[a_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        pos = get_next_index(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_after_c = pivots.index[pos]

        e_idx = pivots.loc[idx_after_c:, "P"].idxmin()
        e = pivots.at[e_idx, "P"]

        d_idx = pivots.loc[c_idx:e_idx, "P"].idxmax()
        d = pivots.at[d_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.loc[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.loc[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.loc[c_idx, "P"].min()

            if isinstance(d, pd.Series):
                d = pivots.loc[d_idx, "P"].max()

            if isinstance(e, pd.Series):
                e = pivots.loc[e_idx, "P"].min()

        df_slice = df.loc[b_idx:d_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).median()

        if is_reverse_hns(a, b, c, d, e, f, avgBarLength):
            if (
                a == df.at[a_idx, "High"]
                or b == df.at[b_idx, "Low"]
                or c == df.at[c_idx, "High"]
                or d == df.at[d_idx, "Low"]
                or e == df.at[e_idx, "High"]
            ):
                # Make sure pattern is well formed
                c_idx, c = e_idx, e
                continue

            neckline_price = min(b, d)

            highest_after_e = df.loc[e_idx:, "High"].max()

            if (
                highest_after_e > neckline_price
                and abs(highest_after_e - neckline_price) > avgBarLength
            ):
                # check if neckline was breached after pattern formation
                c_idx, c = e_idx, e
                continue

            # bd is the trendline coordinates from B to D (neckline)
            tline = generate_trend_line(df.High, b_idx, d_idx)

            # Get the y coordinate of the trendline at the end of the chart
            # With the given slope(m) and y-intercept(b) as y_int,
            # Get the x coordinate (index position of last date in DataFrame)
            # and calculate value of y coordinate using y = mx + b
            x = df.index.get_loc(df.index[-1])

            assert isinstance(x, int)

            y = tline.slope * x + tline.y_int

            # if close price is greater than neckline (trendline), skip
            if f > y:
                c_idx, c = e_idx, e
                continue

            logger.debug(f"{sym} - HNSU")

            return dict(
                sym=sym,
                pattern="HNSU",
                start=a_idx,
                end=f_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                slope=tline.line,
                y_intercept=tline.y_int,
                points=dict(
                    A=(a_idx, a),
                    B=(b_idx, b),
                    C=(c_idx, c),
                    D=(d_idx, d),
                    E=(e_idx, e),
                    F=(f_idx, f),
                ),
                extra_points=dict(direction=(b_idx, b)),
            )

        c_idx, c = e_idx, e


def find_downtrend_line(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Downtrend line detection"""

    selected: Optional[dict] = None

    pivots_len = len(pivots)

    if not pivots_len:
        return

    # Get the highest point in pivots.
    a_idx = pivots.P.idxmax()
    a = pivots.at[a_idx, "P"]

    if isinstance(a, pd.Series):
        a = a.max()

    threshold = a * 0.001
    last_idx = df.index[-1]

    # A is the last pivot
    if a_idx == pivots.index[-1]:
        return None

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(a_idx, pd.Timestamp)

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivots_len:
            break

        # Get the next highest point in pivots after A
        b_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        if isinstance(b, pd.Series):
            b = b.max()

        # Calculate the slope and y-intercept of the trendline AB.
        try:
            tline = generate_trend_line(df.High, a_idx, b_idx)
        except ZeroDivisionError:
            return

        y_close = getY(tline.slope, tline.y_int, len(df) - 1)

        close = df.at[last_idx, "Close"]

        pct_close = (close - y_close) / y_close * 100

        if close > y_close or pct_close < -10:
            a_idx, a = b_idx, b
            continue

        line_pivots = pivots.loc[a_idx:, "P"]

        # Calculate y values for trendline at each pivot.
        y_values = line_pivots.index.map(
            lambda x: getY(tline.slope, tline.y_int, df.index.get_loc(x))
        )

        # Get the absolute distance of each pivot from trendline.
        diff = (line_pivots - y_values).abs()

        # Count pivots with distance from line, within a fixed threshold.
        touch_count = (diff <= threshold).sum()

        if touch_count > 2:
            closes = df.loc[a_idx:last_idx, "Close"]

            # Calculate price at each point on the trendline
            y_values = closes.index.map(
                lambda x: getY(tline.slope, tline.y_int, df.index.get_loc(x))
            )

            # Sum up number of times, price closed above the trendline
            if (closes > y_values).sum() > 0:
                # skip if trendline was breached
                a_idx, a = b_idx, b
                continue

            # Filter the distances for pivots located above the trendline
            # and use the absolute sum of their distances as the score.
            # For two lines with equal touch points, the lower score indicates
            # a better fitting line.
            # The lowest possible score is 0. Sum of empty Series is 0
            score = diff[diff < threshold].sum()

            # Update if no trendline is detected yet or
            # if we have higher touch counts.
            # if touch count is same, check for lower scores
            if selected is None or (
                touch_count > selected["touches"]
                or (touch_count == selected["touches"] and score < selected["score"])
            ):
                touch_points = line_pivots.loc[diff <= threshold]
                str_keys = ascii_upper[: len(touch_points)]

                selected = dict(
                    touches=touch_count,
                    start=a_idx,
                    end=last_idx,
                    slope=tline.slope,
                    y_intercept=tline.y_int,
                    y_close=y_close,
                    points=dict(zip(str_keys, tuple(touch_points.items()))),
                    extra_points=dict(start=tline.line.start, end=tline.line.end),
                    score=score,
                )

        a_idx, a = b_idx, b

    if selected:
        selected.update(
            dict(
                sym=sym,
                pattern="DNTL",
                df_start=df.index[0],
                df_end=last_idx,
            )
        )

    return selected


def find_uptrend_line(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """Uptrend line detection"""

    selected: Optional[dict] = None
    pivots_len = len(pivots)

    if not pivots_len:
        return

    # Get the lowest point in pivots.
    a_idx = pivots.P.idxmin()
    a = pivots.at[a_idx, "P"]

    if isinstance(a, pd.Series):
        a = a.min()

    threshold = a * 0.001
    last_idx = df.index[-1]

    # A is the last pivot
    if a_idx == pivots.index[-1]:
        return

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(a_idx, pd.Timestamp)

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivots_len:
            break

        # Get the next lowest point in pivots.
        b_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        if isinstance(b, pd.Series):
            b = b.max()

        # Calculate the slope and y-intercept of the trendline AB.
        try:
            tline = generate_trend_line(df.Low, a_idx, b_idx)
        except ZeroDivisionError:
            return

        y_close = getY(tline.slope, tline.y_int, len(df) - 1)

        close = df.at[last_idx, "Close"]

        pct_close = (close - y_close) / y_close * 100

        if close < y_close or pct_close > 10:
            a_idx, a = b_idx, b
            continue

        line_pivots = pivots.loc[a_idx:, "P"]

        # Calculate y values for trendline at each pivot.
        y_values = line_pivots.index.map(
            lambda x: getY(tline.slope, tline.y_int, df.index.get_loc(x))
        )

        # Get the distance of each pivot from trendline.
        diff = (line_pivots - y_values).abs()

        # Count pivots, whose absolute distance from line is
        # within a fixed threshold.
        touch_count = (diff <= threshold).sum()

        if touch_count > 2:
            # Get all closes from line start to last close
            closes = df.loc[a_idx:last_idx, "Close"]

            # Calculate the price on trendline from line start to last close
            y_values = closes.index.map(
                lambda x: getY(tline.slope, tline.y_int, df.index.get_loc(x))
            )

            # Sum up number of times, price closed below the trendline
            if (closes < y_values).sum() > 0:
                # skip if trendline was breached
                a_idx, a = b_idx, b
                continue

            # Filter the distances for pivots located below the trendline
            # and use the absolute sum of their distances as the score.
            # For two lines with equal touch points, the lower score indicates
            # a better fitting line.
            # The lowest possible score is 0. Sum of empty Series is 0
            score = diff[diff < -threshold].sum()

            # Update if no trendline is detected yet or
            # if we have higher touch counts.
            # if touch count is same, check for lower scores
            if selected is None or (
                touch_count > selected["touches"]
                or (touch_count == selected["touches"] and score < selected["score"])
            ):
                touch_points = line_pivots.loc[diff <= threshold]
                str_keys = ascii_upper[: len(touch_points)]

                selected = dict(
                    touches=touch_count,
                    start=a_idx,
                    end=last_idx,
                    slope=tline.slope,
                    y_intercept=tline.y_int,
                    y_close=y_close,
                    points=dict(zip(str_keys, tuple(touch_points.items()))),
                    extra_points=dict(start=tline.line.start, end=tline.line.end),
                    score=score,
                )

        a_idx, a = b_idx, b

    if selected:
        selected.update(
            dict(
                sym=sym,
                pattern="UPTL",
                df_start=df.index[0],
                df_end=last_idx,
            )
        )

    return selected


def find_bullish_abcd(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bullish AB = CD harmonic pattern
    """

    alt_name = "Bull AB=CD"
    pivot_len = pivots.shape[0]

    a_idx = pivots["P"].idxmax()
    a = pivots.at[a_idx, "P"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(a_idx, pd.Timestamp)

    selected: Optional[dict] = None

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        if b_idx == c_idx:
            break

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

        bc_diff = c - b
        ab_diff = a - b

        lowest_low_ac = df.loc[a_idx:c_idx, "Low"].min()
        highest_high_from_b = df.loc[b_idx:, "High"].max()

        if (
            lowest_low_ac != b
            or highest_high_from_b != c
            or a == df.at[a_idx, "Low"]
            or b == df.at[b_idx, "High"]
            or c == df.at[c_idx, "Low"]
        ):
            # Check that the pattern is well formed
            a, a_idx = c, c_idx
            continue

        # Get the FIB ratio nearest to point C
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        if c_retrace < 0.382 or c_retrace > 0.886:
            a, a_idx = c, c_idx
            continue

        c_fib_inverse = 1 / c_retrace

        ab_cd_ext = c - ab_diff
        bc_ext = c - bc_diff * c_fib_inverse
        bc_618_ext = c - bc_diff * 1.618
        ab_27_ext = c - ab_diff * 1.27
        ab_618_ext = c - ab_diff * 1.618

        lowest_close_from_c = df.loc[c_idx:, "Close"].min()

        is_perfect = c_retrace == 0.618 and ab_cd_ext <= bc_618_ext

        is_alternate = lowest_close_from_c < ab_cd_ext

        terminal_point = ab_cd_ext

        if is_perfect:
            terminal_point = ab_cd_ext
        elif is_alternate:
            terminal_point = ab_618_ext

        lows_after_c = df.loc[c_idx:, "Low"]

        lows_below_terminal_point = lows_after_c.loc[lows_after_c < terminal_point]

        if lows_below_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_below_terminal_point = (df.loc[c_idx:, "Close"] < terminal_point).sum()

        ab_completion = (b_idx - a_idx).days
        cd_completion = (d_idx - c_idx).days

        if (
            d < b - (b - terminal_point) * 0.5
            and cd_completion < ab_completion * 2
            and closes_below_terminal_point < 7
            and (
                has_tested
                and (d_idx - lows_below_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "A": (a_idx, a),
                    "B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                alt_name = "Bull Perfect AB=CD"

                selected["extra_points"].update(
                    {
                        "AB=CD": (b_idx, ab_cd_ext),
                        "1.618BC": (b_idx, bc_618_ext),
                    }
                )
            else:
                if is_alternate:
                    alt_name = "Bull Alternate AB=CD"

                    selected["extra_points"].update(
                        {
                            "1.27AB=CD": (b_idx, ab_27_ext),
                            "1.618AB=CD": (b_idx, ab_618_ext),
                        }
                    )

                selected["extra_points"].update(
                    {
                        f"{c_fib_inverse:.3f}BC": (b_idx, bc_ext),
                        "AB=CD": (b_idx, ab_cd_ext),
                    }
                )

        a, a_idx = c, c_idx

    if selected:
        selected.update(
            dict(
                sym=sym,
                pattern="ABCDU",
                alt_name=alt_name,
            )
        )

    return selected


def find_bearish_abcd(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bearish AB = CD harmonic pattern
    """

    alt_name = "Bear AB=CD"
    pivot_len = pivots.shape[0]

    a_idx = pivots["P"].idxmin()
    a = pivots.at[a_idx, "P"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(a_idx, pd.Timestamp)

    selected: Optional[dict] = None

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()

        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        if b_idx == c_idx:
            break

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

        bc_diff = b - c
        ab_diff = b - a

        highest_high_ac = df.loc[a_idx:c_idx, "High"].max()
        lowest_low_from_b = df.loc[b_idx:, "Low"].min()

        if (
            highest_high_ac != b
            or lowest_low_from_b != c
            or a == df.at[a_idx, "High"]
            or b == df.at[b_idx, "Low"]
            or c == df.at[c_idx, "High"]
        ):
            # Check that the pattern is well formed
            a, a_idx = c, c_idx
            continue

        # Get the FIB ratio nearest to point C
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        if c_retrace < 0.382 or c_retrace > 0.886:
            a, a_idx = c, c_idx
            continue

        c_fib_inverse = 1 / c_retrace

        ab_cd_ext = c + ab_diff
        bc_ext = c + bc_diff * c_fib_inverse
        bc_618_ext = c + bc_diff * 1.618
        ab_27_ext = c + ab_diff * 1.27
        ab_618_ext = c + ab_diff * 1.618

        highest_close_after_c = df.loc[c_idx:, "Close"].max()

        is_perfect = c_retrace == 0.618 and ab_cd_ext >= bc_618_ext
        is_alternate = highest_close_after_c > ab_cd_ext

        terminal_point = ab_cd_ext

        if is_perfect:
            terminal_point = ab_cd_ext
        elif is_alternate:
            terminal_point = ab_618_ext

        highs_after_c = df.loc[c_idx:, "High"]

        highs_above_terminal_point = highs_after_c[highs_after_c > terminal_point]

        if highs_above_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_above_terminal_point = (df.loc[c_idx:, "Close"] > terminal_point).sum()

        ab_completion = (b_idx - a_idx).days
        cd_completion = (d_idx - c_idx).days

        if (
            closes_above_terminal_point < 7
            and cd_completion < ab_completion * 2
            and d > b + (terminal_point - b) * 0.5
            and (
                has_tested
                and (d_idx - highs_above_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "A": (a_idx, a),
                    "B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                alt_name = "Bear Perfect AB=CD"

                selected["extra_points"].update(
                    {
                        "AB=CD": (b_idx, ab_cd_ext),
                        "1.618BC": (b_idx, bc_618_ext),
                    }
                )
            else:
                if is_alternate:
                    alt_name = "Bear Alternate AB=CD"

                    selected["extra_points"].update(
                        {
                            "1.27AB=CD": (b_idx, ab_27_ext),
                            "1.618AB=CD": (b_idx, ab_618_ext),
                        }
                    )

                selected["extra_points"].update(
                    {
                        f"{c_fib_inverse:.3f}BC": (b_idx, bc_ext),
                        "AB=CD": (b_idx, ab_cd_ext),
                    }
                )

        a, a_idx = c, c_idx

    if selected:
        selected.update(
            dict(
                sym=sym,
                pattern="ABCDD",
                alt_name=alt_name,
            )
        )

    return selected


def find_bullish_bat(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bullish Bat harmonic pattern
    """
    alt_name = "Bull BAT"
    pivot_len = pivots.shape[0]

    x_idx = pivots["P"].idxmin()
    x = pivots.at[x_idx, "P"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(x_idx, pd.Timestamp)

    while True:
        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
        a = pivots.at[a_idx, "P"]

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].min()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

        xa_diff = a - x
        ab_diff = a - b
        bc_diff = c - b

        lowest_low_ac = df.loc[a_idx:c_idx, "Low"].min()
        highest_high_xb = df.loc[x_idx:b_idx, "High"].max()
        highest_high_from_b = df.loc[b_idx:, "High"].max()

        if (
            highest_high_xb != a
            or lowest_low_ac != b
            or highest_high_from_b != c
            or x == df.at[x_idx, "High"]
            or a == df.at[a_idx, "Low"]
            or b == df.at[b_idx, "High"]
            or c == df.at[c_idx, "Low"]
        ):
            # Check that the pattern is well formed
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
            x = pivots.loc[x_idx, "P"]
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect = b_retrace == 0.5 and (c_retrace == 0.5 or c_retrace == 0.618)

        is_alternate = b_retrace == 0.382

        if (
            b_retrace < 0.382
            or b_retrace > 0.5
            or c_retrace < 0.382
            or c_retrace > 0.886
            or not is_alternate
            and df.loc[c_idx:, "Close"].min() < x
        ):
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
            x = pivots.loc[x_idx, "P"]
            continue

        xa_886_retrace = a - xa_diff * 0.886
        xa_13_ext = a - xa_diff * 1.13

        bc_2_ext = c - bc_diff * 2
        bc_618_ext = c - bc_diff * 1.618

        ab_27_ext = c - ab_diff * 1.27

        terminal_point = xa_13_ext if is_alternate else xa_886_retrace

        lows_from_c = df.loc[c_idx:, "Low"]

        lows_below_terminal_point = lows_from_c.loc[lows_from_c < terminal_point]

        if lows_below_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_below_terminal_point = (df.loc[c_idx:, "Close"] < terminal_point).sum()

        if (
            closes_below_terminal_point < 7
            and d < b - (b - terminal_point) * 0.5
            and (
                has_tested
                and (d_idx - lows_below_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                # Perfect BAT pattern
                alt_name = "Bull Perfect BAT"

                selected["extra_points"].update(
                    {
                        "1.27AB=CD": (b_idx, ab_27_ext),
                        "2BC": (b_idx, bc_2_ext),
                        "0.886XA": (b_idx, xa_886_retrace),
                    }
                )
            else:
                if is_alternate:
                    # Alternate BAT pattern
                    alt_name = "Bull Alternate BAT"

                    # alternate Bat reversal levels
                    clustered_levels = get_relative_clusters(
                        {
                            "1.13XA": xa_13_ext,
                            "2BC": bc_2_ext,
                            "2.618BC": c - bc_diff * 2.618,
                            "3BC": c - bc_diff * 3,
                            "3.618BC": c - bc_diff * 3.618,
                            "1.618AB=CD": c - ab_diff * 1.618,
                        },
                        "1.13XA",
                    )

                    selected["extra_points"].update(
                        {
                            level: (b_idx, price)
                            for level, price in clustered_levels.items()
                            if price <= x
                        }
                    )

                # Bat reversal levels
                clustered_levels = get_relative_clusters(
                    {
                        "0.886XA": xa_886_retrace,
                        "1.618BC": bc_618_ext,
                        "2BC": bc_2_ext,
                        "2.618BC": c - bc_diff * 2.618,
                    },
                    "0.886XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price > x
                    }
                )

        x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
        x = pivots.loc[x_idx, "P"]

    if selected:
        selected.update(dict(sym=sym, pattern="BATU", alt_name=alt_name))

    return selected


def find_bearish_bat(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bearish Bat harmonic pattern
    """
    alt_name = "Bear BAT"
    pivot_len = pivots.shape[0]

    x_idx = pivots["P"].idxmax()
    x = pivots.at[x_idx, "P"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(x_idx, pd.Timestamp)

    while True:
        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
        a = pivots.at[a_idx, "P"]

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].max()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

        xa_diff = x - a
        ab_diff = b - a
        bc_diff = b - c

        lowest_low_from_b = df.loc[b_idx:, "Low"].min()
        highest_high_ac = df.loc[a_idx:c_idx, "High"].max()
        lowest_low_xb = df.loc[x_idx:b_idx, "Low"].min()

        if (
            lowest_low_xb != a
            or highest_high_ac != b
            or lowest_low_from_b != c
            or x == df.at[x_idx, "Low"]
            or a == df.at[a_idx, "High"]
            or b == df.at[b_idx, "Low"]
            or c == df.at[c_idx, "High"]
        ):
            # Check that the pattern is well formed
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
            x = pivots.loc[x_idx, "P"]
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect = b_retrace == 0.5 and (c_retrace == 0.5 or c_retrace == 0.618)

        is_alternate = b_retrace == 0.382

        if (
            b_retrace < 0.382
            or b_retrace > 0.5
            or c_retrace < 0.382
            or c_retrace > 0.886
            or not is_alternate
            and df.loc[c_idx:, "Close"].max() > x
        ):
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
            x = pivots.loc[x_idx, "P"]
            continue

        xa_886_retrace = a + xa_diff * 0.886
        xa_13_ext = a + xa_diff * 1.13

        bc_2_ext = c + bc_diff * 2
        bc_618_ext = c + bc_diff * 1.618

        ab_27_ext = c + ab_diff * 1.27

        terminal_point = xa_13_ext if is_alternate else xa_886_retrace

        highs_from_c = df.loc[c_idx:, "High"]

        highs_above_terminal_point = highs_from_c.loc[highs_from_c > terminal_point]

        if highs_above_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_above_terminal_point = (df.loc[c_idx:, "Close"] > terminal_point).sum()

        if (
            closes_above_terminal_point < 7
            and d > b + (terminal_point - b) * 0.5
            and (
                has_tested
                and (d_idx - highs_above_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                # Perfect BAT pattern
                alt_name = "Bear Perfect BAT"

                selected["extra_points"].update(
                    {
                        "1.27AB=CD": (b_idx, ab_27_ext),
                        "2BC": (b_idx, bc_2_ext),
                        "0.886XA": (b_idx, xa_886_retrace),
                    }
                )
            else:
                if is_alternate:
                    # Alternate BAT pattern
                    alt_name = "Bear Alternate BAT"

                    clustered_levels = get_relative_clusters(
                        {
                            "1.13XA": xa_13_ext,
                            "1.618AB=CD": c + ab_diff * 1.618,
                            "2BC": bc_2_ext,
                            "2.618BC": c + bc_diff * 2.618,
                            "3BC": c + bc_diff * 3,
                            "3.618BC": c + bc_diff * 3.618,
                        },
                        "1.13XA",
                    )

                    selected["extra_points"].update(
                        {
                            level: (b_idx, price)
                            for level, price in clustered_levels.items()
                            if price >= x
                        }
                    )

                clustered_levels = get_relative_clusters(
                    {
                        "0.886XA": xa_886_retrace,
                        "1.618BC": bc_618_ext,
                        "2BC": bc_2_ext,
                        "2.618BC": c + bc_diff * 2.618,
                    },
                    "0.886XA",
                )

                # Normal BAT pattern
                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price < x
                    }
                )

        x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
        x = pivots.loc[x_idx, "P"]

    if selected:
        selected.update(dict(sym=sym, pattern="BATD", alt_name=alt_name))

    return selected


def find_bullish_gartley(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bullish Gartley harmonic pattern
    """
    alt_name = "Bull Gartley"
    pivot_len = pivots.shape[0]

    x_idx = pivots["P"].idxmin()
    x = pivots.at[x_idx, "P"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(x_idx, pd.Timestamp)

    while True:
        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
        a = pivots.at[a_idx, "P"]

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].min()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

        xa_diff = a - x
        ab_diff = a - b
        bc_diff = c - b

        highest_high_xb = df.loc[x_idx:b_idx, "High"].max()
        lowest_low_ac = df.loc[a_idx:c_idx, "Low"].min()
        highest_high_from_b = df.loc[b_idx:, "High"].max()
        lowest_close_from_c = df.loc[c_idx:, "Close"].min()

        if (
            highest_high_xb != a
            or lowest_low_ac != b
            or highest_high_from_b != c
            or lowest_close_from_c < x
            or x == df.at[x_idx, "High"]
            or a == df.at[a_idx, "Low"]
            or b == df.at[b_idx, "High"]
            or c == df.at[c_idx, "Low"]
        ):
            # Check that the pattern is well formed
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
            x = pivots.loc[x_idx, "P"]
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect = b_retrace == 0.618 and c_retrace == 0.618

        if b_retrace != 0.618 or c_retrace < 0.382 or c_retrace > 0.886:
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
            x = pivots.loc[x_idx, "P"]
            continue

        c_fib_inverse = round(1 / c_retrace, 3)

        xa_786_retrace = a - xa_diff * 0.786

        bc_618_ext = c - bc_diff * 1.618

        ab_cd_ext = c - ab_diff

        terminal_point = xa_786_retrace

        closes_below_terminal_point = (df.loc[c_idx:, "Close"] < terminal_point).sum()

        lows_after_c = df.loc[c_idx:, "Low"]

        lows_below_terminal_point = lows_after_c.loc[lows_after_c < terminal_point]

        if lows_below_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        if (
            d < b
            and closes_below_terminal_point < 7
            and (
                has_tested
                and (d_idx - lows_below_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                # Perfect Gartley pattern
                alt_name = "Bull Perfect Gartley"

                clustered_levels = get_relative_clusters(
                    {
                        "AB=CD": ab_cd_ext,
                        "0.786XA": xa_786_retrace,
                        "1.618BC": bc_618_ext,
                    },
                    "0.786XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price > x
                    }
                )
            else:
                clustered_levels = get_relative_clusters(
                    {
                        "AB=CD": ab_cd_ext,
                        "0.786XA": xa_786_retrace,
                        f"{c_fib_inverse}BC": c - bc_diff * c_fib_inverse,
                    },
                    "0.786XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price > x
                    }
                )

        x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
        x = pivots.loc[x_idx, "P"]

    if selected:
        selected.update(dict(sym=sym, pattern="GARTU", alt_name=alt_name))

    return selected


def find_bearish_gartley(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bearish Gartley harmonic pattern
    """
    alt_name = "Bearish Gartley"
    pivot_len = pivots.shape[0]

    x_idx = pivots["P"].idxmax()
    x = pivots.at[x_idx, "P"]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)
    assert isinstance(x_idx, pd.Timestamp)

    while True:
        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
        a = pivots.at[a_idx, "P"]

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].max()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

        xa_diff = x - a
        ab_diff = b - a
        bc_diff = b - c

        lowest_low_xb = df.loc[x_idx:b_idx, "Low"].min()
        highest_high_ac = df.loc[a_idx:c_idx, "High"].max()
        lowest_low_from_c = df.loc[c_idx:, "Low"].min()
        highest_close_from_c = df.loc[c_idx:, "Close"].max()

        if (
            lowest_low_xb != a
            or highest_high_ac != b
            or lowest_low_from_c != c
            or highest_close_from_c > x
            or x == df.at[x_idx, "Low"]
            or a == df.at[a_idx, "High"]
            or b == df.at[b_idx, "Low"]
            or c == df.at[c_idx, "High"]
        ):
            # Check that the pattern is well formed
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
            x = pivots.loc[x_idx, "P"]
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect = b_retrace == 0.618 and c_retrace == 0.618

        if b_retrace != 0.618 or c_retrace < 0.382 or c_retrace > 0.886:
            x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
            x = pivots.loc[x_idx, "P"]
            continue

        c_fib_inverse = round(1 / c_retrace, 3)

        xa_786_retrace = a + xa_diff * 0.786

        bc_618_ext = c + bc_diff * 1.618

        ab_cd_ext = c + ab_diff

        terminal_point = xa_786_retrace

        highs_after_c = df.loc[c_idx:, "High"]

        highs_above_terminal_point = highs_after_c[highs_after_c > terminal_point]

        if highs_above_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_above_terminal_point = (df.loc[c_idx:, "Close"] > terminal_point).sum()

        if (
            d > b
            and closes_above_terminal_point < 7
            and (
                has_tested
                and (d_idx - highs_above_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                # Perfect Gartley pattern
                alt_name = "Bear Perfect Gartley"

                clustered_levels = get_relative_clusters(
                    {
                        "AB=CD": ab_cd_ext,
                        "0.786XA": xa_786_retrace,
                        "1.618BC": bc_618_ext,
                    },
                    "0.786XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price < x
                    }
                )
            else:
                clustered_levels = get_relative_clusters(
                    {
                        "AB=CD": ab_cd_ext,
                        "0.786XA": xa_786_retrace,
                        f"{c_fib_inverse}BC": c + bc_diff * c_fib_inverse,
                    },
                    "0.786XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price < x
                    }
                )

        x_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
        x = pivots.loc[x_idx, "P"]

    if selected:
        selected.update(dict(sym=sym, pattern="GARTD", alt_name=alt_name))

    return selected


def find_bullish_crab(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bullish Crab harmonic pattern
    """
    alt_name = "Bull Crab"
    pivot_len = pivots.shape[0]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)

    for x_idx in pivots.index:
        x = pivots.at[x_idx, "P"]

        if isinstance(x, pd.Series):
            x = pivots.at[x_idx, "P"].min()

        if x == df.at[x_idx, "High"]:
            continue

        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
        a = pivots.at[a_idx, "P"]

        if df.loc[x_idx:a_idx, "Low"].min() != x:
            continue

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].min()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

        xa_diff = a - x
        ab_diff = a - b
        bc_diff = c - b

        highest_high_xb = df.loc[x_idx:b_idx, "High"].max()
        lowest_low_ac = df.loc[a_idx:c_idx, "Low"].min()
        highest_high_from_b = df.loc[b_idx:, "High"].max()

        if (
            highest_high_xb != a
            or lowest_low_ac != b
            or highest_high_from_b != c
            or a == df.at[a_idx, "Low"]
            or b == df.at[b_idx, "High"]
            or c == df.at[c_idx, "Low"]
        ):
            # Check that the pattern is well formed
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect_crab = b_retrace == 0.618 and (
            c_retrace == 0.5 or c_retrace == 0.618
        )

        is_deep_crab = b_retrace == 0.886

        if (
            b_retrace > 0.618
            and not is_deep_crab
            or b_retrace < 0.382
            or (c_retrace < 0.382 or c_retrace > 0.886)
        ):
            continue

        xa_618_ext = a - xa_diff * 1.618

        bc_3_14_ext = c - bc_diff * 3.14
        ab_618_ext = c - ab_diff * 1.618
        ab_27_ext = c - ab_diff * 1.27

        terminal_point = xa_618_ext

        lows_after_c = df.loc[c_idx:, "Low"]

        lows_below_terminal_point = lows_after_c.loc[lows_after_c < terminal_point]

        if lows_below_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_below_terminal_point = (df.loc[c_idx:, "Close"] < terminal_point).sum()

        if (
            d < b - (b - terminal_point) * 0.5
            and closes_below_terminal_point < 7
            and (
                has_tested
                and (d_idx - lows_below_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect_crab:
                # Perfect crab pattern
                alt_name = "Bull Perfect Crab"

                clustered_levels = get_relative_clusters(
                    {
                        "3.14BC": bc_3_14_ext,
                        "1.618XA": xa_618_ext,
                        "1.618AB": ab_618_ext,
                    },
                    "1.618XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price < x
                    }
                )
            elif is_deep_crab:
                # Deep Crab pattern
                alt_name = "Bull Deep Crab"

                clustered_levels = get_relative_clusters(
                    {
                        "1.618XA": xa_618_ext,
                        "1.27AB": ab_27_ext,
                        "1.618AB": ab_618_ext,
                        "2.24BC": c - bc_diff * 2.24,
                        "2.618BC": c - bc_diff * 2.618,
                        "3.14BC": c - bc_diff * 3.14,
                        "3.618BC": c - bc_diff * 3.618,
                        "1.27AB": ab_27_ext,
                        "1.618AB": ab_618_ext,
                    },
                    "1.618XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price < x and price >= xa_618_ext
                    }
                )
            else:
                clustered_levels = get_relative_clusters(
                    {
                        "1.618XA": xa_618_ext,
                        "2.618BC": c - bc_diff * 2.618,
                        "3.14BC": c - bc_diff * 3.14,
                        "3.618BC": c - bc_diff * 3.618,
                        "1.27AB": ab_27_ext,
                        "1.618AB": ab_618_ext,
                    },
                    "1.618XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price < x
                    }
                )

    if selected:
        selected.update(dict(sym=sym, pattern="CRABU", alt_name=alt_name))

    return selected


def find_bearish_crab(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bearish Crab harmonic pattern
    """
    alt_name = "Bear Crab"
    pivot_len = pivots.shape[0]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)

    for x_idx in pivots.index:
        x = pivots.at[x_idx, "P"]

        if isinstance(x, pd.Series):
            x = pivots.at[x_idx, "P"].max()

        if x == df.at[x_idx, "Low"]:
            continue

        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
        a = pivots.at[a_idx, "P"]

        if df.loc[x_idx:a_idx, "High"].max() != x:
            continue

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].max()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

        xa_diff = x - a
        ab_diff = b - a
        bc_diff = b - c

        lowest_low_xb = df.loc[x_idx:b_idx, "Low"].min()
        highest_high_ac = df.loc[a_idx:c_idx, "High"].max()
        lowest_low_from_b = df.loc[b_idx:, "Low"].min()

        if (
            lowest_low_xb != a
            or highest_high_ac != b
            or lowest_low_from_b != c
            or a == df.at[a_idx, "High"]
            or b == df.at[b_idx, "Low"]
            or c == df.at[c_idx, "High"]
        ):
            # Check that the pattern is well formed
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect_crab = b_retrace == 0.618 and (
            c_retrace == 0.5 or c_retrace == 0.618
        )

        is_deep_crab = b_retrace == 0.886

        if (
            b_retrace > 0.618
            and not is_deep_crab
            or b_retrace < 0.382
            or (c_retrace < 0.382 or c_retrace > 0.886)
        ):
            continue

        xa_618_ext = a + xa_diff * 1.618

        bc_3_14_ext = c + bc_diff * 3.14
        ab_618_ext = c + ab_diff * 1.618
        ab_27_ext = c + ab_diff * 1.27

        terminal_point = xa_618_ext

        highs_after_c = df.loc[c_idx:, "High"]

        highs_above_terminal_point = highs_after_c[highs_after_c > terminal_point]

        if highs_above_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_above_terminal_point = (df.loc[c_idx:, "Close"] > terminal_point).sum()

        if (
            closes_above_terminal_point < 7
            and d > b + (terminal_point - b) * 0.5
            and (
                has_tested
                and (d_idx - highs_above_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect_crab:
                # Perfect crab pattern
                alt_name = "Bear Perfect Crab"

                clustered_levels = get_relative_clusters(
                    {
                        "3.14BC": bc_3_14_ext,
                        "1.618XA": xa_618_ext,
                        "1.618AB": ab_618_ext,
                        "1.27AB": ab_27_ext,
                        "1.618AB": ab_618_ext,
                    },
                    "1.618XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price > x
                    }
                )
            elif is_deep_crab:
                # Deep Crab pattern
                alt_name = "Bear Deep Crab"

                clustered_levels = get_relative_clusters(
                    {
                        "1.618XA": xa_618_ext,
                        "1.27AB": ab_27_ext,
                        "1.618AB": ab_618_ext,
                        "2.24BC": c + bc_diff * 2.24,
                        "2.618BC": c + bc_diff * 2.618,
                        "3.14BC": c + bc_diff * 3.14,
                        "3.618BC": c + bc_diff * 3.618,
                    },
                    "1.618XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price > x and price <= xa_618_ext
                    }
                )
            else:
                clustered_levels = get_relative_clusters(
                    {
                        "1.618XA": xa_618_ext,
                        "2.618BC": c + bc_diff * 2.618,
                        "3.14BC": c + bc_diff * 3.14,
                        "3.618BC": c + bc_diff * 3.618,
                        "1.27AB": ab_27_ext,
                        "1.618AB": ab_618_ext,
                    },
                    "1.618XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price > x
                    }
                )

    if selected:
        selected.update(dict(sym=sym, pattern="CRABD", alt_name=alt_name))

    return selected


def find_bullish_butterfly(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bullish Butterfly harmonic pattern
    """
    alt_name = "Bull Butterfly"
    pivot_len = pivots.shape[0]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)

    for x_idx in pivots.index:
        x = pivots.at[x_idx, "P"]

        if isinstance(x, pd.Series):
            x = pivots.at[x_idx, "P"].min()

        if x == df.at[x_idx, "High"]:
            continue

        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmax()
        a = pivots.at[a_idx, "P"]

        if df.loc[x_idx:a_idx, "Low"].min() != x:
            continue

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].min()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].max()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].min()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].max()

        xa_diff = a - x
        ab_diff = a - b
        bc_diff = c - b

        highest_high_xb = df.loc[x_idx:b_idx, "High"].max()
        lowest_low_ac = df.loc[a_idx:c_idx, "Low"].min()
        highest_high_from_b = df.loc[b_idx:, "High"].max()

        if (
            highest_high_xb != a
            or lowest_low_ac != b
            or highest_high_from_b != c
            or a == df.at[a_idx, "Low"]
            or b == df.at[b_idx, "High"]
            or c == df.at[c_idx, "Low"]
        ):
            # Check that the pattern is well formed
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect = b_retrace == 0.786 and (0.5 <= c_retrace <= 0.886)

        if b_retrace != 0.786 or c_retrace < 0.382 or c_retrace > 0.886:
            continue

        xa_27_ext = a - xa_diff * 1.27
        ab_27_ext = c - ab_diff * 1.27

        bc_618_ext = c - bc_diff * 1.618

        terminal_point = xa_27_ext

        lows_below_c = df.loc[c_idx:, "Low"]

        lows_below_terminal_point = lows_below_c.loc[lows_below_c < terminal_point]

        if lows_below_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_below_terminal_point = (df.loc[c_idx:, "Close"] < terminal_point).sum()

        if (
            closes_below_terminal_point < 7
            and d < b - (b - terminal_point) * 0.5
            and (
                has_tested
                and (d_idx - lows_below_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                # Perfect Butterfly pattern
                alt_name = "Bull Perfect Butterfly"

                clustered_levels = get_relative_clusters(
                    {
                        "1.27XA": xa_27_ext,
                        "1.618BC": bc_618_ext,
                        "1.27AB=CD": ab_27_ext,
                    },
                    "1.27XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price <= x
                    }
                )
            else:
                clustered_levels = get_relative_clusters(
                    {
                        "1.27XA": xa_27_ext,
                        "1.618BC": bc_618_ext,
                        "2BC": c - bc_diff * 2,
                        "2.24BC": c - bc_diff * 2.24,
                        "AB=CD": c - ab_diff,
                        "1.27AB=CD": ab_27_ext,
                    },
                    "1.27XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price <= x
                    }
                )

    if selected:
        selected.update(dict(sym=sym, pattern="BFLYU", alt_name=alt_name))

    return selected


def find_bearish_butterfly(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame, config
) -> Optional[dict]:
    """
    Bearish Butterfly harmonic pattern
    """
    alt_name = "Bear Butterfly"
    pivot_len = pivots.shape[0]

    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    selected: Optional[dict] = None

    assert isinstance(pivots.index, pd.DatetimeIndex)

    for x_idx in pivots.index:
        x = pivots.at[x_idx, "P"]

        if isinstance(x, pd.Series):
            x = pivots.at[x_idx, "P"].max()

        if x == df.at[x_idx, "Low"]:
            continue

        pos_after_x = get_next_index(pivots.index, x_idx)

        if pos_after_x >= pivot_len:
            break

        a_idx = pivots.loc[pivots.index[pos_after_x] :, "P"].idxmin()
        a = pivots.at[a_idx, "P"]

        if df.loc[x_idx:a_idx, "High"].max() != x:
            continue

        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        c = pivots.at[c_idx, "P"]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(x, pd.Series):
                x = pivots.at[x_idx, "P"].max()

            if isinstance(a, pd.Series):
                a = pivots.at[a_idx, "P"].min()

            if isinstance(b, pd.Series):
                b = pivots.at[b_idx, "P"].max()

            if isinstance(c, pd.Series):
                c = pivots.at[c_idx, "P"].min()

        xa_diff = x - a
        ab_diff = b - a
        bc_diff = b - c

        lowest_low_xb = df.loc[x_idx:b_idx, "Low"].min()
        highest_high_ac = df.loc[a_idx:c_idx, "High"].max()
        lowest_low_from_b = df.loc[b_idx:, "Low"].min()

        if (
            lowest_low_xb != a
            or highest_high_ac != b
            or lowest_low_from_b != c
            or a == df.at[a_idx, "High"]
            or b == df.at[b_idx, "Low"]
            or c == df.at[c_idx, "High"]
        ):
            # Check that the pattern is well formed
            continue

        b_retrace = fib_ser.loc[(fib_ser - (ab_diff / xa_diff)).abs().idxmin()]
        c_retrace = fib_ser.loc[(fib_ser - (bc_diff / ab_diff)).abs().idxmin()]

        is_perfect = b_retrace == 0.786 and (0.5 <= c_retrace <= 0.886)

        if b_retrace != 0.786 or c_retrace < 0.382 or c_retrace > 0.886:
            continue

        xa_27_ext = a + xa_diff * 1.27
        ab_27_ext = c + ab_diff * 1.27

        bc_618_ext = c + bc_diff * 1.618

        terminal_point = xa_27_ext

        highs_after_c = df.loc[c_idx:, "Low"]

        highs_above_terminal_point = highs_after_c.loc[highs_after_c > terminal_point]

        if highs_above_terminal_point.empty:
            has_tested = False
        else:
            has_tested = True

        closes_above_terminal_point = (df.loc[c_idx:, "Close"] > terminal_point).sum()

        if (
            closes_above_terminal_point < 7
            and d > b + (terminal_point - b) * 0.5
            and (
                has_tested
                and (d_idx - highs_above_terminal_point.index[0]).days < 7
                or not has_tested
            )
        ):
            selected = dict(
                df_start=df.index[0],
                df_end=df.index[-1],
                start=a_idx,
                end=d_idx,
                points={
                    "X": (x_idx, x),
                    "A": (a_idx, a),
                    f"{b_retrace:.3f}B": (b_idx, b),
                    f"{c_retrace:.3f}C": (c_idx, c),
                    "D": (d_idx, d),
                },
                extra_points={
                    "direction": (c_idx, c),
                },
            )

            if is_perfect:
                # Perfect Butterfly pattern
                alt_name = "Bear Perfect Butterfly"

                clustered_levels = get_relative_clusters(
                    {
                        "1.27XA": xa_27_ext,
                        "1.618BC": bc_618_ext,
                        "1.27AB=CD": ab_27_ext,
                    },
                    "1.27XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price >= x
                    }
                )
            else:
                clustered_levels = get_relative_clusters(
                    {
                        "1.27XA": xa_27_ext,
                        "1.618BC": bc_618_ext,
                        "2BC": c + bc_diff * 2,
                        "2.24BC": c + bc_diff * 2.24,
                        "AB=CD": c + ab_diff,
                        "1.27AB=CD": ab_27_ext,
                    },
                    "1.27XA",
                )

                selected["extra_points"].update(
                    {
                        level: (b_idx, price)
                        for level, price in clustered_levels.items()
                        if price >= x
                    }
                )

    if selected:
        selected.update(dict(sym=sym, pattern="BFLYD", alt_name=alt_name))

    return selected
