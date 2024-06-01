from typing import Any, Optional, TypeVar, NamedTuple
import numpy as np
import pandas as pd
import logging

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


def make_serializable(obj: T) -> T:
    """Convert pandas.Timestamp and numpy.Float32 objects in obj
    to serializable native types"""

    def serialize(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, np.generic)):
            # Convert Pandas Timestamp to Python datetime or NumPy item
            return (
                obj.isoformat() if isinstance(obj, pd.Timestamp) else obj.item()
            )
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
    return any(
        datetime_index.to_series().dt.time != pd.Timestamp("00:00:00").time()
    )


def get_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window=15
) -> pd.Series:
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


def get_max_min(df: pd.DataFrame, barsLeft=6, barsRight=6) -> pd.DataFrame:
    window = barsLeft + 1 + barsRight

    l_max_dt = []
    l_min_dt = []
    cols = ["P", "V"]

    for win in df.rolling(window):
        if win.shape[0] < window:
            continue

        idx = win.index[barsLeft + 1]  # center candle

        if win["High"].idxmax() == idx:
            l_max_dt.append(idx)

        if win["Low"].idxmin() == idx:
            l_min_dt.append(idx)

    maxima = pd.DataFrame(df.loc[l_max_dt, ["High", "Volume"]])
    maxima.columns = cols

    minima = pd.DataFrame(df.loc[l_min_dt, ["Low", "Volume"]])
    minima.columns = cols

    return pd.concat([maxima, minima]).sort_index()


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


def find_bullish_vcp(
    sym: str, df: pd.DataFrame, pivots: pd.DataFrame
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
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

        if pivots.index.has_duplicates:
            if isinstance(a, (pd.Series, str)):
                a = pivots.at[a_idx, "P"].iloc[0]

            if isinstance(b, (pd.Series, str)):
                b = pivots.at[b_idx, "P"].iloc[1]

            if isinstance(c, (pd.Series, str)):
                c = pivots.at[c_idx, "P"].iloc[0]

            if isinstance(d, (pd.Series, str)):
                d = pivots.at[d_idx, "P"].iloc[1]

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

            entryLine = ((c_idx, c), (e_idx, c))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))

            logger.debug(f"{sym} - VCPU")

            return dict(
                sym=sym,
                pattern="VCPU",
                start=a_idx,
                end=e_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                lines=(entryLine, ab, bc, cd, de),
            )

        a_idx, a = c_idx, c


def find_bearish_vcp(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
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
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

        if pivots.index.has_duplicates:
            if isinstance(a, (pd.Series, str)):
                a = pivots.at[a_idx, "P"].iloc[1]

            if isinstance(b, (pd.Series, str)):
                b = pivots.at[b_idx, "P"].iloc[0]

            if isinstance(c, (pd.Series, str)):
                c = pivots.at[c_idx, "P"].iloc[1]

            if isinstance(d, (pd.Series, str)):
                d = pivots.at[d_idx, "P"].iloc[0]

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

            entryLine = ((c_idx, c), (e_idx, c))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))

            logger.debug(f"{sym} - VCPD")

            return dict(
                sym=sym,
                pattern="VCPD",
                start=a_idx,
                end=e_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                lines=(entryLine, ab, bc, cd, de),
            )

        # We assign pivot level C to be the new A
        # This may not be the lowest pivot, so additional checks are required.
        a_idx, a = c_idx, c


def find_double_bottom(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
) -> Optional[dict]:
    """Find Double bottom.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    a_idx = pivots["P"].idxmin()
    a, aVol = pivots.loc[a_idx, ["P", "V"]]
    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    atr_ser = get_atr(df.High, df.Low, df.Close)

    assert isinstance(a_idx, pd.Timestamp)

    while True:
        pos_after_a = get_next_index(pivots.index, a_idx)

        if pos_after_a >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmin()
        c, cVol = pivots.loc[c_idx, ["P", "V"]]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = pivots.at[b_idx, "P"]

        atr = atr_ser.at[c_idx]

        if pivots.index.has_duplicates:
            if isinstance(a, (pd.Series, str)):
                a = pivots.at[a_idx, "P"].iloc[1]

            if isinstance(aVol, (pd.Series, str)):
                aVol = pivots.at[a_idx, "V"].iloc[1]

            if isinstance(b, (pd.Series, str)):
                b = pivots.at[b_idx, "P"].iloc[0]

            if isinstance(c, (pd.Series, str)):
                c = pivots.at[c_idx, "P"].iloc[1]

            if isinstance(cVol, (pd.Series, str)):
                cVol = pivots.at[c_idx, "V"].iloc[1]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

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

            entryLine = ((b_idx, b), (d_idx, b))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))

            logger.debug(f"{sym} - DBOT")

            return dict(
                sym=sym,
                pattern="DBOT",
                start=a_idx,
                end=d_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                lines=(entryLine, ab, bc, cd),
            )

        a_idx, a, aVol = c_idx, c, cVol


def find_double_top(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
) -> Optional[dict]:
    """Find Double Top.

    Returns None if no patterns found.

    Else returns an Tuple of dicts containing plot arguments and pattern data.
    """

    assert isinstance(pivots.index, pd.DatetimeIndex)

    pivot_len = pivots.shape[0]
    a_idx = pivots["P"].idxmax()
    a, aVol = pivots.loc[a_idx, ["P", "V"]]
    d_idx = df.index[-1]
    d = df.at[d_idx, "Close"]

    atr_ser = get_atr(df.High, df.Low, df.Close)

    assert isinstance(a_idx, pd.Timestamp)

    while True:
        idx = get_next_index(pivots.index, a_idx)

        if idx >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[idx] :, "P"].idxmax()
        c, cVol = pivots.loc[c_idx, ["P", "V"]]

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = pivots.at[b_idx, "P"]

        atr = atr_ser.at[c_idx]

        if pivots.index.has_duplicates:
            if isinstance(a, (pd.Series, str)):
                a = pivots.at[a_idx, "P"].iloc[0]

            if isinstance(aVol, (pd.Series, str)):
                aVol = pivots.at[a_idx, "V"].iloc[0]

            if isinstance(b, (pd.Series, str)):
                b = pivots.at[b_idx, "P"].iloc[1]

            if isinstance(c, (pd.Series, str)):
                c = pivots.at[c_idx, "P"].iloc[0]

            if isinstance(cVol, (pd.Series, str)):
                cVol = pivots.at[c_idx, "V"].iloc[0]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

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

            entryLine = ((b_idx, b), (d_idx, b))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))

            logger.debug(f"{sym} - DTOP")

            return dict(
                sym=sym,
                pattern="DTOP",
                start=a_idx,
                end=d_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                lines=(entryLine, ab, bc, cd),
            )

        a_idx, a, aVol = c_idx, c, cVol


def find_triangles(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
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

        d_idx = pivots.loc[pivots.index[pos_after_b] :, "P"].idxmin()
        d = pivots.at[d_idx, "P"]

        c_idx = pivots.loc[pivots.index[pos_after_a] :, "P"].idxmax()
        c = pivots.at[c_idx, "P"]

        pos_after_c = get_next_index(pivots.index, c_idx)

        if pos_after_c >= pivot_len:
            break

        e_idx = pivots.loc[pivots.index[pos_after_c] :, "P"].idxmax()
        e = pivots.at[e_idx, "P"]

        if pivots.index.has_duplicates:
            if isinstance(a, (pd.Series, str)):
                a = pivots.at[a_idx, "P"].iloc[0]

            if isinstance(b, (pd.Series, str)):
                b = pivots.at[b_idx, "P"].iloc[1]

            if isinstance(c, (pd.Series, str)):
                c = pivots.at[c_idx, "P"].iloc[0]

            if isinstance(d, (pd.Series, str)):
                d = pivots.at[d_idx, "P"].iloc[1]

            if isinstance(e, (pd.Series, str)):
                e = pivots.at[e_idx, "P"].iloc[0]

        df_slice = df.loc[a_idx:d_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

        triangle = is_triangle(a, b, c, d, e, f, avgBarLength)

        if triangle is not None:
            # check if high of C or low of D has been breached
            # Check if A is indeed the pivot high
            if (
                a == df.at[a_idx, "Low"]
                or c_idx != df.loc[c_idx:, "Close"].idxmax()
                or d_idx != df.loc[d_idx:, "Close"].idxmin()
            ):
                a_idx, a = c_idx, c
                continue

            upper = generate_trend_line(df.High, a_idx, c_idx)
            lower = generate_trend_line(df.Low, b_idx, d_idx)

            # If trendlines have intersected, pattern has played out
            if upper.line.end.y < lower.line.end.y:
                break

            if triangle == "Ascending" and (
                upper.slope > 0.1 and lower.slope < 0.2
            ):
                break

            if triangle == "Descending" and (
                lower.slope < -0.1 and upper.slope > -0.2
            ):
                break

            if triangle == "Symmetric" and (
                upper.slope > -0.2 and lower.slope < 0.2
            ):
                break

            logger.debug(f"{sym} - {triangle}")

            return dict(
                sym=sym,
                pattern=triangle,
                start=a_idx,
                end=f_idx,
                df_start=df.index[0],
                df_end=df.index[-1],
                slope_upper=upper.slope,
                slope_lower=lower.slope,
                lines=(upper.line, lower.line),
            )

        a_idx, c = c_idx, c


def find_hns(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
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
            if isinstance(a, (pd.Series, str)):
                a = pivots.at[a_idx, "P"].iloc[0]

            if isinstance(b, (pd.Series, str)):
                b = pivots.at[b_idx, "P"].iloc[1]

            if isinstance(c, (pd.Series, str)):
                c = pivots.at[c_idx, "P"].iloc[0]

            if isinstance(d, (pd.Series, str)):
                d = pivots.at[d_idx, "P"].iloc[1]

            if isinstance(e, (pd.Series, str)):
                e = pivots.at[e_idx, "P"].iloc[0]

        df_slice = df.loc[b_idx:d_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

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

            # lines
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))
            ef = ((e_idx, e), (f_idx, f))

            if tline.slope < 0:
                entry_line = ((b_idx, b), (f_idx, b))

                lines = (entry_line, tline.line, ab, bc, cd, de, ef)
            else:
                lines = (tline.line, ab, bc, cd, de, ef)

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
                lines=lines,
            )

        c_idx, c = e_idx, e


def find_reverse_hns(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
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
            if isinstance(a, (pd.Series, str)):
                a = pivots.loc[a_idx, "P"].iloc[1]

            if isinstance(b, (pd.Series, str)):
                b = pivots.loc[b_idx, "P"].iloc[0]

            if isinstance(c, (pd.Series, str)):
                c = pivots.loc[c_idx, "P"].iloc[1]

            if isinstance(d, (pd.Series, str)):
                d = pivots.loc[d_idx, "P"].iloc[0]

            if isinstance(e, (pd.Series, str)):
                e = pivots.loc[e_idx, "P"].iloc[1]

        df_slice = df.loc[b_idx:d_idx]
        avgBarLength = (df_slice["High"] - df_slice["Low"]).mean()

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

            # lines
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))
            ef = ((e_idx, e), (f_idx, f))

            if tline.slope > 0:
                entry_line = ((b_idx, b), (f_idx, b))

                lines = (entry_line, tline.line, ab, bc, cd, de, ef)
            else:
                lines = (tline.line, ab, bc, cd, de, ef)

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
                lines=lines,
            )

        c_idx, c = e_idx, e
