from typing import Tuple
import mplfinance as mpl
import matplotlib.pyplot as plt
import pandas as pd

LineCoordinates = Tuple[Tuple[pd.Timestamp, float], Tuple[pd.Timestamp, float]]

plot_args = {
    "type": 'candle',
    "style": 'tradingview',
    'figscale': 2,
    'returnfig': True,
    "alines": {
        'colors': 'royalblue',
        'alines': None,
        'linewidths': 0.8,
        'alpha': 0.7,
    }
}

log = None
is_silent = False


def set_logger(logger):
    # Set the logger for this module
    global log
    global is_silent
    log = logger
    is_silent = logger.getLogger().getEffectiveLevel() == log.WARNING


def has_time_component(datetime_index: pd.DatetimeIndex) -> bool:
    '''Return True if any value in DatetimeIndex has time component 
    other than `00:00:00` (Midnight hour)

    Ex Datetime(2023, 12, 10)           ->  `00:00:00` (Defaults to midnight)
       Datetime(2023, 12, 10, 0, 0)     ->  `00:00:00` (Midnight time)
       Datetime(2023, 12, 10, 12, 10)   ->  `12:10:00`
    '''
    return any(
        datetime_index.to_series().dt.time != pd.Timestamp('00:00:00').time())


def onKeyPress(event):
    if event.key == 'Q':
        plt.close('all')
        exit('User exit')


def plot_chart(df: pd.DataFrame, plot_args: dict):
    plt.ion()
    fig, _ = mpl.plot(df, **plot_args)
    fig.canvas.mpl_connect('key_press_event', onKeyPress)
    mpl.show(block=True)


def isPennant(a: float, b: float, c: float, d: float, e: float, f: float,
              avgBarLength: float) -> bool:
    r'''
          A
         /\        C
        /  \      /\    E
       /    \    /  \  /\
      /      \  /    \/  F
     /        \/      D
    /         B

    Height = A - B
    '''
    if a > c > e and b < d < f and e > f:
        return True

    a_c = abs(a - c) <= avgBarLength
    c_e = abs(c - e) <= avgBarLength

    if a_c and c_e and b < d < f < e:
        return True

    b_d = abs(b - d) <= avgBarLength

    # return a > c > e > f and b < d < f
    return b_d and a > c > e > f and f > d


def isHNS(a: float, b: float, c: float, d: float, e: float, f: float,
          avgBarLength: float) -> bool:
    r'''
    Head and Shoulders
                C
                /\
        A      /  \      E
        /\    /    \    /\
       /  \  /      \  /  \
      /    \/________\/____\F__Neckline
     /      B         D     \
    /                        \
    '''
    return c > max(a, e) and max(b, d) < min(
        a, e) and f < e and abs(b - d) < avgBarLength


def isReverseHNS(a: float, b: float, c: float, d: float, e: float, f: float,
                 avgBarLength: float) -> bool:
    r'''
    Reverse Head and Shoulders
    \
     \                  /
      \   _B_______D___/___ 
       \  /\      /\  /F   Neckline
        \/  \    /  \/
        A    \  /    E
              \/
              C
    '''
    return c < min(a, e) and min(b, d) > max(
        a, e) and f > e and abs(b - d) < avgBarLength


def isDoubleTop(a: float, b: float, c: float, d: float, aVol: int, cVol: int,
                avgBarLength: float) -> bool:
    r'''
    Double Top
          A     C
         /\    /\
        /  \  /  \
       /    \/    D
      /      B     
     /  
    /
    '''
    return abs(a - c) <= avgBarLength and cVol < aVol and b < min(
        a, c) and b < d < c


def isDoubleBottom(a: float, b: float, c: float, d: float, aVol: int,
                   cVol: int, avgBarLength: float) -> bool:
    r'''
    Double Bottom
      \
       \
        \      B
         \    /\    D
          \  /  \  /
           \/    \/
            A     C
    '''

    return abs(a - c) <= avgBarLength and cVol < aVol and b > max(
        a, c) and b > d > c


def bearishVCP(a: float, b: float, c: float, d: float, e: float,
               avgBarLength: float) -> bool:
    r'''
    Volatilty Contraction pattern
          B
         /\      D
        /  \    /\
       /    \  /  \ 
      /      \/    E
     A       C

    B is highest point in pattern
    D is second highest after B
    '''
    return abs(a - c) <= avgBarLength and b > max(a, c, d, e) and d > max(
        a, c, e) and e > c


def bullishVCP(a: float, b: float, c: float, d: float, e: float,
               avgBarLength: float) -> bool:
    r'''
    Volatilty Contraction pattern

       A        C
         \      /\    E
          \    /  \  /
           \  /    \/
            \/      D
             B

    B is lowest point in pattern
    D is second lowest after B
    '''
    return abs(a - c) <= avgBarLength and b < min(a, c, d, e) and d < min(
        a, c, e) and e < c


def getMaxMin(df: pd.DataFrame, barsLeft=6, barsRight=6) -> pd.DataFrame:
    window = barsLeft + 1 + barsRight

    l_max_dt = []
    l_min_dt = []
    cols = ['P', 'V']

    for win in df.rolling(window):
        if win.shape[0] < window:
            continue

        idx = win.index[barsLeft + 1]  # center candle

        if win['High'].idxmax() == idx:
            l_max_dt.append(idx)

        if win['Low'].idxmin() == idx:
            l_min_dt.append(idx)

    maxima = pd.DataFrame(df.loc[l_max_dt, ['High', 'Volume']])
    maxima.columns = cols

    minima = pd.DataFrame(df.loc[l_min_dt, ['Low', 'Volume']])
    minima.columns = cols

    return pd.concat([maxima, minima]).sort_index()


def getNextIndex(index: pd.DatetimeIndex, idx: pd.Timestamp) -> int:
    pos = index.get_loc(idx)

    if isinstance(pos, slice):
        if not isinstance(pos.stop, int):
            raise TypeError("Expected Integer")
        return pos.stop

    if not isinstance(pos, int):
        raise TypeError("Expected Integer")
    return pos + 1


def getPrevIndex(index: pd.DatetimeIndex, idx: pd.Timestamp) -> int:
    pos = index.get_loc(idx)

    if isinstance(pos, slice):
        if not isinstance(pos.stop, int):
            raise TypeError("Expected Integer")
        return pos.stop

    if not isinstance(pos, int):
        raise TypeError("Expected Integer")
    return pos - 1


def generate_trend_line(
        series: pd.Series, date1: pd.Timestamp,
        date2: pd.Timestamp) -> Tuple[LineCoordinates, float, float]:
    """Return the end coordinates for a trendline along with slope and y-intercept
       Input: Pandas series with a pandas.DatetimeIndex, and two dates:
              The two dates are used to determine two "prices" from the series

       Output: tuple(tuple(coord, coord), slope, y-intercept)
       source: https://github.com/matplotlib/mplfinance/blob/master/examples/scratch_pad/trend_line_extrapolation.ipynb

    """
    index = series.index

    p1 = series[date1]
    p2 = series[date2]

    d1 = index.get_loc(date1)
    d2 = index.get_loc(date2)

    lastIdx = index[-1]
    lastIdxPos = index.get_loc(lastIdx)

    if not isinstance(lastIdx, pd.Timestamp):
        raise TypeError('Expected pd.Timestamp')

    if not isinstance(p1, float) or not isinstance(p2, float):
        raise TypeError('Expected float')

    if not isinstance(d1, int) or not isinstance(d2, int) or not isinstance(
            lastIdxPos, int):
        raise TypeError('Expected integer')

    # b = y - mx
    # where m is slope,
    # b is y-intercept
    # slope m = change in y / change in x
    slope = (p2 - p1) / (d2 - d1)

    # b = y - mx
    yintercept = p1 - slope * d1

    # y = mx + b
    start_coords = (date1, slope * d1 + yintercept)
    end_coords = (lastIdx, slope * lastIdxPos + yintercept)

    return ((start_coords, end_coords), slope, yintercept)


def findBullishVCP(sym: str, df: pd.DataFrame, pivots: pd.DataFrame):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmax()

    a = pivots.loc[a_idx, 'P']

    e_idx = df.index[-1]
    e = df.loc[e_idx, 'Close']

    while True:

        b_idx = pivots.loc[a_idx:, 'P'].idxmin()

        # high and low pivots occured on the same date
        if a_idx == b_idx:
            if not isinstance(a_idx, pd.Timestamp):
                raise TypeError("Expected pd.Timestamp")

            idx = getNextIndex(pivots.index, a_idx)

            if idx >= pivot_len:
                break

            a_idx = pivots.index[idx]
            a = pivots.loc[a_idx, 'P']
            continue

        b = pivots.loc[b_idx, 'P']

        # high and low pivots occured on the same date
        idx = getNextIndex(pivots.index, b_idx)

        if idx >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[idx]:, 'P'].idxmin()
        d = pivots.loc[d_idx, 'P']

        c_idx = pivots.loc[b_idx:d_idx, 'P'].idxmax()
        c = pivots.loc[c_idx, 'P']

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[0]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[1]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[0]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[1]

        if bullishVCP(a, b, c, d, e, avgBarLength):

            # check if Level C has been breached after it was formed
            if (c_idx != df.loc[c_idx:, 'Close'].idxmax()
                    or d_idx != df.loc[d_idx:, 'Close'].idxmin()):
                # Level C is breached, current pattern is not valid

                # check if C is the last pivot formed
                if pivots.index[-1] == c_idx or pivots.index[-1] == d_idx:
                    break

                # continue search for patterns
                a_idx, a = c_idx, c
                continue

            if log:
                log.warning(sym)

            # silent mode
            if is_silent:
                break

            plot_args['title'] = f'{sym} - Bull VCP'

            plot_args['alines']['colors'] = (('green', ) +
                                             ('midnightblue', ) * 4)

            entryLine = ((c_idx, c), (e_idx, c))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))

            plot_args['alines']['alines'] = (entryLine, ab, bc, cd, de)

            plot_chart(df, plot_args)
            break

        a_idx, a = c_idx, c


def findBearishVCP(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmin()
    a = pivots.loc[a_idx, 'P']

    e_idx = df.index[-1]
    e = df.loc[e_idx, 'Close']
    idx = None

    while True:
        b_idx = pivots.loc[a_idx:, 'P'].idxmax()

        # high and low pivots occured on the same date
        if a_idx == b_idx:
            if not isinstance(a_idx, pd.Timestamp):
                raise TypeError("Expected pd.Timestamp")

            idx = getNextIndex(pivots.index, a_idx)

            if idx >= pivot_len:
                break

            a_idx = pivots.index[idx]
            a = pivots.loc[a_idx, 'P']
            continue

        b = pivots.loc[b_idx, 'P']

        idx = getNextIndex(pivots.index, b_idx)

        if idx >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[idx]:, 'P'].idxmax()
        d = pivots.loc[d_idx, 'P']

        c_idx = pivots.loc[b_idx:d_idx, 'P'].idxmin()
        c = pivots.loc[c_idx, 'P']

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[1]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[0]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[1]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[0]

        if bearishVCP(a, b, c, d, e, avgBarLength):

            if (d_idx != df.loc[d_idx:, 'Close'].idxmax()
                    or c_idx != df.loc[c_idx:, 'Close'].idxmin()):

                if pivots.index[-1] == d_idx or pivots.index[-1] == c_idx:
                    break

                a_idx, a = c_idx, c
                continue

            if log:
                log.warning(sym)

            # silent mode
            if is_silent:
                break

            plot_args['title'] = f'{sym} - Bear VCP'

            entryLine = ((c_idx, c), (e_idx, c))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))

            plot_args['alines']['alines'] = (entryLine, ab, bc, cd, de)

            plot_args['alines']['colors'] = (('green', ) +
                                             ('midnightblue', ) * 4)
            plot_chart(df, plot_args)
            break

        # We assign pivot level C to be the new A
        # This may not be the lowest pivot, so additional checks are required.
        a_idx, a = c_idx, c


def findDoubleBottom(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmin()
    a, aVol = pivots.loc[a_idx, ['P', 'V']]
    d_idx = df.index[-1]
    d = df.loc[d_idx, 'Close']

    if not isinstance(a_idx, pd.Timestamp):
        raise TypeError("Expected pd.Timestamp")

    while True:
        pos = getNextIndex(pivots.index, a_idx)

        if pos >= pivot_len:
            break

        # A is the high of bar
        if a == df.loc[a_idx, 'High']:
            pos = getNextIndex(pivots.index, a_idx)
            idx = pivots.index[pos]

            a_idx = pivots.loc[idx:, 'P'].idxmin()
            a, aVol = pivots.loc[a_idx, ['P', 'V']]
            continue

        c_idx = pivots.loc[pivots.index[pos]:, 'P'].idxmin()
        c, cVol = pivots.loc[c_idx, ['P', 'V']]

        # check if Level C has been breached after it was formed
        if c_idx != df.loc[c_idx:, 'Close'].idxmin():
            # Level C is breached, current pattern is not valid
            a_idx, a, aVol = c_idx, c, cVol
            continue

        b_idx = pivots.loc[a_idx:c_idx, 'P'].idxmax()
        b = pivots.loc[b_idx, 'P']

        if b_idx != pivots.loc[b_idx:, 'P'].idxmax():
            a_idx, a, aVol = c_idx, c, cVol
            continue

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[1]

            if isinstance(aVol, pd.Series | str):
                aVol = pivots.loc[a_idx, 'V'].iloc[1]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[0]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[1]

            if isinstance(cVol, pd.Series | str):
                cVol = pivots.loc[c_idx, 'V'].iloc[1]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if isDoubleBottom(a, b, c, d, aVol, cVol, avgBarLength):

            if (c_idx != df.loc[c_idx:, 'Close'].idxmin()
                    or b_idx != df.loc[b_idx:, 'Close'].idxmax()):

                a_idx, a, aVol = c_idx, c, cVol
                continue

            if df.loc[c_idx:, 'Close'].max() > b:
                a_idx, a, aVol = c_idx, c, cVol
                continue

            if log:
                log.warning(sym)

            # silent mode
            if is_silent:
                break

            plot_args['title'] = f'{sym} - Double bottom'

            entryLine = ((b_idx, b), (d_idx, b))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))

            plot_args['alines']['alines'] = (entryLine, ab, bc, cd)

            plot_args['alines']['colors'] = (('green', ) +
                                             ('midnightblue', ) * 4)

            plot_chart(df, plot_args)
            break

        a_idx, a, aVol = c_idx, c, cVol


def findDoubleTop(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmax()
    a, aVol = pivots.loc[a_idx, ['P', 'V']]
    d_idx = df.index[-1]
    d = df.loc[d_idx, 'Close']

    if not isinstance(a_idx, pd.Timestamp):
        raise TypeError("Expected pd.Timestamp")

    while True:
        idx = getNextIndex(pivots.index, a_idx)

        if idx >= pivot_len:
            break

        if a == df.loc[a_idx, 'Low']:
            pos = getNextIndex(pivots.index, a_idx)
            idx = pivots.index[pos]

            a_idx = pivots.loc[idx:, 'P'].idxmax()
            a, aVol = pivots.loc[a_idx, ['P', 'V']]
            continue

        c_idx = pivots.loc[pivots.index[idx]:, 'P'].idxmax()
        c, cVol = pivots.loc[c_idx, ['P', 'V']]

        b_idx = pivots.loc[a_idx:c_idx, 'P'].idxmin()
        b = pivots.loc[b_idx, 'P']

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[0]

            if isinstance(aVol, pd.Series | str):
                aVol = pivots.loc[a_idx, 'V'].iloc[0]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[1]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[0]

            if isinstance(cVol, pd.Series | str):
                cVol = pivots.loc[c_idx, 'V'].iloc[0]

        df_slice = df.loc[a_idx:c_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if isDoubleTop(a, b, c, d, aVol, cVol, avgBarLength):

            # check if Level C has been breached after it was formed
            if (c_idx != df.loc[c_idx:, 'Close'].idxmax()
                    or b_idx != df.loc[b_idx:, 'Close'].idxmin()):
                # Level C is breached, current pattern is not valid
                a_idx, a, aVol = c_idx, c, cVol
                continue

            if log:
                log.warning(sym)

            # silent mode
            if is_silent:
                break

            plot_args['title'] = f'{sym} - Double Top'

            entryLine = ((b_idx, b), (d_idx, b))
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))

            plot_args['alines']['alines'] = (entryLine, ab, bc, cd)

            plot_args['alines']['colors'] = (('green', ) +
                                             ('midnightblue', ) * 4)
            plot_chart(df, plot_args)
            break

        a_idx, a, aVol = c_idx, c, cVol


def findPennant(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmax()
    a = pivots.loc[a_idx, 'P']

    while True:
        b_idx = pivots.loc[a_idx:, 'P'].idxmin()
        b = pivots.loc[b_idx, 'P']

        # A is already the lowest point
        if a_idx == b_idx:
            if not isinstance(a_idx, pd.Timestamp):
                raise TypeError("Expected pd.Timestamp")

            idx = getNextIndex(pivots.index, a_idx)

            if idx >= pivot_len:
                break

            a_idx = pivots.index[idx]
            a = pivots.loc[a_idx, 'P']
            continue

        b = pivots.loc[b_idx, 'P']

        idx = getNextIndex(pivots.index, b_idx)

        if idx >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[idx]:, 'P'].idxmin()
        d = pivots.loc[d_idx, 'P']

        c_idx = pivots.loc[b_idx:d_idx, 'P'].idxmax()
        c = pivots.loc[c_idx, 'P']

        idx = getNextIndex(pivots.index, d_idx)

        if idx >= pivot_len:
            break

        f_idx = df.index[-1]
        f = df.loc[f_idx, 'Close']

        e_idx = pivots.loc[d_idx:f_idx, 'P'].idxmax()
        e = pivots.loc[e_idx, 'P']

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[0]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[1]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[0]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[1]

            if isinstance(e, pd.Series | str):
                e = pivots.loc[e_idx, 'P'].iloc[0]

        df_slice = df.loc[a_idx:d_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if isPennant(a, b, c, d, e, f, avgBarLength):

            if (c_idx != df.loc[c_idx:, 'Close'].idxmax()
                    or d_idx != df.loc[d_idx:, 'Close'].idxmin()):
                a_idx, a = c_idx, c
                continue

            if not isinstance(a_idx, pd.Timestamp):
                raise TypeError("Expected pd.Timestamp")

            upper_line, *_ = generate_trend_line(df.High, a_idx, c_idx)
            lower_line, *_ = generate_trend_line(df.Low, b_idx, d_idx)

            if upper_line[1][1] < lower_line[1][1]:
                break

            if log:
                log.warning(sym)

            # silent mode
            if is_silent:
                break

            plot_args['alines'] = ((upper_line), (lower_line))

            plot_args['title'] = f'{sym} - Pennant'

            plot_chart(df, plot_args)
            break

        a_idx, c = c_idx, c


def findHNS(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    f_idx = df.index[-1]
    f = df.at[f_idx, 'Close']

    c_idx = pivots['P'].idxmax()
    c = pivots.at[c_idx, 'P']

    if not isinstance(c_idx, pd.Timestamp):
        raise TypeError('Expected pd.Timestamp')

    while True:
        pos = getPrevIndex(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_before_c = pivots.index[pos]

        a_idx = pivots.loc[:idx_before_c, 'P'].idxmax()
        a = pivots.loc[a_idx, 'P']

        b_idx = pivots.loc[a_idx:c_idx, 'P'].idxmin()
        b = pivots.loc[b_idx, 'P']

        pos = getNextIndex(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_after_c = pivots.index[pos]

        e_idx = pivots.loc[idx_after_c:, 'P'].idxmax()
        e = pivots.loc[e_idx, 'P']

        d_idx = pivots.loc[c_idx:e_idx, 'P'].idxmin()
        d = pivots.loc[d_idx, 'P']

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[0]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[1]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[0]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[1]

            if isinstance(e, pd.Series | str):
                e = pivots.loc[e_idx, 'P'].iloc[0]

        df_slice = df.loc[b_idx:d_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if isHNS(a, b, c, d, e, f, avgBarLength):

            neckline_price = min(b, d)

            lowest_after_e = df.loc[e_idx:, 'Low'].min()

            if (lowest_after_e < neckline_price
                    and abs(lowest_after_e - neckline_price) > avgBarLength):

                c_idx, c = e_idx, e
                continue

            # bd is the line coordinate for points B and D
            bd, m, y_intercept = generate_trend_line(df.Low, b_idx, d_idx)

            # Get the y coordinate of the trendline at the end of the chart
            # With the given slope(m) and y-intercept(b) as y_int,
            # Get the x coordinate (index position of last date in DataFrame)
            # and calculate value of y coordinate using y = mx + b
            x = df.index.get_loc(f_idx)

            if not isinstance(x, int):
                raise TypeError('Expected Integer')

            y = m * x + y_intercept

            # if the close price is below the neckline (trendline), skip
            if f < y:
                c_idx, c = e_idx, e
                continue

            if log:
                log.warning(sym)

            # silent mode
            if is_silent:
                break

            plot_args['title'] = f'{sym} - Head & Shoulders - Bearish'

            # lines
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))
            ef = ((e_idx, e), (f_idx, f))

            plot_args['alines']['alines'] = (bd, ab, bc, cd, de, ef)
            plot_args['alines']['colors'] = (('green', ) +
                                             ('midnightblue', ) * 5)

            plot_chart(df, plot_args)
            break

        c_idx, c = e_idx, e


def findReverseHNS(
    sym: str,
    df: pd.DataFrame,
    pivots: pd.DataFrame,
):

    if not isinstance(pivots.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')

    pivot_len = pivots.shape[0]
    f_idx = df.index[-1]
    f = df.at[f_idx, 'Close']

    c_idx = pivots['P'].idxmin()
    c = pivots.at[c_idx, 'P']

    if not isinstance(c_idx, pd.Timestamp):
        raise TypeError('Expected pd.Timestamp')

    while True:
        pos = getPrevIndex(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_before_c = pivots.index[pos]

        a_idx = pivots.loc[:idx_before_c, 'P'].idxmin()
        a = pivots.at[a_idx, 'P']

        b_idx = pivots.loc[a_idx:c_idx, 'P'].idxmax()
        b = pivots.at[b_idx, 'P']

        pos = getNextIndex(pivots.index, c_idx)

        if pos >= pivot_len:
            break

        idx_after_c = pivots.index[pos]

        e_idx = pivots.loc[idx_after_c:, 'P'].idxmin()
        e = pivots.at[e_idx, 'P']

        d_idx = pivots.loc[c_idx:e_idx, 'P'].idxmax()
        d = pivots.at[d_idx, 'P']

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[1]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[0]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[1]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[0]

            if isinstance(e, pd.Series | str):
                e = pivots.loc[e_idx, 'P'].iloc[1]

        df_slice = df.loc[b_idx:d_idx]
        avgBarLength = (df_slice['High'] - df_slice['Low']).mean()

        if isReverseHNS(a, b, c, d, e, f, avgBarLength):

            neckline_price = min(b, d)

            highest_after_e = df.loc[e_idx:, 'High'].max()

            if (highest_after_e > neckline_price
                    and abs(highest_after_e - neckline_price) > avgBarLength):
                c_idx, c = e_idx, e
                continue

            # bd is the trendline coordinates from B to D (neckline)
            bd, m, y_intercept = generate_trend_line(df.High, b_idx, d_idx)

            # Get the y coordinate of the trendline at the end of the chart
            # With the given slope(m) and y-intercept(b) as y_int,
            # Get the x coordinate (index position of last date in DataFrame)
            # and calculate value of y coordinate using y = mx + b
            x = df.index.get_loc(df.index[-1])

            if not isinstance(x, int):
                raise TypeError('Expected Integer')

            y = m * x + y_intercept

            # if close price is greater than neckline (trendline), skip
            if f > y:
                c_idx, c = e_idx, e
                continue

            # silent mode
            if log:
                log.warning(sym)

            if is_silent:
                break

            plot_args['title'] = f'{sym} - Reverse Head & Shoulders - Bullish'

            # lines
            ab = ((a_idx, a), (b_idx, b))
            bc = ((b_idx, b), (c_idx, c))
            cd = ((c_idx, c), (d_idx, d))
            de = ((d_idx, d), (e_idx, e))
            ef = ((e_idx, e), (f_idx, f))

            plot_args['alines']['alines'] = (bd, ab, bc, cd, de, ef)
            plot_args['alines']['colors'] = (('green', ) +
                                             ('midnightblue', ) * 5)

            plot_chart(df, plot_args)
            break

        c_idx, c = e_idx, e
