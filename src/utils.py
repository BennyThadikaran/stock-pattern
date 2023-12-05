from ctypes import c_void_p
from mplfinance import plot
import pandas as pd

plot_args = {
    "type": 'candle',
    "style": 'tradingview',
    'figscale': 2,
    "alines": {
        'colors': ['crimson'],
        'alines': None,
        'linewidths': (0.5, ),
        'alpha': 0.4,
    }
}


def isPennant(a, b, c, d, e, f, avgCandleLength):
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
    if a > c > e and b < d < f and e > f and a - b > a - a * 0.95:
        return True

    a_c = abs(a - c) <= avgCandleLength
    c_e = abs(c - e) <= avgCandleLength

    if a_c and c_e and b < d < f < e:
        return True

    b_d = abs(b - d) <= avgCandleLength

    # return a > c > e > f and b < d < f
    return b_d and a > c > e > f and f > d


def isHeadAndShoulders(a, b, c, d, e, f, avgCandleLength):
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
    neckline = min(b, d)

    return c > max(a, e) and abs(b - d) <= avgCandleLength and neckline < min(
        a, e) and e > f < neckline


def isReverseHeadAndShoulders(a, b, c, d, e, f, avgCandleLength):
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
    neckline = min(b, d)

    return c < min(a, e) and abs(b - d) <= avgCandleLength and neckline > max(
        a, e) and f > e and f < neckline


def isDoubleTop(a, b, c, d, aVol, cVol, avgCandleLength):
    r'''
    Double Bottom
          A     C
         /\    /\
        /  \  /  \
       /    \/    D
      /      B     
     /  
    /
    '''
    return abs(a - c) <= avgCandleLength and cVol < aVol and b < min(
        a, c) and b < d < c


def isDoubleBottom(a, b, c, d, aVol, cVol, avgCandleLength):
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

    return abs(a - c) <= avgCandleLength and cVol < aVol and b > max(
        a, c) and b > d > c


def bearishVCP(a, b, c, d, e, avgCandleLength):
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
    return abs(a - c) <= avgCandleLength and b > max(a, c, d, e) and d > max(
        a, c, e) and e > c


def bullishVCP(a, b, c, d, e, avgCandleLength):
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
    return abs(a - c) <= avgCandleLength and b < min(a, c, d, e) and d < min(
        a, c, e) and e < c


def getMaxMin(df, window=6):
    pivot = window + 1
    window = window * 2 + 1
    l_max_dt = []
    l_min_dt = []

    for win in df.rolling(window):
        if win.shape[0] < window:
            continue

        idx = win.index[pivot]
        high, low = win.loc[idx, ['High', 'Low']]

        if win['High'].max() == high:
            l_max_dt.append(idx)

        if win['Low'].min() == low:
            l_min_dt.append(idx)

    maxima = pd.DataFrame(df.loc[l_max_dt, ['High', 'Volume']])
    maxima.columns = ['P', 'V']

    minima = pd.DataFrame(df.loc[l_min_dt, ['Low', 'Volume']])
    minima.columns = ['P', 'V']

    max_min = pd.concat([maxima, minima]).sort_index()
    return max_min


def getNextIndex(index, idx):
    try:
        return index.get_loc(idx) + 1
    except TypeError:
        # duplicate entry returns a slice of high and low values
        # on the same date, we take the second entry.
        # slice object (start, stop, step)
        return index.get_loc(idx).stop


def generate_trend_line(series, date1, date2):
    """Return a trend line given two dates and a time-series.
       Input: Pandas series with a pandas.DatetimeIndex, and two dates:
              The two dates are used to determine two "prices" from the series

       Output: Trend line as
       source: https://github.com/matplotlib/mplfinance/blob/master/examples/scratch_pad/trend_line_extrapolation.ipynb

    """
    index = series.index

    p1 = series[date1]
    p2 = series[date2]

    d1 = float(index.get_loc(date1))
    d2 = float(index.get_loc(date2))

    # slope m = change in y / change in x
    slope = (p2 - p1) / (d2 - d1)
    yintercept = ((p1 - slope * d1) + (p2 - slope * d2)) / 2.0

    # y = mx + b
    # where m is slope, b is y coordinate
    return ((date1, slope * index.get_loc(date1) + yintercept),
            (index[-1], slope * index.get_loc(index[-1]) + yintercept))


def findBullishVCP(sym, df, pivots):
    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmax()
    a = pivots.loc[a_idx, 'P']
    idx = None

    while True:

        b_idx = pivots.loc[a_idx:, 'P'].idxmin()

        # high and low pivots occured on the same date
        if a_idx == b_idx:
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

        # Check if Level D has been breached after it was formed
        if d_idx != df.loc[d_idx:, 'Close'].idxmin():
            # Level D is breached, current pattern is not valid

            # check if D is the last pivot formed
            if pivots.index[-1] == d_idx:
                # if yes, no more pattern possible
                break

            # continue search for patterns
            a_idx = c_idx
            a = c
            continue

        # check if Level C has been breached after it was formed
        if c_idx != df.loc[c_idx:, 'Close'].idxmax():
            # Leve C is breached, current pattern is not valid

            # check if C is the last pivot formed
            if pivots.index[-1] == c_idx:
                break

            # continue search for patterns
            a_idx = c_idx
            a = c
            continue

        df_slice = df.loc[a_idx:c_idx]
        avgCandleLength = (df_slice['High'] - df_slice['Low']).mean()

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[0]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[1]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[0]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[1]

        last = df.loc[df.index[-1], 'Close']

        if bullishVCP(a, b, c, d, last, avgCandleLength):

            if df.loc[d_idx:, 'Close'].max() > c:
                a_idx = c_idx
                a = c
                continue

            plot_args['title'] = f'{sym} - Bull VCP'

            plot_args['alines']['alines'] = (((a_idx, a), (b_idx, b)),
                                             ((b_idx, b), (c_idx, c)),
                                             ((c_idx, c), (d_idx, d)),
                                             ((d_idx, d), (df.index[-1],
                                                           last)))

            print(sym)
            plot(df, **plot_args)
            break

        a_idx = c_idx
        a = c


def findBearishVCP(sym, df, pivots):

    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmin()
    a = pivots.loc[a_idx, 'P']
    idx = None

    while True:
        b_idx = pivots.loc[a_idx:, 'P'].idxmax()

        # high and low pivots occured on the same date
        if a_idx == b_idx:
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

        # Check if Level D has been breached after it was formed
        if d_idx != df.loc[d_idx:, 'Close'].idxmax():
            # Level D is breached, current pattern is not valid

            # check if D is the last pivot formed
            if pivots.index[-1] == d_idx:
                # if yes, no more pattern possible
                break

            # continue search for patterns
            a_idx = c_idx
            a = c
            continue

        # check if Level C has been breached after it was formed
        if c_idx != df.loc[c_idx:, 'Close'].idxmin():
            # Level C is breached, current pattern is not valid

            # check if C is the last pivot formed
            if pivots.index[-1] == c_idx:
                break

            # continue search for patterns
            a_idx = c_idx
            a = c
            continue

        df_slice = df.loc[a_idx:c_idx]
        avgCandleLength = (df_slice['High'] - df_slice['Low']).mean()

        if pivots.index.has_duplicates:
            if isinstance(a, pd.Series | str):
                a = pivots.loc[a_idx, 'P'].iloc[1]

            if isinstance(b, pd.Series | str):
                b = pivots.loc[b_idx, 'P'].iloc[0]

            if isinstance(c, pd.Series | str):
                c = pivots.loc[c_idx, 'P'].iloc[1]

            if isinstance(d, pd.Series | str):
                d = pivots.loc[d_idx, 'P'].iloc[0]

        last = df.at[df.index[-1], 'Close']

        if bearishVCP(a, b, c, d, last, avgCandleLength):
            if df.loc[d_idx:, 'Close'].max() < c:
                a_idx = c_idx
                a = c
                continue

            plot_args['title'] = f'{sym} - Bear VCP'

            plot_args['alines']['alines'] = (((a_idx, a), (b_idx, b)),
                                             ((b_idx, b), (c_idx, c)),
                                             ((c_idx, c), (d_idx, d)),
                                             ((d_idx, d), (df.index[-1],
                                                           last)))

            print(sym)
            plot(df, **plot_args)
            break

        # We assign pivot level C to be the new A
        # This may not be the lowest pivot, so additional checks are required.
        a_idx = c_idx
        a = c


def findDoubleBottom(sym, df, pivots):
    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmin()
    a, aVol = pivots.loc[a_idx, ['P', 'V']]

    while True:
        pos = getNextIndex(pivots.index, a_idx)

        if pos >= pivot_len:
            break

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
        avgCandleLength = (df_slice['High'] - df_slice['Low']).mean()
        last = df.loc[df.index[-1], 'Close']

        if isDoubleBottom(a, b, c, last, aVol, cVol, avgCandleLength):

            if df.loc[c_idx:, 'Close'].max() > b:
                a_idx, a, aVol = c_idx, c, cVol
                continue

            idx = df.loc[:a_idx, 'High'].idxmax()

            # idx = pivots.loc[:a_idx, 'P'].idxmax()
            plot_args['title'] = f'{sym} - Double bottom'

            plot_args['alines']['alines'] = (((idx, df.loc[idx, 'High']),
                                              (a_idx, a)), ((a_idx, a), (b_idx,
                                                                         b)),
                                             ((b_idx, b), (c_idx, c)),
                                             ((c_idx, c), (df.index[-1],
                                                           last)))

            print(sym)
            plot(df, **plot_args)
            break

        a_idx, a, aVol = c_idx, c, cVol


def findDoubleTop(sym, df, pivots):
    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmax()
    a, aVol = pivots.loc[a_idx, ['P', 'V']]

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

        # check if Level C has been breached after it was formed
        if c_idx != df.loc[c_idx:, 'Close'].idxmax():
            # Level C is breached, current pattern is not valid
            a_idx, a, aVol = c_idx, c, cVol
            continue

        b_idx = pivots.loc[a_idx:c_idx, 'P'].idxmin()
        b = pivots.loc[b_idx, 'P']

        if b_idx != pivots.loc[b_idx:, 'P'].idxmin():
            # Level B has been breached
            a_idx, a, aVol = c_idx, c, cVol
            continue

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
        avgCandleLength = (df_slice['High'] - df_slice['Low']).mean()
        last = df.loc[df.index[-1], 'Close']

        if isDoubleTop(a, b, c, last, aVol, cVol, avgCandleLength):

            idx = df.loc[:a_idx, 'Low'].idxmin()

            plot_args['title'] = f'{sym} - Double Top'

            plot_args['alines']['alines'] = (((idx, df.loc[idx, 'Low']),
                                              (a_idx, a)), ((a_idx, a), (b_idx,
                                                                         b)),
                                             ((b_idx, b), (c_idx, c)),
                                             ((c_idx, c), (df.index[-1],
                                                           last)))

            print(sym)
            plot(df, **plot_args)

        a_idx, a, aVol = c_idx, c, cVol


def find(sym, df, pivots, pattern):
    pivot_len = pivots.shape[0]
    a_idx = pivots['P'].idxmax()
    a = pivots.loc[a_idx, 'P']
    idx = None

    while True:
        b_idx = pivots.loc[a_idx:, 'P'].idxmin()
        b = pivots.loc[b_idx, 'P']

        # A is already the lowest point
        if a_idx == b_idx:
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
        avgCandleLength = (df_slice['High'] - df_slice['Low']).mean()

        if pattern == 'HeadAndShoulders' and isHeadAndShoulders(
                a, b, c, d, e, f, avgCandleLength):

            plot_args['title'] = f'{sym} - Bearish Head & Shoulders'
            plot_args['alines']['alines'] = (((a_idx, a), (b_idx, b)),
                                             ((b_idx, b), (c_idx, c)),
                                             ((c_idx, c), (d_idx, d)),
                                             ((d_idx, d), (e_idx, e)))

            print(sym)
            plot(df, **plot_args)
            break

        if pattern == 'ReverseHeadAndShoulders' and isReverseHeadAndShoulders(
                a, b, c, d, e, f, avgCandleLength):

            plot_args['title'] = f'{sym} - Reverse Bullish Head & Shoulders'
            plot_args['alines']['alines'] = (((a_idx, a), (b_idx, b)),
                                             ((b_idx, b), (c_idx, c)),
                                             ((c_idx, c), (d_idx, d)),
                                             ((d_idx, d), (e_idx, e)))

            print(sym)
            plot(df, **plot_args)
            break

        if pattern == 'Pennant' and isPennant(a, b, c, d, e, f,
                                              avgCandleLength):

            if c_idx != df.loc[c_idx:, 'Close'].idxmax():
                a_idx, a = c_idx, c
                continue

            if d_idx != df.loc[d_idx:, 'Close'].idxmin():
                a_idx, c = c_idx, c
                continue

            upper_line = generate_trend_line(df['High'], a_idx, c_idx)
            lower_line = generate_trend_line(df['Low'], b_idx, d_idx)

            if upper_line[1][1] < lower_line[1][1]:
                break

            plot_args['alines'] = ((upper_line), (lower_line))

            plot_args['title'] = f'{sym} - Pennant'

            print(sym)
            plot(df, **plot_args)
            break

        a_idx, c = c_idx, c
