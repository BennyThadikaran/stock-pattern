import utils
import pandas as pd

# SET YOUR STOCK DATA SOURCE HERE
data_path = '/home/benny/Documents/python/eod2/src/eod2_data/daily/'

# SET YOUR WATCHLIST FILE HERE
file = '/home/benny/Desktop/data.csv'

if data_path == '' and file == '':
    exit("Set `data_path` and `file before` scanning")

while True:
    key = input('''
Enter a number to select a pattern.

1. Bullish VCP (Volatility Contraction pattern)
2. Bearish VCP
3. Double Bottom (Bullish)
4. Double Top (Bearish)
5. Head and Shoulder (Untested)
6. Reverse Head and Shoulder (Untested)
7. Pennant (Ascending, Descending, Wedges)
''')

    try:
        key = int(key)
    except ValueError:
        print('Enter a number from the list')
        continue
    break

print('Scanning stocks. Press Ctrl - C to exit')

with open(file) as f:
    data = f.read().strip().split('\n')

try:
    for sym in data:
        df = pd.read_csv(f'{data_path}/{sym.lower()}.csv',
                         index_col='Date',
                         parse_dates=True)[-160:]

        pivots = utils.getMaxMin(df)[-15:]

        if not pivots.shape[0]:
            continue

        if key == 1:
            utils.findBullishVCP(sym, df, pivots)
        elif key == 2:
            utils.findBearishVCP(sym, df, pivots)
        elif key == 3:
            utils.findDoubleBottom(sym, df, pivots)
        elif key == 4:
            utils.findDoubleTop(sym, df, pivots)
        elif key == 5:
            utils.find(sym, df, pivots, pattern='HeadAndShoulder')
        elif key == 6:
            utils.find(sym, df, pivots, pattern='ReverseHeadAndShoulder')
        elif key == 7:
            utils.find(sym, df, pivots, pattern='Pennant')
except KeyboardInterrupt:
    # silent exit without ugly erorr trace
    exit()
