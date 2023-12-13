import utils
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime

parser = ArgumentParser(prog='init.py')

parser.add_argument('-d',
                    '--date',
                    type=datetime.fromisoformat,
                    metavar='str',
                    help='ISO format date YYYY-MM-DD.')

parser.add_argument('-p',
                    '--pattern',
                    type=int,
                    metavar='int',
                    help='A number (1-7) representing the pattern')

args = parser.parse_args()

if args.date and args.date.weekday() in (5, 6):
    exit('Date falls on weekend (Sat / Sun)')

# SET YOUR STOCK DATA SOURCE HERE
data_path = '/home/benny/Documents/python/eod2/src/eod2_data/daily/'

# SET YOUR WATCHLIST FILE HERE
file = '/home/benny/Desktop/data.csv'

if data_path == '' and file == '':
    exit("Set `data_path` and `file before` scanning")

if args.pattern:
    key = args.pattern
else:
    while True:
        key = input('''
    Enter a number to select a pattern.

    1. Bullish VCP (Volatility Contraction pattern)
    2. Bearish VCP
    3. Double Bottom (Bullish)
    4. Double Top (Bearish)
    5. Head and Shoulder
    6. Reverse Head and Shoulder
    7. Pennant (Triangle - Ascending, Descending, Wedges)
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

fn_dict = {
    1: utils.findBullishVCP,
    2: utils.findBearishVCP,
    3: utils.findDoubleBottom,
    4: utils.findDoubleTop,
    5: utils.findHNS,
    6: utils.findReverseHNS,
    7: utils.findPennant,
}

holiday_warn = False

try:
    for sym in data:
        df = pd.read_csv(f'{data_path}/{sym.lower()}.csv',
                         index_col='Date',
                         parse_dates=True)

        if args.date:
            if args.date not in df.index:
                if not holiday_warn:
                    holiday_warn = True
                    print('WARN: Specified date could be a trading holiday')
                continue

            end = df.index.get_loc(args.date)
            df = df[end - 160:end + 1]
        else:
            df = df[-160:]

        pivots = utils.getMaxMin(df)

        if not pivots.shape[0]:
            continue

        fn_dict[key](sym, df, pivots)
except KeyboardInterrupt:
    # silent exit without ugly erorr trace
    exit()
