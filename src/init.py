import utils
import pandas as pd
import pathlib
from argparse import ArgumentParser
from datetime import datetime

# # SET YOUR STOCK DATA SOURCE HERE
data_path = ''

#
# #                           ###
# # DO NOT EDIT BELOW THIS LINE #
# #                           ###
if data_path == '':
    exit("Set `data_path` in init.py before` scanning")

version = '0.2.2-alpha'

parser = ArgumentParser(
    description='Python CLI tool to identify common Chart patterns',
    epilog='https://github.com/BennyThadikaran/stock-pattern')

parser.add_argument('-s',
                    '--silent',
                    action='store_true',
                    help='Only print symbols. No chart plots')

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

parser.add_argument('-l',
                    '--left',
                    type=int,
                    metavar='int',
                    default=6,
                    help='Number of candle bars on left side of pivot')

parser.add_argument('-r',
                    '--right',
                    type=int,
                    metavar='int',
                    default=6,
                    help='Number of candle bars on right side of pivot')

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('-f',
                   '--file',
                   type=pathlib.Path,
                   metavar='filepath',
                   help='File containing list of stocks. One on each line')

group.add_argument('--sym',
                   nargs='+',
                   metavar='SYM',
                   help='Space separated list of stock symbols.')

group.add_argument('-v',
                   '--version',
                   action='store_true',
                   help='Print the current version.')

args = parser.parse_args()

if args.version:
    exit(f'''
    Stock-Pattern | Version {version}
    Copyright (C) 2023 Benny Thadikaran 

    Github: https://github.com/BennyThadikaran/stock-pattern

    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions.
    See license: https://www.gnu.org/licenses/gpl-3.0.en.html#license-text
    ''')

if args.date and args.date.weekday() in (5, 6):
    exit('Date falls on weekend (Sat / Sun)')

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
    > ''')

        try:
            key = int(key)
        except ValueError:
            print('Enter a number from the list')
            continue
        break

print('Scanning stocks. Press Ctrl - C to exit')

data = args.file.read_text().strip().split('\n') if args.file else args.sym

fn_dict = {
    1: utils.findBullishVCP,
    2: utils.findBearishVCP,
    3: utils.findDoubleBottom,
    4: utils.findDoubleTop,
    5: utils.findHNS,
    6: utils.findReverseHNS,
    7: utils.findPennant,
}

try:
    for sym in data:

        sym_file = f'{data_path}/{sym.lower()}.csv'

        try:
            df = pd.read_csv(sym_file, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f'File not found: {sym_file}')
            continue

        if args.date:
            # Date is out of bounds
            if args.date < df.index[0] or args.date > df.index[-1]:
                continue

            if args.date not in df.index:
                continue

            end = df.index.get_loc(args.date)
            df = df[end - 160:end + 1]
        else:
            df = df[-160:]

        pivots = utils.getMaxMin(df, barsLeft=args.left, barsRight=args.right)

        if not pivots.shape[0]:
            continue

        fn_dict[key](sym, df, pivots, args.silent)
except KeyboardInterrupt:
    # silent exit without ugly erorr trace
    exit()
