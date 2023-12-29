import utils
import pandas as pd
import json
import logging
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from sys import argv

DIR = Path(__file__).parent
CONFIG_PATH = DIR / 'user.json'

version = '1.1.0-alpha'

if CONFIG_PATH.exists():
    config = json.loads(CONFIG_PATH.read_bytes())
    data_path = Path(config['DATA_PATH']).expanduser()
else:
    help = "DATA_PATH: OHLC folder path, SYM_LIST: Optional stocklist filepath"

    json_content = {"_comment": help, "DATA_PATH": ""}

    CONFIG_PATH.write_text(json.dumps(json_content, indent=2))
    exit("user.json file generated. Edit `DATA_PATH` to add a data source")

if config['DATA_PATH'] == '' or not data_path.exists():
    exit("`DATA_PATH` not found or not provided. Edit user.json.")

sym_list = Path(config['SYM_LIST']) if 'SYM_LIST' in config else None

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
                   type=Path,
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

if sym_list is not None and not ('-f' in argv or '--file' in argv
                                 or '--sym' in argv or '-v' in argv
                                 or '--version' in argv):
    argv.extend(('-f', str(sym_list.expanduser())))

args = parser.parse_args()

logging.basicConfig(level=logging.WARNING if args.silent else logging.INFO,
                    format='%(message)s',
                    datefmt='%d-%m-%Y %H:%M')

utils.set_logger(logging)

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
    logging.info('Date falls on weekend (Sat / Sun)')

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
    7. Pennant (Triangle - Symetrical, Ascending, Descending, Wedges)
    > ''')

        try:
            key = int(key)
        except ValueError:
            print('Enter a number from the list')
            continue
        break

logging.info('Scanning stocks. Press Ctrl - C to exit')

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

        sym_file = data_path / f'{sym.lower()}.csv'

        try:
            df: pd.DataFrame = pd.read_csv(sym_file,
                                           index_col='Date',
                                           parse_dates=['Date'])
        except FileNotFoundError:
            logging.warning(f'File not found: {sym_file}')
            continue

        if args.date:
            dt_index = df.index.date

            # Date is out of bounds
            if args.date.date() not in dt_index:
                continue

            # if has time component for ex. intraday data
            if utils.has_time_component(df.index):
                # get the last available datetime index for the date
                last_idx = df[args.date.date() == dt_index].index.max()
                end = df.index.get_loc(last_idx)
            else:
                # get the index position in the DataFrame
                end = df.index.get_loc(args.date)

            if not isinstance(end, int):
                raise TypeError('Expected integer')

            df = df.iloc[end - 160:end + 1]
        else:
            df = df.iloc[-160:]

        pivots = utils.getMaxMin(df, barsLeft=args.left, barsRight=args.right)

        if not pivots.shape[0]:
            continue

        try:
            fn_dict[key](sym, df, pivots)
        except TypeError as e:
            logging.error('%s', str(e), exc_info=True)
            continue

except KeyboardInterrupt:
    # silent exit without ugly erorr trace
    exit()
