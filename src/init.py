import json
import pandas as pd
import concurrent.futures
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from typing import Callable, List, Tuple, Union, Dict
from sys import argv
from Plotter import Plotter
import utils

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    exit("tqdm is required. Run `pip install tqdm` to install")


def get_user_input() -> str:
    user_input = input(
        """
    Enter a number to select a pattern.

    0. Scan all patterns
    1: All Bullish patterns
    2: All Bearish patterns
    3. Bullish VCP (Volatility Contraction pattern)
    4. Bearish VCP
    5. Double Bottom (Bullish)
    6. Double Top (Bearish)
    7. Head and Shoulder
    8. Reverse Head and Shoulder
    9. Triangles (Symetrical, Ascending, Descending)
    > """
    )

    if not (user_input.isdigit() and int(user_input) in range(10)):
        print("Enter a key from the list")
        return get_user_input()

    return user_input


def scan_pattern(
    sym: str,
    file: Path,
    date: Union[datetime, None],
    fns: Tuple[Callable, ...],
    bars_left=6,
    bars_right=6,
):
    patterns: List[dict] = []

    df = utils.get_DataFrame(file)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected pd.DatetimeIndex")

    if date:
        dt_index = df.index.date

        # Date is out of bounds
        if date.date() not in dt_index:
            date = df.index.asof(date)

        # if has time component for ex. intraday data
        if utils.has_time_component(df.index):
            # get the last available datetime index for the date
            last_idx = df[date.date() == dt_index].index.max()
            end = df.index.get_loc(last_idx)
        else:
            # get the index position in the DataFrame
            end = df.index.get_loc(date)

        if not isinstance(end, int):
            raise TypeError("Expected integer")

        df = df.iloc[end - 160 : end + 1]
    else:
        df = df.iloc[-160:]

    if df.empty:
        return patterns

    pivots = utils.get_max_min(df, barsLeft=bars_left, barsRight=bars_right)

    if not pivots.shape[0]:
        return patterns

    for fn in fns:
        if not callable(fn):
            raise TypeError(f"Expected callable. Got {type(fn)}")

        result = fn(sym, df, pivots)

        if result:
            patterns.append(utils.make_serializable(result))

    return patterns


def process(sym_list: List, fns: Tuple[Callable, ...]) -> List[dict]:
    patterns: List[dict] = []
    futures = []

    save_folder: Union[Path, None] = None

    image_folder = f"{datetime.now():%d_%b_%y_%H%M}"

    if "SAVE_FOLDER" in config:
        save_folder = Path(config["SAVE_FOLDER"]) / image_folder

    if args.save:
        save_folder = args.save / image_folder

    if save_folder and not save_folder.exists():
        save_folder.mkdir(parents=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sym in sym_list:
            file = data_path / f"{sym.lower()}.csv"

            if not file.exists():
                utils.logging.warning(f"File not found: {file}")
                continue

            future = executor.submit(scan_pattern, sym, file, args.date, fns)

            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            patterns.extend(result)

        if save_folder and len(patterns):
            plotter = Plotter(None, data_path, save_folder=save_folder)
            futures = []

            for i in patterns:
                future = executor.submit(plotter.save, i.copy())
                futures.append(future)

            utils.logging.info("Saving images")

            for _ in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                pass

        return patterns


if __name__ == "__main__":
    version = "2.0.0-alpha"

    # Run the below code only when imported
    DIR = Path(__file__).parent
    CONFIG_PATH = DIR / "user.json"

    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_bytes())

        data_path = Path(config["DATA_PATH"]).expanduser()
    else:
        json_content = {
            "DATA_PATH": "",
            "POST_SCAN_PLOT": True,
        }

        CONFIG_PATH.write_text(json.dumps(json_content, indent=2))

        print("user.json file generated. Edit `DATA_PATH` to add a data source")

        print(
            "\nConfig help\n",
            "DATA_PATH: Folder path for OHLC csv data.\n\n",
            "SYM_LIST: Optional file with list of symbols, one per line.\n\n",
            "SAVE_FOLDER: Optional folder path to save charts as images.\n\n",
            "POST_SCAN_PLOT: If True, plots the results on chart, after a scan.",
        )

        exit()

    if config["DATA_PATH"] == "" or not data_path.exists():
        exit("`DATA_PATH` not found or not provided. Edit user.json.")

    sym_list = Path(config["SYM_LIST"]) if "SYM_LIST" in config else None

    key_list = (
        "all",
        "bull",
        "bear",
        "vcpu",
        "vcpd",
        "dbot",
        "dtop",
        "hnsu",
        "hnsd",
        "trng",
    )

    parser = ArgumentParser(
        description="Python CLI tool to identify common Chart patterns",
        epilog="https://github.com/BennyThadikaran/stock-pattern",
    )

    parser.add_argument(
        "-d",
        "--date",
        type=datetime.fromisoformat,
        metavar="str",
        help="ISO format date YYYY-MM-DD.",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        metavar="str",
        choices=key_list,
        help=f"String pattern. One of {', '.join(key_list)}",
    )

    parser.add_argument(
        "-l",
        "--left",
        type=int,
        metavar="int",
        default=6,
        help="Number of candles on left side of pivot",
    )

    parser.add_argument(
        "-r",
        "--right",
        type=int,
        metavar="int",
        default=6,
        help="Number of candles on right side of pivot",
    )

    parser.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=DIR / "images",
        help="Specify the save directory",
    )

    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Index to plot",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-f",
        "--file",
        type=lambda x: Path(x)
        .expanduser()
        .resolve()
        .read_text()
        .strip()
        .split("\n"),
        default=None,
        metavar="filepath",
        help="File containing list of stocks. One on each line",
    )

    group.add_argument(
        "--sym",
        nargs="+",
        metavar="SYM",
        help="Space separated list of stock symbols.",
    )

    group.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print the current version.",
    )

    group.add_argument(
        "--plot",
        type=lambda x: json.loads(Path(x).expanduser().resolve().read_bytes()),
        default=None,
        help="Plot results from json file",
    )

    if sym_list is not None and not (
        "-f" in argv
        or "--file" in argv
        or "--sym" in argv
        or "-v" in argv
        or "--version" in argv
        or "--plot" in argv
    ):
        argv.extend(("-f", str(sym_list)))

    args = parser.parse_args()

    if args.plot:
        plotter = Plotter(args.plot, data_path)
        plotter.plot(args.idx)
        exit()

    if args.version:
        exit(
            f"""
        Stock-Pattern | Version {version}
        Copyright (C) 2023 Benny Thadikaran 

        Github: https://github.com/BennyThadikaran/stock-pattern

        This program comes with ABSOLUTELY NO WARRANTY.
        This is free software, and you are welcome to redistribute it
        under certain conditions.
        See license: https://www.gnu.org/licenses/gpl-3.0.en.html#license-text
        """
        )

    fn_dict: Dict[str, Union[str, Callable]] = {
        "all": "all",
        "bull": "bull",
        "bear": "bear",
        "vcpu": utils.find_bullish_vcp,
        "dbot": utils.find_double_bottom,
        "hnsu": utils.find_reverse_hns,
        "vcpd": utils.find_bearish_vcp,
        "dtop": utils.find_double_top,
        "hnsd": utils.find_hns,
        "trng": utils.find_triangles,
    }

    if args.pattern:
        key: str = args.pattern
    else:
        user_input = get_user_input()

        key = key_list[int(user_input)]

    fn = fn_dict[key]

    utils.logging.info(
        f"Scanning for all {key.upper()} patterns. Press Ctrl - C to exit"
    )

    data = args.file if args.file else args.sym

    patterns: List[dict] = []

    if callable(fn):
        fns = (fn,)
    elif fn == "bull":
        fns = tuple(
            val
            for key, val in fn_dict.items()
            if key in key_list[3:6] and callable(val)
        )
    elif fn == "bear":
        fns = tuple(
            val
            for key, val in fn_dict.items()
            if key in key_list[6:9] and callable(val)
        )
    else:
        fns = tuple(
            val
            for key, val in fn_dict.items()
            if key in key_list[3:] and callable(val)
        )

    patterns = process(data, fns)

    count = len(patterns)

    if count == 0:
        exit("No patterns detected")

    utils.logging.info(
        f"Got {count} patterns for {key}.\n\nRun `py init.py --plot {key.lower()}.json` to view results."
    )

    (DIR / f"{key.lower()}.json").write_text(json.dumps(patterns, indent=2))

    if config["POST_SCAN_PLOT"]:
        plotter = Plotter(patterns, data_path)

        plotter.plot()
