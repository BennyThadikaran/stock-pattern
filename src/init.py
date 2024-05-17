import sys
import json
import logging
import concurrent.futures
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union, Dict
from Plotter import Plotter
import utils

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    exit("tqdm is required. Run `pip install tqdm` to install")

futures: List[concurrent.futures.Future] = []

config_help = """
Config help
DATA_PATH: Folder path for OHLC csv data.

SYM_LIST: Optional file with list of symbols, one per line.

SAVE_FOLDER: Optional folder path to save charts as images.

POST_SCAN_PLOT: If True, plots the results on chart, after a scan.

SAVE_STATE: If True, previously detected patterns will not be displayed in
subsequent scans.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


def uncaught_exception_handler(*args):
    """
    Handle all Uncaught Exceptions

    Function passed to sys.excepthook
    """
    logger.critical("Uncaught Exception", exc_info=args)


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
    date: Optional[datetime],
    fns: Tuple[Callable, ...],
    bars_left=6,
    bars_right=6,
):
    patterns: List[dict] = []

    if date:
        try:
            df = utils.csv_loader(file, end_date=date)
        except IndexError:
            return patterns
    else:
        df = utils.csv_loader(file)

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

    # Load or initialize state dict for storing previously detected patterns
    state = None
    state_file = None

    if config.get("SAVE_STATE", False) and args.file and not args.date:
        state_file = DIR / f"state/{args.file.stem}_{args.pattern}.json"

        if not state_file.parent.is_dir():
            state_file.parent.mkdir(parents=True)

        if state_file.exists():
            state = json.loads(state_file.read_bytes())
        else:
            state = {}

    # determine the folder to save to in case save option is set
    save_folder: Optional[Path] = None
    image_folder = f"{datetime.now():%d_%b_%y_%H%M}"

    if "SAVE_FOLDER" in config:
        save_folder = Path(config["SAVE_FOLDER"]) / image_folder

    if args.save:
        save_folder = args.save / image_folder

    if save_folder and not save_folder.exists():
        save_folder.mkdir(parents=True)

    # begin scan process
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sym in sym_list:
            file = data_path / f"{sym.lower()}.csv"

            if not file.exists():
                logger.warning(f"File not found: {file}")
                continue

            future = executor.submit(scan_pattern, sym, file, args.date, fns)

            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            patterns.extend(result)

        if state is None:
            # if no args.file option, no need to save state
            return patterns

        ####
        #   Filter for newly detected patterns and remove stale patterns
        ####

        # list for storing newly detected patterns
        filtered = []

        # initial length of state dict
        len_state = len(state)

        # Will contain keys to all patterns currently detected
        detected = set()

        for dct in patterns:
            # unique identifier
            key = f'{dct["sym"]}-{dct["pattern"]}'

            detected.add(key)

            if not len_state:
                # if state is empty, this is a first run
                # no need to filter
                state[key] = dct
                filtered.append(dct)
                continue

            if key in state:
                if dct["start"] == state[key]["start"]:
                    # if the pattern starts on the same date,
                    # they are the same previously detected pattern
                    continue

                # Else there is a new pattern for the same key
                state[key] = dct
                filtered.append(dct)

            # new pattern
            filtered.append(dct)
            state[key] = dct

        # set difference - get keys in state dict not existing in detected
        # These are pattern keys no longer detected and can be removed
        invalid_patterns = set(state.keys()) - detected

        # Clean up stale patterns in state dict
        for key in invalid_patterns:
            state.pop(key)

        if state_file:
            state_file.write_text(json.dumps(state, indent=2))

        # Save the images if required
        if save_folder and len(filtered):
            plotter = Plotter(None, data_path, save_folder=save_folder)
            futures = []

            for i in filtered:
                future = executor.submit(plotter.save, i.copy())
                futures.append(future)

            logger.info("Saving images")

            for _ in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                pass

        if state_file:
            logger.info(
                f"\nTo view all current market patterns, run `py init.py --plot state/{state_file.name}\n"
            )

        return filtered


if __name__ == "__main__":
    version = "2.1.10"

    sys.excepthook = uncaught_exception_handler

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
            "SAVE_STATE": False,
        }

        CONFIG_PATH.write_text(json.dumps(json_content, indent=2))

        print("user.json file generated. Edit `DATA_PATH` to add a data source")

        print(config_help)

        exit()

    if config["DATA_PATH"] == "" or not data_path.exists():
        exit("`DATA_PATH` not found or not provided. Edit user.json.")

    sym_list = Path(config["SYM_LIST"]) if "SYM_LIST" in config else None

    key_list = (
        "all",
        "bull",
        "bear",
        "vcpu",
        "dbot",
        "hnsu",
        "vcpd",
        "dtop",
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
        type=lambda x: Path(x).expanduser().resolve(),
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
        "-f" in sys.argv
        or "--file" in sys.argv
        or "--sym" in sys.argv
        or "-v" in sys.argv
        or "--version" in sys.argv
        or "--plot" in sys.argv
    ):
        sys.argv.extend(("-f", str(sym_list)))

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

        args.pattern = key

    fn = fn_dict[key]

    logger.info(f"Scanning `{key.upper()}` patterns. Press Ctrl - C to exit")

    data = args.file.read_text().strip().split("\n") if args.file else args.sym

    patterns: List[dict] = []

    if callable(fn):
        fns = (fn,)
    elif fn == "bull":
        fns = tuple(
            v for k, v in fn_dict.items() if k in key_list[3:6] and callable(v)
        )
    elif fn == "bear":
        fns = tuple(
            v for k, v in fn_dict.items() if k in key_list[6:9] and callable(v)
        )
    else:
        fns = tuple(
            v for k, v in fn_dict.items() if k in key_list[3:] and callable(v)
        )

    try:
        patterns = process(data, fns)
    except KeyboardInterrupt:
        # Is there a better / graceful way to handle this?
        logger.info("User exit")
        exit()

    count = len(patterns)

    if count == 0:
        exit("No patterns detected")

    (DIR / f"{key.lower()}.json").write_text(json.dumps(patterns, indent=2))

    logger.info(
        f"Got {count} patterns for `{key}`.\n\nRun `py init.py --plot {key.lower()}.json` to view results.\n"
    )

    if config["POST_SCAN_PLOT"]:
        plotter = Plotter(patterns, data_path)

        plotter.plot()
