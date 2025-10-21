import concurrent.futures
import importlib
import json
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import questionary

import utils
from loaders.AbstractLoader import AbstractLoader
from Plotter import Plotter

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    exit("tqdm is required. Run `pip install tqdm` to install")

if metadata("fast_csv_loader")["version"] != "2.0.0":
    exit("fast_csv_loader v2.0.0 is required. Run `pip install -U fast_csv_loader")


def uncaught_exception_handler(*args):
    """
    Handle all Uncaught Exceptions

    Function passed to sys.excepthook
    """
    logger.critical("Uncaught Exception", exc_info=args)
    cleanup(loader, futures)


def get_user_selection() -> Optional[str]:
    while True:
        ptn_key = questionary.select(
            message="Select an option [ ** Excludes Trendlines & Triangles & Harmonics ]",
            choices=[
                questionary.Choice("[ALL] Scan All **", value="all"),
                questionary.Choice("[BULL] Scan only Bull **", value="bull"),
                questionary.Choice("[BEAR] Scan only Bear **", value="bear"),
                questionary.Choice(
                    "[BULL_HARM] All BULL Harmonic pattern", value="bull_harm"
                ),
                questionary.Choice(
                    "[BEAR_HARM] All BEAR Harmonic pattern", value="bear_harm"
                ),
                questionary.Choice(
                    "[TRNG] Triangles (Symetrical, Ascending, Descending)",
                    value="trng",
                ),
                questionary.Choice("[UPTL] Uptrend lines", value="uptl"),
                questionary.Choice("[DNTL] Downtrend lines", value="dntl"),
                questionary.Choice("SELECT BULL pattern", value="SELECT_BULL"),
                questionary.Choice("SELECT BEAR pattern", value="SELECT_BEAR"),
            ],
            use_shortcuts=True,
        ).ask()

        if ptn_key == "SELECT_BULL":
            ptn_key = questionary.select(
                message="Select a BULL pattern",
                choices=[
                    questionary.Choice("[FLAGU] High Pole and Flag", value="flagu"),
                    questionary.Choice("[VCPU] Volatility Contraction", value="vcpu"),
                    questionary.Choice("[DBOT] Double Bottom", value="dbot"),
                    questionary.Choice("[HSNU] Head & Shoulders", value="hnsu"),
                    questionary.Choice("[ABCDU] AB=CD Harmonic", value="abcdu"),
                    questionary.Choice("[BATU] BAT Harmonic", value="batu"),
                    questionary.Choice("[GARTU] Gartley Harmonic", value="gartu"),
                    questionary.Choice("[CRABU] Crab Harmonic", value="crabu"),
                    questionary.Choice("[BFLYU] Butterfly Harmonic", value="bflyu"),
                    questionary.Choice("Back to main", value="MAIN"),
                ],
                use_shortcuts=True,
            ).ask()
        elif ptn_key == "SELECT_BEAR":
            ptn_key = questionary.select(
                message="Select a BEAR pattern",
                choices=[
                    questionary.Choice("[FLAGD] High Pole and Flag", value="flagd"),
                    questionary.Choice("[VCPD] Volatility Contraction", value="vcpd"),
                    questionary.Choice("[DTOP] Double Top", value="dtop"),
                    questionary.Choice("[HNSD] Head & Shoulders", value="hnsd"),
                    questionary.Choice("[ABCDD] AB=CD Harmonic", value="abcdd"),
                    questionary.Choice("[BATD] BAT Harmonic", value="batd"),
                    questionary.Choice("[GARTD] Gartley Harmonic", value="gartd"),
                    questionary.Choice("[CRABD] Crab Harmonic", value="crabd"),
                    questionary.Choice("[BFLYD] Butterfly Harmonic", value="bflyd"),
                    questionary.Choice("Back to main", value="MAIN"),
                ],
                use_shortcuts=True,
            ).ask()

        if ptn_key == "MAIN":
            continue

        return ptn_key


def get_loader_class(config):
    # Load data loader from config. Default loader is EODFileLoader
    loader_name = config.get("LOADER", "EODFileLoader")

    loader_module = importlib.import_module(f"loaders.{loader_name}")

    return getattr(loader_module, loader_name)


def cleanup(loader: AbstractLoader, futures: List[concurrent.futures.Future]):
    if futures:
        for future in futures:
            future.cancel()

        concurrent.futures.wait(futures)

    if loader.closed:
        loader.close()


def scan_pattern(
    sym: str,
    pattern: str,
    fns: Tuple[Callable, ...],
    loader: AbstractLoader,
    logger: logging.Logger,
    config: dict,
    bars_left=6,
    bars_right=6,
):
    patterns: List[dict] = []

    df = loader.get(sym)

    if df is None or df.empty:
        return patterns

    if df.index.has_duplicates:
        df = df.loc[~df.index.duplicated()]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=True)

    if pattern == "uptl" or pattern == "flagu":
        pivot_type = "low"
    elif pattern == "dntl" or pattern == "flagd":
        pivot_type = "high"
    else:
        pivot_type = "both"

    pivots = utils.get_max_min(
        df, barsLeft=bars_left, barsRight=bars_right, pivot_type=pivot_type
    )

    if not len(pivots):
        return patterns

    for fn in fns:
        if not callable(fn):
            raise TypeError(f"Expected callable. Got {type(fn)}")

        try:
            result = fn(sym, df, pivots, config)
        except Exception as e:
            logger.exception(f"SYMBOL name: {sym}", exc_info=e)
            return patterns

        if result:
            patterns.append(utils.make_serializable(result))

    return patterns


def process(
    sym_list: Tuple[str, ...],
    pattern: str,
    fns: Tuple[Callable, ...],
    futures: List[concurrent.futures.Future],
) -> List[dict]:
    patterns: List[dict] = []

    # Load or initialize state dict for storing previously detected patterns
    state = None
    state_file = None
    filtered = None

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
            future = executor.submit(
                scan_pattern,
                sym,
                pattern,
                fns,
                loader,
                logger,
                config,
                bars_left=args.left,
                bars_right=args.right,
            )

            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
        ):
            try:
                result = future.result()
            except Exception as e:
                cleanup(loader, futures)
                logger.exception("Error in Future - scaning patterns", exc_info=e)
                return []

            patterns.extend(result)

        futures.clear()

        if state is not None:
            # if no args.file option, no need to save state, return patterns
            # Filter for newly detected patterns and remove stale patterns

            # list for storing newly detected patterns
            filtered = []

            # initial length of state dict
            len_state = len(state)

            # Will contain keys to all patterns currently detected
            detected = set()

            for dct in patterns:
                # unique identifier
                key = f"{dct['sym']}-{dct['pattern']}"

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

                logger.info(
                    f"\nTo view all current market patterns, run `py init.py --plot state/{state_file.name}\n"
                )

        patterns_to_output = patterns if state is None else filtered

        if not patterns_to_output:
            return []

        # Save the images if required
        if save_folder:
            plotter = Plotter(
                patterns_to_output,
                loader,
                save_folder=save_folder,
                config=config.get("CHART", {}),
            )

            for i in range(len(patterns_to_output)):
                future = executor.submit(plotter.plot, i)
                futures.append(future)

            logger.info("Saving images")

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                try:
                    future.result()
                except Exception as e:
                    cleanup(loader, futures)
                    logger.exception("Error in Futures - Saving images ", exc_info=e)
                    return []

    patterns_to_output.append(
        {
            "timeframe": loader.tf,
            "end_date": args.date.isoformat() if args.date else None,
            "config": str(CONFIG_PATH),
        }
    )

    return patterns_to_output


# START

# Differentiate between the main thread and child threads on Windows
# see https://stackoverflow.com/a/57811249
if __name__ == "__main__":
    version = "4.1.0"

    futures: List[concurrent.futures.Future] = []

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger = logging.getLogger(__name__)

    sys.excepthook = uncaught_exception_handler

    # Load configuration
    DIR = Path(__file__).parent

    fn_dict: Dict[str, Callable] = {
        "flagu": utils.find_bullish_flag,
        "flagd": utils.find_bearish_flag,
        "vcpu": utils.find_bullish_vcp,
        "dbot": utils.find_double_bottom,
        "hnsu": utils.find_reverse_hns,
        "vcpd": utils.find_bearish_vcp,
        "dtop": utils.find_double_top,
        "hnsd": utils.find_hns,
        "trng": utils.find_triangles,
        "uptl": utils.find_uptrend_line,
        "dntl": utils.find_downtrend_line,
        "abcdu": utils.find_bullish_abcd,
        "abcdd": utils.find_bearish_abcd,
        "batu": utils.find_bullish_bat,
        "batd": utils.find_bearish_bat,
        "gartu": utils.find_bullish_gartley,
        "gartd": utils.find_bearish_gartley,
        "crabu": utils.find_bullish_crab,
        "crabd": utils.find_bearish_crab,
        "bflyu": utils.find_bullish_butterfly,
        "bflyd": utils.find_bearish_butterfly,
    }

    # Parse CLI arguments
    parser = ArgumentParser(
        description="Python CLI tool to identify common Chart patterns. Includes Harmonic chart patterns.",
        epilog="https://github.com/BennyThadikaran/stock-pattern",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=lambda x: Path(x).expanduser().resolve(),
        default=None,
        metavar="filepath",
        help="Custom config file",
    )

    parser.add_argument(
        "-d",
        "--date",
        type=datetime.fromisoformat,
        metavar="str",
        help="ISO format date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--tf",
        action="store",
        help="Timeframe string.",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        metavar="str",
        help=f"String pattern. One of {', '.join(fn_dict.keys())} or bull, bear, bull_harm, bear_harm.",
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
        type=lambda x: Path(x).expanduser().resolve(),
        default=None,
        help="Plot results from json file",
    )

    if "-c" in sys.argv or "--config" in sys.argv:
        idx = sys.argv.index("-c" if "-c" in sys.argv else "--config")

        CONFIG_PATH = Path(sys.argv[idx + 1]).expanduser().resolve()
    else:
        CONFIG_PATH = DIR / "user.json"

    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_bytes())

        data_path = Path(config["DATA_PATH"]).expanduser()
    else:
        print(
            "Configuration file is missing. Run `setup-config.py` to setup a `user.json` file"
        )
        exit()

    if config["DATA_PATH"] == "" or not data_path.exists():
        exit("`DATA_PATH` not found or not provided. Edit user.json.")

    sym_list = config["SYM_LIST"] if "SYM_LIST" in config else None

    has_required_arg_set = (
        "-f" in sys.argv
        or "--file" in sys.argv
        or "--sym" in sys.argv
        or "-v" in sys.argv
        or "--version" in sys.argv
        or "--plot" in sys.argv
    )

    if not has_required_arg_set:
        sys.argv.extend(("-f", sym_list if sym_list else config["DATA_PATH"]))

    args = parser.parse_args()

    # Print version
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

    if args.plot:
        data = json.loads(args.plot.read_bytes())

        # Last item contains meta data about the timeframe used, end_date etc
        meta = data.pop()

        config = json.loads(Path(meta["config"]).expanduser().resolve().read_bytes())

        loader_class = get_loader_class(config)

        end_date = None

        if meta["end_date"]:
            end_date = datetime.fromisoformat(meta["end_date"])

        try:
            loader = loader_class(
                config,
                meta["timeframe"],
                end_date=end_date,
            )
        except ValueError as e:
            logger.exception("", exc_info=e)
            exit()

        plotter = Plotter(data, loader, config=config.get("CHART", {}))
        plotter.plot(args.idx)
        cleanup(loader, futures)
        exit()

    loader_class = get_loader_class(config)

    try:
        loader = loader_class(
            config,
            args.tf,
            end_date=args.date,
        )
    except ValueError as e:
        logger.exception("", exc_info=e)
        exit()

    if args.pattern:
        key = args.pattern
    else:
        key = get_user_selection()

        if key is None:
            cleanup(loader, futures)
            exit()

        args.pattern = key

    logger.info(
        f"Scanning `{key.upper()}` patterns on `{loader.tf}`. Press Ctrl - C to exit"
    )

    if args.file:
        if args.file.is_dir():
            data = tuple(file.name[:-4] for file in args.file.iterdir())
        else:
            data = tuple(args.file.read_text().strip().split("\n"))
    else:
        data = tuple(args.sym)

    patterns: List[dict] = []

    if "PATTERNS" in config and key in config["PATTERNS"]:
        custom_list = config["PATTERNS"][key]
        fns = []

        for k in custom_list:
            if k not in fn_dict:
                raise KeyError(f"No such pattern defined: {k}")
            fns.append(fn_dict[k])

        fns = tuple(fns)
    elif key in fn_dict:
        fns = (fn_dict[key],)
    elif key == "bull":
        bull_list = ("vcpu", "hnsu", "dbot", "flagu")

        fns = tuple(v for k, v in fn_dict.items() if k in bull_list)
    elif key == "bear":
        bear_list = ("vcpd", "hnsd", "dtop", "flagd")

        fns = tuple(v for k, v in fn_dict.items() if k in bear_list)
    elif key == "bull_harm":
        bull_list = ("abcdu", "batu", "gartu", "crabu", "bflyu")

        fns = tuple(v for k, v in fn_dict.items() if k in bull_list)
    elif key == "bear_harm":
        bear_list = ("abcdd", "batd", "gartd", "crabd", "bflyd")

        fns = tuple(v for k, v in fn_dict.items() if k in bear_list)
    elif key == "all":
        fns = tuple(
            fn_dict[k]
            for k in ("vcpu", "hnsu", "dbot", "flagu", "vcpd", "hnsd", "dtop", "flagd")
        )
    else:
        raise KeyError(f"{key} did not match any defined patterns.")

    try:
        patterns = process(data, key, fns, futures)
    except KeyboardInterrupt:
        cleanup(loader, futures)
        logger.info("User exit")
        exit()

    count = len(patterns)

    if count == 0:
        cleanup(loader, futures)
        exit("No patterns detected")

    fname = f"{key.lower()}-{loader.tf}.json"

    (DIR / fname).write_text(json.dumps(patterns, indent=2))

    logger.info(
        f"Got {count - 1} patterns for `{key}`.\n\nRun `py init.py --plot {fname}` to view results."
    )

    if config.get("POST_SCAN_PLOT", True) and not args.save:
        # last item in patterns is the meta data like timeframe, end_date etc
        # Pop it out as we don't require it here
        patterns.pop()

        plotter = Plotter(patterns, loader, config=config.get("CHART", {}))
        plotter.plot()

    cleanup(loader, futures)
