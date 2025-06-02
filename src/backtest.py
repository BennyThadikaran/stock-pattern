import argparse
import concurrent.futures
import importlib
import json
import logging
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Union

import pandas as pd
from tqdm import tqdm

import utils
from loaders.AbstractLoader import AbstractLoader
from Plotter import Plotter

if metadata("fast_csv_loader")["version"] != "2.0.0":
    exit("fast_csv_loader v2.0.0 is required. Run `pip install -U fast_csv_loader")

"""
Backtest.py | Copyright (C) 2023 Benny Thadikaran

This tool allows you to scan for patterns within a specified date range and 
visualize the chart pattern with look-ahead and look-back periods for 
backtesting.

# Basic Usage:
Ensure you have user.json with `DATA_PATH` and `SYM_LIST` set.

```python
py backtest.py -p vcpu -d 2024-10-20
```

# Custom Watchlist
Provide a custom watchlist of symbols as a CSV or TXT file

```python 
py backtest.py -p hnsd -d 2024-10-20 --file symlist.csv
```

# Set Scan parameters
By default, scan end date is 120 days prior to current date. 
Scan period is 30 days prior to end date.

```python 
py backtest.py -p trng --date 2023-12-01 --period 60
```
"""


def uncaught_exception_handler(*args):
    """
    Handle all Uncaught Exceptions

    Function passed to sys.excepthook
    """
    logger.critical("Uncaught Exception", exc_info=args)


def parse_cli_args():
    key_list = (
        "vcpu",
        "vcpd",
        "dbot",
        "dtop",
        "hnsu",
        "hnsd",
        "trng",
        "uptl",
        "dntl",
        "flagu",
        "flagd",
        "abcdu",
        "abcdd",
        "batu",
        "batd",
        "gartu",
        "gartd",
        "crabu",
        "crabd",
        "bflyu",
        "bflyd",
    )

    parser = argparse.ArgumentParser(description="Run backdated pattern scan")

    parser.add_argument(
        "-c",
        "--config",
        type=lambda x: Path(x).expanduser().resolve(),
        default=None,
        metavar="filepath",
        help="Custom config file",
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
        choices=key_list,
        help=f"Pattern to test. One of {', '.join(key_list)}",
    )

    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Index to plot",
    )

    parser.add_argument(
        "-d",
        "--date",
        type=lambda x: datetime.fromisoformat(x) if x else x,
        help=(
            "Last available date on file. The scan end date is 120 periods prior to this date. "
            "Scan start date is 30 days from the scan end date. "
            "Use ISO FORMAT date YYYY-MM-DD."
        ),
    )

    parser.add_argument(
        "--period",
        type=int,
        default=30,
        help="Scan Period, prior to ending date. Default 30",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--plot",
        type=lambda x: json.loads(Path(x).read_bytes()),
        default=None,
        help="Plot results from json file",
    )

    group.add_argument(
        "-f",
        "--file",
        type=lambda x: Path(x).expanduser().resolve(),
        default=None,
        help="Filepath with symbol list. One on each line",
    )

    group.add_argument(
        "--sym",
        nargs="+",
        metavar="SYM",
        help="Space separated list of stock symbols.",
    )

    args = parser.parse_args()

    if not args.pattern and not args.plot:
        raise RuntimeError("Error: the following arguments are required: -p")

    return args


def get_loader_class(config):
    # Load data loader from config. Default loader is EODFileLoader
    loader_name = config.get("LOADER", "EODFileLoader")

    loader_module = importlib.import_module(f"loaders.{loader_name}")

    return getattr(loader_module, loader_name)


def scan(
    loader: AbstractLoader,
    end_dt: Union[datetime, pd.Timestamp],
    scan_period: int,
    fn: str,
    sym: str,
    look_ahead_period: int,
    look_back_period: int,
) -> List[dict]:
    seen = {}
    results = []

    fn_dict = {
        "vcpu": utils.find_bullish_vcp,
        "vcpd": utils.find_bearish_vcp,
        "dbot": utils.find_double_bottom,
        "dtop": utils.find_double_top,
        "trng": utils.find_triangles,
        "hnsd": utils.find_hns,
        "hnsu": utils.find_reverse_hns,
        "uptl": utils.find_uptrend_line,
        "dntl": utils.find_downtrend_line,
        "flagu": utils.find_bullish_flag,
        "flagd": utils.find_bearish_flag,
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

    df = loader.get(sym)

    if df is None or df.empty:
        return results

    if df.index[0].tzinfo:
        end_dt = end_dt.replace(tzinfo=df.index[0].tzinfo)

    if end_dt < df.index[0]:
        return results

    if df.index.has_duplicates:
        df = df[~df.index.duplicated()]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=True)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)

    end_pos = df.index.get_loc(df.index.asof(end_dt))

    if isinstance(end_pos, slice):
        end_pos = end_pos.start

    scan_period_with_look_ahead = scan_period + look_ahead_period

    # If start date is out of bounds, start at first available date in DataFrame
    if end_pos > scan_period_with_look_ahead:
        scan_start_pos = end_pos - scan_period_with_look_ahead
    else:
        scan_start_pos = 0

    scan_start_dt = df.index[scan_start_pos]

    if look_ahead_period > end_pos:
        return results

    scan_end_dt = df.index[end_pos - look_ahead_period]

    assert isinstance(scan_start_dt, pd.Timestamp)

    if fn == "uptl" or fn == "flagu":
        pivot_type = "low"
    elif fn == "dntl" or fn == "flagd":
        pivot_type = "high"
    else:
        pivot_type = "both"

    pivots_all = utils.get_max_min(df, pivot_type=pivot_type)

    for i in df.loc[scan_start_dt:scan_end_dt].index:
        pos = df.index.get_loc(i)

        if isinstance(pos, slice):
            pos = pos.start

        if look_back_period > pos:
            start_idx = df.index[0]
        else:
            start_idx = df.index[-(pos - look_back_period)]

        dfi = df.loc[start_idx : df.index[pos]]

        if not len(dfi):
            break

        pivots = pivots_all.loc[dfi.index[0] : dfi.index[-1]]

        if not len(pivots):
            break

        result = fn_dict[fn](sym, dfi, pivots)

        if result:
            if sym not in seen:
                seen[sym] = {}

            pt_start_dt = result["start"].isoformat()

            if pt_start_dt in seen[sym]:
                continue

            seen[sym][pt_start_dt] = {}
            results.append(utils.make_serializable(result))

    seen.clear()
    return results


def main(
    sym_list: tuple[str, ...],
    out_file: Path,
    loader: AbstractLoader,
    end_date: datetime,
    scan_period: int,
    look_ahead_period: int,
    look_back_period: int,
):
    results: List[dict] = []
    futures = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sym in sym_list:
            future = executor.submit(
                scan,
                loader,
                args.date,
                scan_period,
                args.pattern,
                sym,
                look_ahead_period,
                look_back_period,
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            results.extend(result)

    if len(results):
        logger.info(
            f"Got {len(results)} patterns for {args.pattern.upper()}.\nRun `py backtest.py --plot {out_file.name}` to view results."
        )

        results.append(
            {
                "timeframe": loader.tf,
                "end_date": end_date.isoformat(),
                "period": loader.period,
                "config": str(config_file),
            }
        )
        out_file.write_text(json.dumps(results, indent=2))

        return results
    else:
        logger.info("No patterns found.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logger = logging.getLogger(__name__)

    sys.excepthook = uncaught_exception_handler
    DIR = Path(__file__).parent

    if "-c" in sys.argv or "--config" in sys.argv:
        idx = sys.argv.index("-c" if "-c" in sys.argv else "--config")

        config_file = Path(sys.argv[idx + 1]).expanduser().resolve()
    else:
        config_file = DIR / "user.json"

    if not config_file.exists():
        logger.fatal("Missing user.json. Run setup-config.py to generate user.json")
        exit()

    config = json.loads(config_file.read_bytes())

    sym_list = config["SYM_LIST"] if "SYM_LIST" in config else None

    has_required_args_set = (
        "-f" in sys.argv
        or "--file" in sys.argv
        or "--sym" in sys.argv
        or "--plot" in sys.argv
    )

    if not has_required_args_set:
        sys.argv.extend(("-f", sym_list if sym_list else config["DATA_PATH"]))

    args = parse_cli_args()

    if args.plot:
        meta = args.plot.pop()

        config = json.loads(Path(meta["config"]).expanduser().resolve().read_bytes())

        loader_class = get_loader_class(config)

        loader = loader_class(
            config,
            meta["timeframe"],
            end_date=datetime.fromisoformat(meta["end_date"]),
            period=meta["period"],
        )

        plotter = Plotter(
            args.plot, loader, mode="expand", config=config.get("CHART", {})
        )

        plotter.plot(args.idx)
        exit()

    if args.date is None:
        exit("-d/--date argument is required")

    look_ahead_period = 120
    look_back_period = 160

    loader_class = get_loader_class(config)

    loader = loader_class(
        config,
        args.tf,
        end_date=args.date,
        period=look_back_period + args.period + look_ahead_period,
    )

    if args.file:
        if args.file.is_dir():
            sym_list = tuple(file.name[:-4] for file in args.file.iterdir())
        else:
            sym_list = args.file.read_text().strip().split("\n")
    elif args.sym:
        sym_list = args.sym
    else:
        raise RuntimeError(
            "Error: -f or --file is required. Else define SYM_LIST in user.json"
        )

    output_file = DIR / f"bt_{args.pattern}_{loader.tf}.json"

    result = main(
        sym_list,
        output_file,
        loader,
        args.date,
        args.period,
        look_ahead_period,
        look_back_period,
    )

    if result and config.get("POST_SCAN_PLOT", True):
        result.pop()
        plotter = Plotter(
            result,
            loader,
            mode="expand",
            config=config.get("CHART", {}),
        )

        plotter.plot(args.idx)
