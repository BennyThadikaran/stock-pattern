import argparse
import concurrent.futures
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pandas as pd
from tqdm import tqdm

import utils
from loaders.AbstractLoader import AbstractLoader
from Plotter import Plotter

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
        "abcdu",
        "abcdd",
        "batu",
        "batd",
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


def scan(
    loader: AbstractLoader,
    end: Union[datetime, pd.Timestamp],
    period: int,
    fn: str,
    sym: str,
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
        "abcdu": utils.find_bullish_abcd,
        "abcdd": utils.find_bearish_abcd,
        "batu": utils.find_bullish_bat,
        "batd": utils.find_bearish_bat,
    }

    df = loader.get(sym)

    if df is None or df.empty or end < df.index[0]:
        return results

    if df.index.has_duplicates:
        df = df[~df.index.duplicated()]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=True)

    assert isinstance(df.index, pd.DatetimeIndex)

    pos = df.index.get_loc(df.index.asof(end))

    if isinstance(pos, slice):
        pos = pos.start

    # If start date is out of bounds, start at first available date in DataFrame
    start = df.index[pos - period if pos > period else 0]

    if start < df.index[0]:
        return results

    assert isinstance(start, pd.Timestamp)

    dt_index = df.index.date

    has_time_component = utils.has_time_component(df.index)

    if has_time_component:
        start_dt = df.loc[start.date() == dt_index].index.max()
        end_dt = df.loc[end.date() == dt_index].index.max()
    else:
        start_dt = start
        end_dt = end

    start_pos = df.index.get_loc(df.index.asof(start_dt))
    end_pos = df.index.get_loc(df.index.asof(end_dt))

    if isinstance(start_pos, slice):
        start_pos = start_pos.start

    if isinstance(end_pos, slice):
        end_pos = end_pos.start

    if fn == "uptl":
        pivot_type = "low"
    elif fn == "dntl":
        pivot_type = "high"
    else:
        pivot_type = "both"

    pivots_all = utils.get_max_min(
        df.iloc[start_pos - 160 : end_pos], pivot_type=pivot_type
    )

    for i in df.loc[start_dt:end_dt].index:
        pos = df.index.get_loc(i)

        if isinstance(pos, slice):
            pos = pos.start

        dfi = df.iloc[pos - 160 : pos]

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
    sym_list: List[str],
    out_file: Path,
    loader: AbstractLoader,
    end_date: datetime,
    period: int,
):
    results: List[dict] = []
    futures = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sym in sym_list:

            future = executor.submit(
                scan,
                loader,
                args.date,
                args.period,
                args.pattern,
                sym,
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            results.extend(result)

    if len(results):
        logger.warning(
            f"Got {len(results)} patterns for {args.pattern.upper()}.\nRun `py backtest.py --plot {out_file.name}` to view results."
        )

        results.append(
            {
                "timeframe": loader.tf,
                "end_date": end_date.isoformat(),
                "period": loader.period,
            }
        )
        out_file.write_text(json.dumps(results, indent=2))
    else:
        logger.warning("No patterns found.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger = logging.getLogger(__name__)

    DIR = Path(__file__).parent

    if "-c" in sys.argv or "--config" in sys.argv:
        idx = sys.argv.index("-c" if "-c" in sys.argv else "--config")

        config_file = Path(sys.argv[idx + 1]).expanduser().resolve()
    else:
        config_file = DIR / "user.json"

    if not config_file.exists():
        logger.fatal("Missing user.json. Run init.py to generate user.json")
        exit()

    config = json.loads(config_file.read_bytes())

    sym_list = config["SYM_LIST"] if "SYM_LIST" in config else None

    if sym_list is not None and not (
        "-f" in sys.argv
        or "--file" in sys.argv
        or "--sym" in sys.argv
        or "--plot" in sys.argv
    ):
        sys.argv.extend(("-f", sym_list))

    args = parse_cli_args()

    # Import Loader module
    loader_name = config.get("LOADER", "EODFileLoader")

    loader_module = importlib.import_module(f"loaders.{loader_name}")

    if args.plot:
        meta = args.plot.pop()

        loader = getattr(loader_module, loader_name)(
            config,
            meta["timeframe"],
            end_date=datetime.fromisoformat(meta["end_date"]),
            period=meta["period"],
        )

        plotter = Plotter(args.plot, loader, mode="expand")
        plotter.plot(args.idx)
        exit()

    if args.date is None:
        exit("-d/--date argument is required")

    look_ahead_period = 120
    look_back_period = 160

    loader = getattr(loader_module, loader_name)(
        config,
        args.tf,
        end_date=args.date,
        period=look_back_period + args.period + look_ahead_period,
    )

    if args.file:
        sym_list = args.file.read_text().strip().split("\n")
    elif args.sym:
        sym_list = args.sym
    else:
        raise RuntimeError(
            "Error: -f or --file is required. Else define SYM_LIST in user.json"
        )

    output_file = DIR / f"bt_{args.pattern}_{loader.tf}.json"

    main(sym_list, output_file, loader, args.date, period)
