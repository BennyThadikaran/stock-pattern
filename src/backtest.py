import argparse
import concurrent.futures
import importlib
import json
import logging
from datetime import datetime, timedelta
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
py backtest.py -p vcpu
```

# Custom Watchlist
Provide a custom watchlist of symbols as a CSV or TXT file

```python 
py backtest.py -p hnsd --file symlist.csv
```

# Set Scan parameters
Default scan end date is 120 days prior to current date. 
Scan period is 30 days prior to end date.

```python 
py backtest.py -p trng --date 2023-12-01 --period 60
```
"""


def parse_cli_args():
    key_list = ("vcpu", "vcpd", "dbot", "dtop", "hnsu", "hnsd", "trng")

    parser = argparse.ArgumentParser(description="Run backdated pattern scan")

    parser.add_argument(
        "-f",
        "--file",
        type=lambda x: Path(x).expanduser().resolve(),
        default=None,
        help="Filepath with symbol list. One on each line",
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
        "--plot",
        type=lambda x: json.loads(Path(x).read_bytes()),
        default=None,
        help="Plot results from json file",
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
        help="Ending date of scan.",
    )

    parser.add_argument(
        "--period",
        type=int,
        default=30,
        help="Scan Period, prior to ending date. Default 30",
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
    }

    df = loader.get(sym)

    if df is None or df.empty or end is not None and end < df.index[0]:
        return results

    assert isinstance(df.index, pd.DatetimeIndex)

    if end is None:
        dt = df.index[-120]

        assert isinstance(dt, pd.Timestamp)

        end = dt.to_pydatetime()

    pos = df.index.get_loc(df.index.asof(end))

    if isinstance(pos, slice):
        pos = pos.start

    start = df.index[pos - period]

    if start < df.index[0]:
        return results

    assert isinstance(start, pd.Timestamp)

    dt_index = df.index.date

    has_time_component = utils.has_time_component(df.index)

    if has_time_component:
        start_dt = df[start.date() == dt_index].index.max()
        end_dt = df[end.date() == dt_index].index.max()
    else:
        start_dt = start
        end_dt = end

    start_pos = df.index.get_loc(df.index.asof(start_dt))
    end_pos = df.index.get_loc(df.index.asof(end_dt))

    if isinstance(start_pos, slice):
        start_pos = start_pos.start

    if isinstance(end_pos, slice):
        end_pos = end_pos.start

    pivots_all = utils.get_max_min(df.iloc[start_pos - 160 : end_pos])

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
    end_date,
    period,
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
                "end_date": (end_date + timedelta(120)).isoformat(),
                "period": period,
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

    config_file = DIR / "user.json"

    if not config_file.exists():
        logger.fatal("Missing user.json. Run init.py to generate user.json")
        exit()

    config = json.loads(config_file.read_bytes())

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

    period = 120 + args.period + 120

    loader = getattr(loader_module, loader_name)(
        config,
        args.tf,
        end_date=args.date,
        period=period,
    )

    if args.file:
        file = args.file
    elif "SYM_LIST" in config:
        file = Path(config["SYM_LIST"]).expanduser().resolve()
    else:
        raise RuntimeError(
            "Error: -f or --file is required. Else define SYM_LIST in user.json"
        )

    sym_list = file.read_text().strip().split("\n")
    output_file = DIR / f"bt_{args.pattern}_{loader.tf}.json"

    main(sym_list, output_file, loader, args.date, period)
