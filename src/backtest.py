import json
import utils
import argparse
import logging
import concurrent.futures
from typing import List
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
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
    now = datetime.combine(datetime.now(), datetime.min.time())

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
        type=lambda x: datetime.fromisoformat(x),
        default=now - timedelta(120),
        help="Ending date of scan. Defaults to 120 days back",
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
    file, end: datetime, period: int, fn: str, sym: str, count: int
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

    df = pd.read_csv(
        file,
        index_col="Date",
        parse_dates=["Date"],
    )

    if df.empty or end < df.index[0] or end > df.index[-1]:
        return results

    pos = df.index.get_loc(end)

    if isinstance(pos, slice):
        pos = pos.start

    start = df.index[pos - period]

    if start < df.index[0]:
        return results

    if count == 0:
        utils.logger.info(
            f"Scan period: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("SCAN: expected DatetimeIndex")

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


def main(sym_list: List[str], out_file: Path, source_folder: Path):
    results: List[dict] = []
    futures = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for count, sym in enumerate(sym_list):
            file = source_folder / f"{sym.lower()}.csv"

            if not file.exists():
                utils.logger.warning(f"WARN: File not found {sym}: {file}")
                continue

            future = executor.submit(
                scan, file, args.date, args.period, args.pattern, sym, count
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            results.extend(result)

    if len(results):
        utils.logger.warning(
            f"Got {len(results)} patterns for {args.pattern.upper()}.\nRun `py backtest.py --plot {out_file.name}` to view results."
        )

        out_file.write_text(json.dumps(results, indent=2))
    else:
        utils.logger.warning("No patterns found.")


if __name__ == "__main__":
    utils.logger.setLevel(logging.INFO)

    DIR = Path(__file__).parent

    config_file = DIR / "user.json"

    if not config_file.exists():
        utils.logger.fatal(
            "Missing user.json. Run init.py to generate user.json"
        )
        exit()

    config = json.loads(config_file.read_bytes())

    args = parse_cli_args()

    source_folder = Path(config["DATA_PATH"]).expanduser().resolve()

    if args.plot:
        plotter = Plotter(args.plot, source_folder, mode="expand")
        plotter.plot(args.idx)
        exit()

    if args.file:
        file = args.file
    elif "SYM_LIST" in config:
        file = Path(config["SYM_LIST"]).expanduser().resolve()
    else:
        raise RuntimeError(
            "Error: -f or --file is required. Else define SYM_LIST in user.json"
        )

    sym_list = file.read_text().strip().split("\n")
    output_file = DIR / f"bt_{args.pattern}.json"

    main(sym_list, output_file, source_folder)
