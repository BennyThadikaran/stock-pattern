# Stock-Pattern

A Python CLI scanner to detect and plot common chart patterns

If you :heart: my work so far, please :star2: this repo.

![stock-pattern-cli](https://res.cloudinary.com/doyu4uovr/image/upload/s--9HW6Yk6D--/c_scale,f_auto,w_700/v1702918851/stock-pattern/stock-pattern-cli_bzd7ze.png)

These were a series of pattern scanners I wrote for experimentation and fun in 2022. My inspiration was this Medium article [Algorithmically Detecting (and Trading) Technical Chart Patterns with Python](https://medium.com/automation-generation/algorithmically-detecting-and-trading-technical-chart-patterns-with-python-c577b3a396ed)

See Wiki to understand how the code works: [Pattern Algorithms](https://github.com/BennyThadikaran/stock-pattern/wiki/Pattern-Algorithms)

## Important Info

**Update 27th Dec 2023:** Many UI improvements and other changes. See [all changes](https://github.com/BennyThadikaran/stock-pattern/commit/d7081c63c8ecfec9afaec15fcf237f43f4e28a74)

**Update 18th Dec 2023:** All patterns are tested and working perfectly, as of **version 0.2.2-alpha**.

**All patterns are detected, prior to breakout.** Patterns are detected at the last leg of the pattern. Allowing you to add the stock to your watchlist and track them further.

This program does not provide any buy or sell signals. It only detects the pattern. It is upto you to decide, if a pattern is valid or tradeable.

## Installation

1. Clone or download the repo: `git clone https://github.com/BennyThadikaran/stock-pattern.git`
2. Install dependencies: `pip install -r requirements.txt`
3. `cd` into `/src` folder, run `py init.py`. It will generate a `user.json` file
4. Open `user.json` and edit the `DATA_PATH` with the absolute path to your data source
   - If using [EOD2](https://github.com/BennyThadikaran/eod2) point it to `src/eod2_data/daily`
   - Optionally, add a `SYM_LIST` with an absolute file path (CSV or TXT file) containing a list of stock symbols to scan. (One on each line)
   - This will serve as a default watchlist to scan. `SYM_LIST` can be overriden with `-f/--file` option. See [Usage](#usage)

```json
{
  "_comment": "DATA_PATH: OHLC folder path, SYM_LIST: Optional stocklist filepath",
  "DATA_PATH": "~/Documents/python/eod2/src/eod2_data/daily",
  "SYM_LIST": "~/Documents/python/stock-pattern/src/data.csv"
}
```

**Stock-Pattern is being constantly updated with new features and bug fixes. Run `git pull` to get the latest updates.**

Every time a pattern is detected, the chart is plotted with lines marking the pattern.

Press `q` to quit the chart and resume scanning.

To quit the program, press `CTRL + C` in the terminal or `SHIFT + q` in the chart.

## Usage

**Display Help:** `py init.py -h`

**Show Version:** `py init.py -v`

`py init.py (-f filepath | --sym SYM [SYM ...] | -v) options`

**Scan stocks from a file.** Pass a file containing stocks (one on each line).

```bash
# starts an interactive prompt
py init.py -f data.csv
```

**Scan stocks from a list of symbols.** Pass a list of symbols space separated.

```bash
py init.py --sym tcs astral datapattns
```

**NOTE**: If `SYM_LIST` is specified in `user.json`, `-f` or `--sym` is not required unless you wish to override the default watchlist.

**Identify chart patterns on a specific date.**:

```bash
# iso date format YYYY-MM-DD
py init.py -f data.csv -d 2023-01-01
```

**To skip the interactive prompt, specify the pattern number:**

```bash
# Bullish VCP
py init.py -f data.csv -p 1
```

**To run silent, without opening chart plots:**

```bash
# Only symbol names are printed
py init.py -f data.csv -p 1 -s
```

There are other CLI options available for those wanting to tinker.

## Screenshots

**HUDCO - Double Bottom 16th June 2022**

![HUDCO - Double Bottom](https://res.cloudinary.com/doyu4uovr/image/upload/s--nQ10RsSG--/c_scale,f_auto,w_700/v1702918851/stock-pattern/hudco-double-bottom_dtcvwj.png)

**GRAPHITE - Bear VCP 18th Oct 2022**

![GRAPHITE - Bear VCP](https://res.cloudinary.com/doyu4uovr/image/upload/s--LLToQmNv--/c_scale,f_auto,w_700/v1702918851/stock-pattern/graphite-bear-vcp_yi7cmv.png)

**MGL - Reverse Head & Shoulder 16th June 2022**

![MGL - Reverse Head & Shoulder](https://res.cloudinary.com/doyu4uovr/image/upload/s--RsyvqU2E--/c_scale,f_auto,w_700/v1702918852/stock-pattern/mgl-reverse-hns_pwkmel.png)

**GSFC - Triangle 13th March 2023**

![GSFC - Triangle pattern](https://res.cloudinary.com/doyu4uovr/image/upload/s--oiNVrg8u--/c_scale,f_auto,w_700/v1702918851/stock-pattern/gsfc-triangle_j32yi3.png)

**RECLTD - Double Tops 13th Oct 2021**

![RECLTD Double Top pattern](https://res.cloudinary.com/doyu4uovr/image/upload/s--lFTiRydt--/c_scale,f_auto,w_700/v1702918852/stock-pattern/recltd-double-top_accoad.png)

## TODO

As of Version 1.1.0-alpha

- ~~Support all timeframes.~~

As of Version 1.0.0-alpha

- ~~Add key binding to exit the program from the chart itself~~
- ~~Add documentation for all pattern algorithms~~

As of Version 0.2.2-alpha

- ~~Allow the user to skip plotting and output the symbol names.~~
- ~~Improve the user interface and ability to pass watchlist files as arguments.~~

- Make package available via pip.
- Reduce code duplication.
- Ability to scan all patterns in a single run.

# Disclaimer

This program identifies common chart patterns and does it reasonably well (IMHO).

Stock-Pattern is not aimed at replacing a trader's trained eye. It is merely an assistance and learning tool.

CLI options available, for you to backtest the patterns. If you intend to trade these patterns, study them well and trade at your own risk.
