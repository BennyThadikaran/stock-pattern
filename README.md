# Stock-Pattern

A Python CLI scanner to detect and plot common chart patterns

**Supports Python >= 3.8** (As of v1.1.1-alpha)

If you :heart: my work so far, please :star2: this repo.

![stock-pattern-cli](https://res.cloudinary.com/doyu4uovr/image/upload/s--PG_RPaP6--/c_scale,f_auto,w_800/v1706704565/stock-pattern/stock-pattern-cli_sbw1ny.png)

These were a series of pattern scanners I wrote for experimentation and fun in 2022. My inspiration was this Medium article [Algorithmically Detecting (and Trading) Technical Chart Patterns with Python](https://medium.com/automation-generation/algorithmically-detecting-and-trading-technical-chart-patterns-with-python-c577b3a396ed)

See Wiki to understand how the code works: [Pattern Algorithms](https://github.com/BennyThadikaran/stock-pattern/wiki/Pattern-Algorithms)

## Important Info

**All patterns are detected, prior to breakout.** at the last leg of the pattern. Add the stock to your watchlist and track them further.

This program does not provide any buy or sell signals. It only detects the pattern. It is upto you to decide, if a pattern is valid or tradeable.

## Installation

> **Stock-Pattern is being constantly updated with new features and bug fixes. Run `git pull` to get the latest updates.**

1. Clone or download the repo: `git clone https://github.com/BennyThadikaran/stock-pattern.git`
2. Install dependencies: `pip install -r requirements.txt`
3. `cd` into `/src` folder, run `py init.py`. It will generate a `user.json` file
4. Open `user.json` and edit the `DATA_PATH` with the folder path to your OHLC data. Files must be in CSV format in any timeframe.
   - If using [EOD2](https://github.com/BennyThadikaran/eod2) point it to `src/eod2_data/daily`
   - Optionally, add a `SYM_LIST` with a file path (CSV or TXT file) containing a list of symbols to scan. (One on each line). It will serve as a default watchlist to scan. See [Usage](#usage)
   - **Windows users: add an extra backslash for file paths to avoid JSON decode errors. `\\Documents\\python\\stock-pattern`**


```json
{
  "DATA_PATH": "~/Documents/python/eod2/src/eod2_data/daily",
  "POST_SCAN_PLOT": true,
  "SYM_LIST": "./nifty_500.csv"
}
```

## Usage

**Display Help:** `py init.py -h`

**Show Version:** `py init.py -v`

`py init.py (-f filepath | --sym SYM [SYM ...] | -v) options`

**Scan stocks using a watchlist file.** Pass a file containing stocks (one on each line).

```bash
# starts an interactive prompt
py init.py -f nifty_500.csv
```

**Scan stocks from a list of symbols.** Pass a list of symbols space separated.

```bash
py init.py --sym tcs astral datapattns
```

**NOTE**: If `SYM_LIST` is specified in `user.json`, `-f` or `--sym` is not required unless you wish to override the default watchlist.

**Identify chart patterns on a specific date.**:

```bash
# iso date format YYYY-MM-DD
py init.py -f nifty_500.csv -d 2023-01-01
```

**Save results as images to folder**

```bash
# default folder is src/images
py init.py -f nifty_500.csv --save
```

```bash
# add a custom folder
py init.py -f nifty_500.csv --save ~/Desktop/pattern/
```
**To skip the interactive prompt, specify a pattern string:**

```bash
# Bullish VCP
py init.py -f nifty_500.csv -p vcpu
```

Pattern string can be one of:
| Pattern String | Description                    |
| -------------- | ------------------------------ |
| all            | All patterns                   |
| bull           | All Bullish patterns           |
| bear           | All Bearish patterns           |
| vcpu           | VCP **U**p (Bullish)           |
| vcpd           | VCP **D**own (Bearish)         |
| dbot           | Double Bottom                  |
| dtop           | Double Top                     |
| hnsu           | Head & Shoulder **U**p (Bullish) |
| hnsd           | Head & Shoulder **D**own (Bearish) |
| trng           | Triangles (Symmetric, Ascending, Descending) |

There are other CLI options available for those wanting to tinker.

## Chart Plot options

When patterns are detected the results are saved as json files. Filename is same as pattern string, for Ex. vcpu.json

**To plot results of scan, use `--plot`**. 
```bash
py init.py --plot vcpu.json
```

**To jump to a particular count in the result, use `--idx`**
```bash
py init.py --plot vcpu.json --idx 15
```

When scanning using a watchlist file, all patterns are tracked in json files stored in `src/state` folder. Files format is <watchlist_name>_<pattern_name>.json for ex. nifty_500_vcpu.json.

**Only newly detected patterns are notified and plotted on charts.**

**To see all currently active patterns in market, use the `--plot` passing the json file from `state` folder**
```bash
py init.py --plot state/nifty_500_vcpu.json
```

## Chart keyboard navigation

| Key             | Description    |
| --------------- | -------------- |
| n               | Next Chart     |
| p               | Previous Chart |
| NUM_KEY + j     | Type the number followed by `j` to jump to index. Press `ESC` to clear  |


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

- Make package available via pip [Delayed].

# Disclaimer

This program identifies common chart patterns and does it reasonably well (IMHO).

Stock-Pattern is not aimed at replacing a trader's trained eye. It is merely an assistance and learning tool.

CLI options available, for you to backtest the patterns. If you intend to trade these patterns, study them well and trade at your own risk.
