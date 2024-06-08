# Stock-Pattern

A Python CLI scanner to detect and plot common chart patterns

**Supports Python >= 3.8** (As of v1.1.1-alpha)

If you :heart: my work so far, please :star2: this repo.

![stock-pattern-cli](https://res.cloudinary.com/doyu4uovr/image/upload/s--CieAES1A--/c_scale,f_auto,w_800/v1716034129/stock-pattern/stock-pattern-cli-v3_z92axi.png)

These were a series of pattern scanners I wrote for experimentation and fun in 2022. My inspiration was this Medium article [Algorithmically Detecting (and Trading) Technical Chart Patterns with Python](https://medium.com/automation-generation/algorithmically-detecting-and-trading-technical-chart-patterns-with-python-c577b3a396ed)

Understand how the code works: [wiki Pattern Algorithms](https://github.com/BennyThadikaran/stock-pattern/wiki/Pattern-Algorithms)

## Important Info

**All patterns are detected, prior to breakout.** at the last leg of the pattern. Add the stock to your watchlist and track them further.

This program does not provide any buy or sell signals. It only detects the pattern. It is upto you to decide, if a pattern is valid or tradeable.

## Installation

> **Stock-Pattern is being constantly updated with new features and bug fixes. Run `git pull` to get the latest updates.**

1. Clone or download the repo:
   ```
   git clone https://github.com/BennyThadikaran/stock-pattern.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run `init.py`. It will generate a `user.json` file
4. Open `user.json` and edit the `DATA_PATH` with the folder path to your OHLC data.
   - Files must be in CSV format in any timeframe and filenames must be lowercase. For ex. infy.csv
   - If using [EOD2](https://github.com/BennyThadikaran/eod2) point it to `src/eod2_data/daily`
6. Optionally, add a `SYM_LIST` with a file path (CSV or TXT file) containing a list of symbols to scan (One on each line).
   
```
# Assuming your files are named infy.csv etc.
infy
tcs
marksans
godrejind
```

7. See [wiki Configuration](https://github.com/BennyThadikaran/stock-pattern/wiki/Usage#configuration) for more options

```json
{
  "DATA_PATH": "~/Documents/python/eod2/src/eod2_data/daily",
  "SYM_LIST": "~/Desktop/nifty_500.csv",
  "POST_SCAN_PLOT": true
}
```

**Note for Window users**

- Add an extra backslash `\\` for directory separators in JSON. This will avoid JSON decode errors.
- Tilde `~` works on both Windows and Linux and expands to user's folder

```json
{
  "DATA_PATH": "~\\Documents\\python\\eod2\\src\\eod2_data\\daily",
  "SYM_LIST": "~\\Documents\\python\\stock-pattern\\src\\data.csv",
  "POST_SCAN_PLOT": true
}
```

## Usage

**Moving to version 3.0.0?** See [Quick usage and config](https://github.com/BennyThadikaran/stock-pattern/wiki/Quick-config-and-usage-summary-%E2%80%90-version-3.0.0)

See [wiki Usage instructions](https://github.com/BennyThadikaran/stock-pattern/wiki/Usage)

For backtest.py usage, see [wiki backtest](https://github.com/BennyThadikaran/stock-pattern/wiki/backtest-usage)

## Notes on structure of OHLC data
- The following columns are expected: **Date, Open, High, Low, Close, Volume.** Additonal columns may exist.
- Dates must be in Descending order (Oldest to Newest).

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

# Disclaimer

This program identifies common chart patterns and does it reasonably well (IMHO).

Stock-Pattern is not aimed at replacing a trader's trained eye. It is merely an assistance and learning tool.

CLI options available, for you to backtest the patterns. If you intend to trade these patterns, study them well and trade at your own risk.
