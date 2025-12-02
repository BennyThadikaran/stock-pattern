# Stock-Pattern

A Python CLI scanner to detect and plot common chart patterns. Includes Harmonic chart patterns

**Supports Python >= 3.8**

If you :heart: my work so far, please :star2: this repo.

---

✨ [Precise-Patterns](https://github.com/BennyThadikaran/precise-patterns) has just been released. 🚀 It is currently in initial development and reimagines the pattern detection process from the ground up, with many new improvements. 🔍💡

See the [initial announcement and further discussions here](https://github.com/BennyThadikaran/stock-pattern/discussions/128). 📣

![stock-pattern-cli-1](https://res.cloudinary.com/doyu4uovr/image/upload/s--HjrsEmtc--/c_scale,f_auto,w_500/v1732622055/stock-pattern/stock-pattern-ui-1_agjjxp.png)

![stock-pattern-cli-2](https://res.cloudinary.com/doyu4uovr/image/upload/s--qoHrLTS---/c_scale,f_auto,w_500/v1732622054/stock-pattern/stock-pattern-ui-2_jwpvv1.png)

These were a series of pattern scanners I wrote for experimentation and fun in 2022. My inspiration was this Medium article [Algorithmically Detecting (and Trading) Technical Chart Patterns with Python](https://medium.com/automation-generation/algorithmically-detecting-and-trading-technical-chart-patterns-with-python-c577b3a396ed)

Understand how the code works: [wiki Pattern Algorithms](https://github.com/BennyThadikaran/stock-pattern/wiki/Pattern-Algorithms)

## Important Info

**All patterns are detected, prior to breakout.** at the last leg of the pattern. Add the stock to your watchlist and track them further.

This program does not provide any buy or sell signals. It only detects the pattern. It is upto you to decide, if a pattern is valid or tradeable.

## Installation

1. Clone or download the repo:

   ```
   git clone https://github.com/BennyThadikaran/stock-pattern.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Setup configuration file with setup-config.py and answer the questions to generate a user.json

   ```bash
   py setup-config.py
   ```

   - Use direction keys to select a choice and press `ENTER`.
   - `TAB` key can be used to autocomplete filepaths. Use forward slashes `/` regardless of your OS to specify filepaths.
   - By default, filepath is mapped to your home folder `/home/YourUsername` on mac/linux and `C:/Users/YourUsername` on Windows.
   - You can use `../../` to go to C drive.

   **Alternatively, I've written step by step instructions to [create user.json manually](https://github.com/BennyThadikaran/stock-pattern/wiki/Writing-JSON-configuration-manually)**

## Usage

See [Quick usage summary](https://github.com/BennyThadikaran/stock-pattern/wiki/Quick-usage-summary)

See [Usage instructions](https://github.com/BennyThadikaran/stock-pattern/wiki/Usage)

See [Configuration](https://github.com/BennyThadikaran/stock-pattern/wiki/Writing-JSON-configuration-manually)

For backtest.py usage, see [wiki backtest](https://github.com/BennyThadikaran/stock-pattern/wiki/backtest-usage)

## Notes on structure of OHLC data

- The following columns are expected: **Open, High, Low, Close, Volume.** Additonal columns may exist.

## Screenshots

**HUDCO - Bull VCP 25th Apr 2024**

![HUDCO - Bull VCP](https://res.cloudinary.com/doyu4uovr/image/upload/s--ACa-Ml5n--/c_scale,f_auto,w_800/v1732622055/stock-pattern/hudco-vcpu-daily_sccvoh.png)

**RRKABEL - Bull BAT Harmonic pattern 13th Mar 2024**

![RRKABEL - Bull BAT](https://res.cloudinary.com/doyu4uovr/image/upload/s--ge7esWPA--/c_scale,f_auto,w_800/v1732622055/stock-pattern/rrkabel-bull-bat-daily_yewyxv.png)

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
