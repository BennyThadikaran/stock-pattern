# Stock-Pattern
A Python scanner to detect and plot common chart patterns

If you :heart: my work so far, please :star2: this repo.

![stock-pattern-cli](https://res.cloudinary.com/doyu4uovr/image/upload/s--z3JHzPt---/f_auto/v1701779360/stock-pattern/stock-pattern-cli_nkrdtx.png)

These were a series of pattern scanners I wrote for experimentation and fun in 2022. My inspiration was this Medium article [Algorithmically Detecting (and Trading) Technical Chart Patterns with Python](https://medium.com/automation-generation/algorithmically-detecting-and-trading-technical-chart-patterns-with-python-c577b3a396ed)

See Wiki to understand how the code works: [Pattern Algorithms](https://github.com/BennyThadikaran/stock-pattern/wiki/Pattern-Algorithms
)

Stock-Pattern is not aimed at replacing a trader's trained eye. It is merely an assistance and learning tool. There are still some tweaks required and changes will be rolled out over the coming weeks.

## Installation

1. Clone or download the repo: `git clone https://github.com/BennyThadikaran/stock-pattern.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `src/init.py`:
     1. Replace `data_path` with the full path to your data source. If using [EOD2](https://github.com/BennyThadikaran/eod2) point it to `src/eod2_data/daily`
     2. Replace the `file` with the full path to a stock list file. (Each stock on a separate line)
4. `CD` into `/src` folder, run the program: `py init.py`, and follow the instructions.

Every time a pattern is detected, the chart is plotted with lines marking the pattern.

Press `q` to quit the chart and resume scanning.

**Triangle**

![Stock chart with Triangle pattern](https://res.cloudinary.com/doyu4uovr/image/upload/s--LNJdebaf--/f_auto/v1701779361/stock-pattern/stock-pattern-pennant_sjby8n.png)

**Double Tops**

![Stock chart with Double top pattern](https://res.cloudinary.com/doyu4uovr/image/upload/s--BK9BeUdJ--/f_auto/v1701779360/stock-pattern/stock-pattern-double-tops_od8lro.png)

# TODO
- Improve the user interface and ability to pass watchlist files as arguments.
- Add key binding to exit the program from the chart itself
- Allow the user to skip plotting and output the symbol names.
- Reduce code duplication


