import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


class Plotter:
    idx = 0
    fig = None
    plt.ion()

    def __init__(self, data, source_folder, mode="default"):
        self.data = data
        self.source_folder = source_folder
        self.mode = mode
        self.len = len(data) - 1
        self.plot_args = {
            "type": "candle",
            "style": "tradingview",
            "figscale": 2,
            "returnfig": True,
            "scale_padding": {
                "left": 0.05,
                "right": 0.45,
                "top": 0.35,
                "bottom": 0.7,
            },
            "alines": {"linewidths": 0.8, "alpha": 0.7},
        }
        print("\tq: quit\n\tn: Next\n\tp: Previous" "")

    def plot(self, idx=None):
        if idx:
            self.idx = idx

        stmt = f"{self.idx} of {self.len}"
        print(stmt, flush=True, end="\r" * len(stmt))

        dct = self.data[self.idx]
        sym = dct["sym"]
        pattern = dct["pattern"]
        lines = dct["lines"]

        df = pd.read_csv(
            self.source_folder / f"{sym.lower()}.csv",
            index_col="Date",
            parse_dates=["Date"],
            na_filter=False,
        )

        if self.mode == "default":
            df = df.loc[dct["df_start"] : dct["df_end"]]
        else:
            start = df.index.get_loc(dct["start"])
            end = df.index.get_loc(dct["end"])

            if isinstance(start, slice):
                start = int(start.start)

            if isinstance(end, slice):
                end = int(end.start)

            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError("expected int")

            start = max(start - 120, 0)
            end = min(end + 120, df.shape[0])

            df = df.iloc[start:end]

        if pattern in ("Symetric", "Ascending", "Descending"):
            colors = "midnightblue"
        else:
            colors = ("green",) + ("midnightblue",) * (len(lines) - 1)

        self.plot_args["title"] = f"{sym} - {pattern}"

        self.plot_args["alines"].update({"alines": lines, "colors": colors})

        self.fig, axs = mpf.plot(df, **self.plot_args)

        self.main_ax = axs[0]

        axs[0].set_title(
            stmt, loc="left", color="black", fontdict={"fontweight": "bold"}
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        mpf.show(block=True)

    def on_key_press(self, event):
        key = event.key

        if key not in ("n", "p"):
            return

        if key == "n":
            if self.idx >= self.len:
                return self.main_ax.set_title(
                    "At Last Chart",
                    loc="right",
                    color="crimson",
                    fontdict={"fontweight": "bold"},
                )
            else:
                self.idx += 1

        if key == "p":
            if self.idx == 0:
                return self.main_ax.set_title(
                    "At First Chart",
                    loc="right",
                    color="crimson",
                    fontdict={"fontweight": "bold"},
                )
            else:
                self.idx -= 1

        plt.close(self.fig)
        self.plot()
