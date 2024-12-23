import functools
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplfinance as mpf
import pandas as pd

from loaders.AbstractLoader import AbstractLoader

try:
    pairwise_fn = itertools.pairwise
except AttributeError:
    # Support for Python 3.8 and 3.9
    # https://docs.python.org/3.8/library/itertools.html#itertools-recipes
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    pairwise_fn = pairwise

try:
    batched_fn = itertools.batched
except AttributeError:
    # Support for batched function for Python 3.12 and lower
    # Source https://docs.python.org/3/library/itertools.html#itertools.batched
    def batched(iterable, n, *, strict=False):
        """
        batched('ABCDEFG', 3) → ABC DEF G
        """
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch

    batched_fn = batched


class Plotter:
    idx = 0
    idx_str = ""
    default_backend = plt.get_backend()

    def __init__(
        self,
        data,
        loader: AbstractLoader,
        save_folder: Optional[Path] = None,
        mode: Literal["default", "expand"] = "default",
        config: dict = {},
    ):
        self.save_folder = save_folder
        self.mode = mode
        self.loader = loader
        self.timeframe = loader.tf
        self.config = config

        self.line_color = self.config.get("LINE_COLOR", "midnightblue")

        self.plot_args: Dict[str, Any] = dict(
            type=self.config.get("TYPE", "candle"),
            style=self.config.get("STYLE", "tradingview"),
            xrotation=0,
            scale_padding=dict(left=0.05, right=0.6, top=0.35, bottom=0.5),
            alines=dict(
                linewidths=self.config.get("LINE_WIDTH", 0.8),
                alpha=self.config.get("ALPHA", 0.7),
            ),
            returnfig=True,
            figscale=1,
        )

        self.data = data

        if save_folder:
            self.plot_args["figscale"] = 1.5

            # Save images with non interactive backend and
            # switch off interactive mode
            # Switch to a Non GUI backend to work with threads
            plt.switch_backend("AGG")
            plt.ioff()
        else:
            plt.ion()
            plt.switch_backend(self.default_backend)
            self.len = len(data) - 1

            print("\nChart Controls\n\tq: quit\n\tn: Next\n\tp: Previous")

    def plot(self, idx=None) -> None:
        if self.data is None:
            raise TypeError("Missing dict data")

        if idx:
            self.idx = idx

        dct = self.data[self.idx]
        sym = dct["sym"].upper()

        if "alt_name" in dct:
            pattern = dct["alt_name"]
        else:
            pattern = dct["pattern"]

        df = self.loader.get(sym)

        if df is None:
            raise ValueError(f"Unable to load data for {sym}")

        if df.index.has_duplicates:
            df = df.loc[~df.index.duplicated()]

        if not df.index.is_monotonic_increasing:
            df = df.sort_index(ascending=True)

        self.df = df

        if self.mode == "expand":
            start = df.index.get_loc(dct["df_start"])
            end = df.index.get_loc(dct["df_end"])

            if isinstance(start, slice):
                start = start.start

            if isinstance(end, slice):
                end = end.start

            assert isinstance(start, int)
            assert isinstance(end, int)

            start = max(start - 120, 0)
            end = min(end + 120, len(df))

            df = df.iloc[start:end]

        lines = self._build_lines(dct, pattern)

        self.plot_args["title"] = (
            f"{sym} - {pattern} - {self.timeframe.capitalize()}"
        )

        self.plot_args["alines"].update(
            dict(alines=lines, colors=(self.line_color,) * len(lines))
        )

        self.fig, axs = mpf.plot(df, **self.plot_args)

        self.main_ax = axs[0]

        y_min, y_max = plt.ylim()

        # Calculate a threshold for line labels.
        # If distance between lines if less than threshold, they can result in
        # overlapping labels, which are too difficult to read.
        self.threshold = y_max - y_min

        self._annotate_fn = functools.partial(
            axs[0].annotate,
            textcoords="offset pixels",
            horizontalalignment="center",
            fontweight="bold",
            color=self.config.get("LABEL_COLOR", "midnightblue"),
        )

        self._set_date_formatter(axs)

        self._annotate_points(dct["points"], dct["end"])

        if pattern not in (
            "Symmetric",
            "Ascending",
            "Descending",
            "DNTL",
            "UPTL",
        ):
            self._annotate_extra_points(dct["extra_points"], dct["end"])

        if self.save_folder:

            return plt.savefig(
                self.save_folder / f"{sym}_{pattern}_{self.timeframe}.png"
            )

        stmt = f"{self.idx} of {self.len}"

        print(stmt, flush=True, end="\r" * len(stmt))

        axs[0].set_title(
            stmt,
            loc="left",
            color=self.config.get("NAV_TITLE_COLOR", "black"),
            fontdict={"fontweight": "bold"},
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        window_manager = plt.get_current_fig_manager()

        if window_manager:
            if "win" in sys.platform:
                try:
                    # Only works with tkAgg backend
                    window_manager.window.state("zoomed")
                except AttributeError:
                    window_manager.full_screen_toggle()
            else:
                window_manager.full_screen_toggle()

        mpf.show(block=True)

    def _on_key_press(self, event):
        key = event.key

        if key.isdigit():
            self.idx_str += key

            return self._alert(f"{self.idx_str}j")

        if key not in ("n", "p", "j", "escape"):
            return

        if key == "escape":
            self.idx_str = ""

            return self._alert()

        if key == "j":
            if self.idx_str == "":
                return

            idx = int(self.idx_str)

            if idx > self.len:
                self.idx_str = ""
                return self._alert()

            self.idx = idx

        if key == "n":
            if self.idx >= self.len:
                return self._alert("At Last Chart")

            self.idx += 1

        if key == "p":
            if self.idx == 0:
                return self._alert("At First Chart")

            self.idx -= 1

        self.idx_str = ""
        plt.close("all")
        self.plot()

    def _alert(self, string=""):
        return self.main_ax.set_title(
            string,
            loc="right",
            color=self.config.get("NAV_STATUS_COLOR", "crimson"),
            fontdict={"fontweight": "bold"},
        )

    def _build_lines(self, dct: dict, pattern: str):
        if pattern in ("Symmetric", "Ascending", "Descending", "DNTL", "UPTL"):
            points = dct["extra_points"]

            return tuple(line for line in batched_fn(points.values(), 2))

        points = dct["points"]
        return tuple(line for line in pairwise_fn(points.values()))

    def format_coords(self, x, y):
        s = " " * 5

        if self.df is None:
            return

        if not x or round(x) >= len(self.df):
            return ""

        dt = self.df.index[round(x)]

        dt_str = f"{dt:%d %b'%y %H:%M}".upper()

        open, high, low, close, vol = self.df.loc[
            dt, ["Open", "High", "Low", "Close", "Volume"]
        ]

        _str = f"Price: {y:.2f} | {dt_str}{s}O: {open}{s}H: {high}{s}L: {low}{s}C: {close}{s}V: {vol:,.0f}"

        return _str

    def _get_tick_locs(self, tick_mdates, dtix: pd.DatetimeIndex):
        """Return the tick locs to be passed to Locator instance."""

        ticks = []

        # Convert the matplotlib dates to python datetime and iterate
        for dt in mdates.num2date(tick_mdates):
            # remove the timezone info to match the DataFrame index
            dt = dt.replace(tzinfo=dtix.tzinfo)

            # Get the index position if available
            # else get the next available index position
            idx = (
                dtix.get_loc(dt)
                if dt in dtix
                else dtix.searchsorted(dt, side="right")
            )
            # store the tick positions to be displayed on chart
            ticks.append(idx)

        return ticks

    def _set_date_formatter(self, axs):
        assert isinstance(self.df.index, pd.DatetimeIndex)

        # Locator sets the major tick locations on xaxis
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)

        # Formatter set the tick labels for the xaxis
        concise_formatter = mdates.ConciseDateFormatter(locator=locator)

        # Extract the tick values from locator.
        # These are matplotlib dates not python datetime
        tick_mdates = locator.tick_values(self.df.index[0], self.df.index[-1])

        # Extract the ticks labels from ConciseDateFormatter
        labels = concise_formatter.format_ticks(tick_mdates)

        ticks = self._get_tick_locs(tick_mdates, self.df.index)

        # Initialise FixedFormatter and FixedLocator
        # passing the tick labels and tick positions
        fixed_formatter = ticker.FixedFormatter(labels)
        fixed_locator = ticker.FixedLocator(ticks)

        fixed_formatter.set_offset_string(concise_formatter.get_offset())

        for ax in axs:
            ax.xaxis.set_major_locator(fixed_locator)
            ax.xaxis.set_major_formatter(fixed_formatter)
            ax.format_coord = self.format_coords

    def _annotate_extra_points(self, points, last):
        hline_levels = []
        colors = []
        xmax = self.df.index.get_loc(last)
        xmin = prev_line_price = prev_diff = None

        for label, point in points.items():
            x, y = point

            if label == "direction":
                self.main_ax.hlines(
                    y,
                    self.df.index.get_loc(x),
                    xmax,
                    colors=self.config.get("BIAS_LINE_COLOR", "green"),
                )
            else:
                if xmin is None:
                    xmin = self.df.index.get_loc(x)

                diff = (
                    self.threshold
                    if prev_line_price is None
                    else abs(prev_line_price - y)
                )

                prev_line_price = y

                hline_levels.append(y)

                colors.append(self.line_color)

                x_offset = 0
                y_offset = -10

                # If labels are too close, adjust the label positions
                if diff <= self.threshold:
                    if self.mode == "expand":
                        y_offset = -50
                        x_offset = -40

                    if prev_diff and prev_diff <= self.threshold:
                        y_offset = -30

                prev_diff = diff

                self._annotate_fn(
                    text=f"{label} - {y:.2f}",
                    xy=(xmin, y),
                    xytext=(x_offset, y_offset),
                    rotation=15,
                )

        self.main_ax.hlines(hline_levels, xmin, xmax, colors=colors)

    def _annotate_points(self, points, last):
        for label, point in points.items():
            x, y = point

            if x == last:
                text_loc = (15, 0)
            else:
                text_loc = (0, 15 if y == self.df.at[x, "High"] else -15)

            self._annotate_fn(
                text=label,
                xy=(self.df.index.get_loc(x), y),
                xytext=text_loc,
            )
