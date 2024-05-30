import pandas as pd
import numpy as np
import unittest
from context import utils


class Test_generate_trend_line(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = [10, 12, 14, 16, 18, 18, 16, 14, 12, 10]
        value_floats = np.concatenate(
            [np.linspace(0.1, 0.2, 5), np.linspace(0.2, 0.1, 5)]
        )

        self.series = pd.Series(values, index=dates)
        self.series_floats = pd.Series(value_floats, index=dates)

    def test_slope(self):
        d1 = pd.Timestamp("2022-01-01")
        d2 = pd.Timestamp("2022-01-05")
        d3 = pd.Timestamp("2022-01-06")
        d4 = pd.Timestamp("2022-01-10")

        assert isinstance(d1, pd.Timestamp)
        assert isinstance(d2, pd.Timestamp)
        assert isinstance(d3, pd.Timestamp)
        assert isinstance(d4, pd.Timestamp)

        t1 = utils.generate_trend_line(self.series, d1, d2)
        t2 = utils.generate_trend_line(self.series, d3, d4)

        self.assertGreater(t1.slope, 0)
        self.assertLess(t2.slope, 0)
        self.assertEqual(t1.line.start.y, self.series[d1])
        self.assertEqual(t2.line.start.y, self.series[d3])

    def test_slope_with_float_values(self):
        d1 = pd.Timestamp("2022-01-01")
        d2 = pd.Timestamp("2022-01-05")
        d3 = pd.Timestamp("2022-01-06")
        d4 = pd.Timestamp("2022-01-10")

        assert isinstance(d1, pd.Timestamp)
        assert isinstance(d2, pd.Timestamp)
        assert isinstance(d3, pd.Timestamp)
        assert isinstance(d4, pd.Timestamp)

        t1 = utils.generate_trend_line(self.series_floats, d1, d2)
        t2 = utils.generate_trend_line(self.series_floats, d3, d4)

        self.assertGreater(t1.slope, 0)
        self.assertLess(t2.slope, 0)
        self.assertEqual(t1.line.start.y, self.series_floats[d1])
        self.assertEqual(t2.line.start.y, self.series_floats[d3])


if __name__ == "__main__":
    unittest.main()
