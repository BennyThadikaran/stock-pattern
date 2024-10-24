import unittest

import pandas as pd
from context import utils


class TestGetMaxMin(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = dict(
            High=[5, 6, 7, 6, 9, 8, 10, 7, 6, 5],
            Low=[4, 5, 5, 3, 6, 5, 5, 2, 4, 4],
            Volume=[10, 20, 30, 20, 40, 30, 50, 20, 10, 5],
        )

        self.df = pd.DataFrame(data)
        self.df.index = pd.date_range(
            start="2023-01-01", periods=len(data["High"])
        )

    def test_get_max_min_both(self):
        result = utils.get_max_min(self.df, barsLeft=1, barsRight=1)

        self.assertEqual(result.at[result.index[0], "P"], 7)
        self.assertEqual(result.at[result.index[1], "P"], 3)
        self.assertEqual(result.at[result.index[2], "P"], 9)

    def test_get_max_min_high(self):
        result = utils.get_max_min(
            self.df,
            barsLeft=1,
            barsRight=1,
            pivot_type="high",
        )

        self.assertEqual(result.at[result.index[0], "P"], 7)
        self.assertEqual(result.at[result.index[1], "P"], 9)
        self.assertEqual(result.at[result.index[2], "P"], 10)

    def test_get_max_min_low(self):
        result = utils.get_max_min(
            self.df,
            barsLeft=1,
            barsRight=1,
            pivot_type="low",
        )

        self.assertEqual(result.at[result.index[0], "P"], 3)
        self.assertEqual(result.at[result.index[1], "P"], 2)


if __name__ == "__main__":
    unittest.main()
