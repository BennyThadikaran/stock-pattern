import unittest

import pandas as pd
from context import utils


class TestGetPrevIndex(unittest.TestCase):

    def test_unique_index_returns_previous_position(self):
        index = pd.DatetimeIndex(
            ["2023-01-01", "2023-01-02", "2023-01-03"]
        )
        self.assertEqual(
            utils.get_prev_index(index, pd.Timestamp("2023-01-02")), 0
        )

    def test_duplicate_index_returns_position_before_group(self):
        # Duplicated timestamps resolve to a slice; the previous index
        # is the position right before the duplicated group starts.
        index = pd.DatetimeIndex(
            ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"]
        )
        self.assertEqual(
            utils.get_prev_index(index, pd.Timestamp("2023-01-02")), 0
        )

    def test_duplicate_index_does_not_match_next_index(self):
        # Regression test: get_prev_index must not return the same
        # position as get_next_index for duplicated timestamps.
        index = pd.DatetimeIndex(
            ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"]
        )
        idx = pd.Timestamp("2023-01-02")
        self.assertNotEqual(
            utils.get_prev_index(index, idx),
            utils.get_next_index(index, idx),
        )


if __name__ == "__main__":
    unittest.main()
