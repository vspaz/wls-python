#!/usr/bin/env python

import unittest
from array import array

from wls import wls


class TestWLS(unittest.TestCase):
    def _assert_regression_model(self, y, x, w, expected):
        expected_intercept, expected_slope = expected
        got_intercept, got_slope = wls.fit_linear_regression(y, x, w)
        self.assertAlmostEqual(expected_intercept, got_intercept, places=6)
        self.assertAlmostEqual(expected_slope, got_slope, places=6)

    def test_wls_model_stable_weight(self):
        y = [1, 3, 4, 5, 2, 3, 4]
        x = [1, 2, 3, 4, 5, 6, 7]
        w = 1

        self._assert_regression_model(y, x, w, expected=(2.14285714, 0.25))

    def test_wls_model_weight(self):
        y = [1, 3, 4, 5, 2, 3, 4]
        x = [1, 2, 3, 4, 5, 6, 7]
        w = 0.9

        self._assert_regression_model(y, x, w, expected=(2.14285714, 0.25))

    def test_single_point_disallowed(self):
        # this behavior differs from statsmodels.WLS
        x = [10]
        y = [1]
        with self.assertRaises(AssertionError):
            wls.fit_linear_regression(y, x, 1)

    def test_horizontal_line(self):
        x = [0, 1]
        y = [10, 10]
        w = 1

        self._assert_regression_model(y, x, w, expected=(10, 0))

    def test_vertical_line(self):
        # this behavior differs from statsmodels.WLS
        x = [1, 1]
        y = [0, 1]
        w = 1
        intercept, slope = wls.fit_linear_regression(y, x, w)

        self.assertIsNone(intercept)
        self.assertIsNone(slope)

    def test_run_uphill(self):
        x = [0, 1]
        y = [0, 1]
        w = 1
        self._assert_regression_model(y, x, w, expected=(0, 1))

    def test_run_downhill(self):
        x = [1, 0]
        y = [0, 1]
        w = 1
        self._assert_regression_model(y, x, w, expected=(1., -1.))

    def test_wrong_parameters_length(self):
        x = [0, 1, 2, 3]
        y = [1, 2, 0, 1, 1]
        w = 1
        with self.assertRaises(AssertionError):
            wls.fit_linear_regression(y, x, w)

    def test_wrong_weight_type(self):
        x = [0, 1, 2, 3, 1]
        y = [1, 2, 0, 1, 1]
        w = None
        with self.assertRaises(AssertionError):
            wls.fit_linear_regression(y, x, w)

    def test_not_iterable_type_should_fail(self):
        x = 10
        y = [1, 2, 0, 1, 1]
        w = 1
        with self.assertRaises(AssertionError):
            wls.fit_linear_regression(y, x, w)

    def test_iterable_parameters_allowed(self):
        x = (0, 1, 2, 3, 4)
        y = array("i", [1, 2, 0, 1, 1])
        w = 1
        self._assert_regression_model(y, x, w, expected=(1.2, -0.1))


if __name__ == "__main__":
    unittest.main()
