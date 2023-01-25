########################################################################
#
# wls - Calculate slope & intercept using weighted least squares method.
# does not require any extra dependencies e.g. statsmodels, pandas, numpy
#
########################################################################


from collections.abc import Iterable
from numbers import Number


def fit_linear_regression(y, x, w=1):
    """fits WLS linear regression"""
    assert isinstance(y, Iterable) and isinstance(x, Iterable)
    assert isinstance(w, Iterable) or isinstance(w, Number)
    elements_count = len(x)
    assert elements_count >= 2

    if isinstance(w, Number):
        w = [w] * elements_count

    assert len(w) == len(y) == elements_count

    sum_w = 0
    sum_of_w_by_x_squared = 0
    sum_of_x_by_y_by_w = 0
    sum_of_x_by_w = 0
    sum_of_y_by_w = 0

    for x_i, y_i, w_i in zip(x, y, w):
        sum_w += w_i
        x_i_by_w_i = x_i * w_i
        sum_of_x_by_w += x_i_by_w_i
        sum_of_x_by_y_by_w += x_i_by_w_i * y_i
        sum_of_y_by_w += w_i * y_i
        sum_of_w_by_x_squared += x_i_by_w_i * x_i

    dividend = sum_w * sum_of_x_by_y_by_w - sum_of_x_by_w * sum_of_y_by_w
    divisor = sum_w * sum_of_w_by_x_squared - sum_of_x_by_w ** 2
    if not divisor:
        return None, None
    slope = dividend / divisor
    intercept = (sum_of_y_by_w - slope * sum_of_x_by_w) / sum_w

    return intercept, slope
