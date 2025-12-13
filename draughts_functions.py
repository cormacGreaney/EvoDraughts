"""
Functions for use in draughts strategy grammar.
These functions evaluate board positions and help select moves.
"""

import numpy as np
from draughts_game import DraughtsBoard


def add(a, b):
    """Add two values."""
    return np.add(a, b)


def sub(a, b):
    """Subtract two values."""
    return np.subtract(a, b)


def mul(a, b):
    """Multiply two values."""
    return np.multiply(a, b)


def pdiv(a, b):
    """Protected division (returns 1 if dividing by zero)."""
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b == 0, np.ones_like(a), a / b)
    except ZeroDivisionError:
        return 1.0


def max_(a, b):
    """Return maximum of two values."""
    return np.maximum(a, b)


def min_(a, b):
    """Return minimum of two values."""
    return np.minimum(a, b)


def greater_than_or_equal(a, b):
    """Check if a >= b."""
    return a >= b


def less_than_or_equal(a, b):
    """Check if a <= b."""
    return a <= b


def if_(condition, value_if_true, value_if_false):
    """If condition is true, return value_if_true, else value_if_false."""
    return np.where(condition, value_if_true, value_if_false)


def abs_(a):
    """Absolute value."""
    return np.abs(a)


def neg(a):
    """Negate a value."""
    return -a

