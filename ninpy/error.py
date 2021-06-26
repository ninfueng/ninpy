#!/usr/bin/env python3
"""Collections of all warnings and errors in ninpy."""


class DuplicationWarning(UserWarning):
    """Warning for Duplicated function or class that considered to be remove in future iteration."""


class EarlyStoppingException(Exception):
    """Exception for catching early stopping. For exiting out of loop."""
