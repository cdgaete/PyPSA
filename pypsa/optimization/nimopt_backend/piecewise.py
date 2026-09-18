# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Method selection and option groups for piecewise curves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from linopy.constants import BREAKPOINT_DIM

from pypsa.optimization.piecewise import piecewise_option_groups

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import xarray as xr

    from pypsa.optimization.piecewise import PiecewiseOptions

TOLERANCE = 1e-10
SIGNS = {"=": "==", "==": "==", "<=": "<=", ">=": ">="}
METHODS = ("auto", "lp", "incremental", "sos2")


def _strictly_monotonic(x: xr.DataArray) -> bool:
    """Return whether the breakpoints of every entity strictly increase or decrease."""
    step = x.diff(BREAKPOINT_DIM)
    rising = ((step > 0) | step.isnull()).all(BREAKPOINT_DIM)
    falling = ((step < 0) | step.isnull()).all(BREAKPOINT_DIM)
    present = step.notnull().any(BREAKPOINT_DIM)
    return bool(((rising | falling) & present).all())


def _curvature_matches(x: xr.DataArray, y: xr.DataArray, sign: str) -> bool:
    """Return whether every curve is convex under `>=` or concave under `<=`."""
    dx = x.diff(BREAKPOINT_DIM)
    slope = y.diff(BREAKPOINT_DIM) / dx
    change = slope.diff(BREAKPOINT_DIM) * np.sign(dx.sum(BREAKPOINT_DIM))
    if sign == ">=":
        return bool(((change >= -TOLERANCE) | change.isnull()).all())
    return bool(((change <= TOLERANCE) | change.isnull()).all())


def resolve_method(
    requested: str,
    sign: str,
    *,
    has_status: bool,
    x_points: xr.DataArray,
    y_points: xr.DataArray,
    owner: str,
) -> str:
    """Return the nimopt method, "tangent" or "incremental", for one group of curves.

    `requested` is a linopy method and `sign` is a nimopt sign. "auto" returns
    "tangent" where linopy's "auto" returns "lp", and "incremental" where the
    x breakpoints are strictly monotonic. Raises NotImplementedError for
    "sos2" and for "auto" with x breakpoints that are not strictly monotonic.
    Raises ValueError for an unknown method.
    """
    if requested not in METHODS:
        msg = f"method of {owner} is one of {METHODS}; got {requested!r}"
        raise ValueError(msg)
    if requested == "sos2":
        msg = (
            f"method 'sos2' of {owner} is not supported by the nimopt backend; "
            f"use method 'auto', 'lp' or 'incremental'"
        )
        raise NotImplementedError(msg)
    if requested == "lp":
        return "tangent"
    if requested == "incremental":
        return "incremental"
    monotonic = _strictly_monotonic(x_points)
    if (
        sign != "=="
        and not has_status
        and monotonic
        and _curvature_matches(x_points, y_points, sign)
    ):
        return "tangent"
    if monotonic:
        return "incremental"
    msg = (
        f"{owner} has x breakpoints that are not strictly monotonic; give "
        f"strictly increasing x breakpoints"
    )
    raise NotImplementedError(msg)


def option_groups(
    names: pd.Index, options: Iterable[PiecewiseOptions], sign: str
) -> list[tuple[str, pd.Index, str, str]]:
    """Return the suffix, names, method and sign of each group, in PyPSA's order.

    The named options come first, sorted by name in reverse, and the k-th has
    the suffix "-option{k}". An option without names covers every remaining
    name. The names no option covers form the last group, with the method
    "auto" and `sign`. A group with no name is omitted.
    """
    groups = []
    named = 0
    for option, covered, method, held in piecewise_option_groups(
        names, options, "auto", sign
    ):
        suffix = ""
        if option is not None and option.name:
            suffix = f"-option{named}"
            named += 1
        if not covered.empty:
            groups.append((suffix, covered, method, held))
    return groups
