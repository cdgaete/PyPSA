# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Method selection and option groups for piecewise curves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypsa.optimization.piecewise import piecewise_option_groups

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from pypsa.optimization.piecewise import PiecewiseOptions

SIGNS = {"=": "==", "==": "==", "<=": "<=", ">=": ">="}
METHODS = ("auto", "lp", "incremental", "sos2")


def resolve_method(requested: str, *, has_status: bool, owner: str) -> str:
    """Return the nimopt method for one group of curves.

    `requested` is a linopy method. "lp" returns "tangent" and "incremental"
    returns "incremental". "auto" returns nimopt's "auto", and "incremental"
    for a group with a status. Raises NotImplementedError for "sos2" and
    ValueError for an unknown method.
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
    if requested == "incremental" or has_status:
        return "incremental"
    return "auto"


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
