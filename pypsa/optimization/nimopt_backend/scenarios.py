# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The scenario axis a stochastic network carries, and the readers that answer over it."""

from __future__ import annotations

from typing import Any

import nimopt as no
import numpy as np
import pandas as pd
import xarray as xr


def readable(values: np.ndarray) -> np.ndarray:
    """Return labels a nimopt file can save.

    An object array is converted to strings. nimopt raises ValueError when it
    saves a set whose labels are an object array.
    """
    return values.astype(str) if values.dtype.hasobject else values


def labels_of(da: xr.DataArray, dim: str) -> np.ndarray:
    """Labels of one dimension, as the array a nimopt set is declared over."""
    return readable(da.indexes[dim].to_numpy())


def axes_of(da: xr.DataArray, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    """One axis per dimension and one value column, for the cells `mask` marks.

    An axis is the labels of a dimension and the position of each marked cell
    along it, so a label is read once however many cells stand at it. A
    dimension indexed by pairs -- the snapshots of a multi-period horizon --
    states one axis per level, under the level's own name.
    """
    at = np.nonzero(np.asarray(mask))
    axes = {}
    for k, dim in enumerate(da.dims):
        index = da.indexes[dim]
        if isinstance(index, pd.MultiIndex):
            for level, name in enumerate(index.names):
                labels = readable(index.levels[level].to_numpy())
                axes[name] = (labels, np.asarray(index.codes[level])[at[k]])
            continue
        axes[dim] = (readable(index.to_numpy()), at[k])
    values = np.asarray(da.to_numpy(), dtype=np.float64)[at]
    return axes, values


def time_coords(sns: pd.Index) -> Any:
    """Coordinates a `snapshot` dimension labelled by `sns` carries.

    A multi-period horizon indexes its snapshots by a pair, and an array
    stating the pair as levels aligns with one that does not only by accident,
    so every array the backend builds states them.
    """
    if isinstance(sns, pd.MultiIndex):
        return xr.Coordinates.from_pandas_multiindex(sns, "snapshot")
    return {"snapshot": sns}


class Scenarios:
    """The scenario axis of a stochastic network, and the readers over it.

    A deterministic network carries an empty axis, whose sets and labels are
    empty tuples: a family prepending them states what it states without one.
    """

    def __init__(self, names: pd.Index | None, weights: np.ndarray | None) -> None:
        """Hold the scenario names and their probabilities, or neither."""
        self.names = names
        self.weights = weights
        self._set = (
            no.Set("scenario", readable(np.asarray(names)))
            if names is not None
            else None
        )

    @classmethod
    def empty(cls) -> Scenarios:
        """Return the axis a deterministic network carries."""
        return cls(None, None)

    @classmethod
    def of(cls, n: Any) -> Scenarios:
        """Return the axis the network carries, empty where it states no scenarios."""
        if not n.has_scenarios:
            return cls.empty()
        weighting = n.scenario_weightings["weight"]
        return cls(
            pd.Index(n.scenarios, name="scenario"),
            np.asarray(weighting.loc[list(n.scenarios)], dtype=np.float64),
        )

    def __bool__(self) -> bool:
        """Whether the network states scenarios."""
        return self._set is not None

    @property
    def sets(self) -> tuple:
        """The sets a family prepends to its own."""
        return (self._set,) if self._set is not None else ()

    @property
    def labels(self) -> tuple:
        """The label arrays a family prepends to its own."""
        return (readable(np.asarray(self.names)),) if self.names is not None else ()

    @property
    def probability(self) -> Any:
        """The probability of each scenario, as an array a coefficient multiplies.

        It carries the scenario dimension by name, so multiplying a reader's
        answer by it aligns on that dimension whatever order the reader
        answered in. An empty axis answers one, which scales nothing.
        """
        if self.weights is None:
            return 1.0
        return xr.DataArray(
            self.weights,
            coords={"scenario": readable(np.asarray(self.names))},
            dims=("scenario",),
        )

    def declare(self, m: Any) -> None:
        """Declare the scenario set on the model, where the network states one."""
        if self._set is not None:
            m.add_set("scenario", self.names, dim="scenario")

    def over(self, da: xr.DataArray, dims: tuple) -> xr.DataArray:
        """Order an array's axes as the scenario axis and `dims` state them."""
        if not self and "scenario" in da.dims:
            msg = (
                "An empty scenario axis read an array carrying a scenario "
                "dimension; the axis and the network disagree."
            )
            raise ValueError(msg)
        wanted = ("scenario", *dims) if self else dims
        return da.transpose(*wanted)

    def _timed(self, da: xr.DataArray, sns: pd.Index) -> xr.DataArray:
        if "snapshot" in da.dims:
            return da.sel(snapshot=sns)
        return da.expand_dims(snapshot=len(sns)).assign_coords(time_coords(sns))

    def static(self, c: Any, attr: str, names: pd.Index) -> xr.DataArray:
        """Read a static attribute of the named components, over the axis and `name`."""
        return self.over(c.da[attr].sel(name=names), ("name",))

    def grid(self, c: Any, attr: str, sns: pd.Index, names: pd.Index) -> xr.DataArray:
        """Read an attribute over the axis, `name` and `snapshot`, broadcasting a static value."""
        da = self._timed(c.da[attr].sel(name=names), sns)
        return self.over(da, ("name", "snapshot"))

    def bounds_pu(
        self, c: Any, attr: str, sns: pd.Index, names: pd.Index
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Per-unit lower and upper bounds, each over the axis, `name` and `snapshot`.

        `get_bounds_pu` answers its two arrays in different dimension orders,
        so each is ordered here rather than read at the order it arrives in.
        """
        out = [
            self.over(self._timed(da.sel(name=names), sns), ("name", "snapshot"))
            for da in c.get_bounds_pu(attr=attr)
        ]
        return out[0], out[1]

    def active(self, c: Any, sns: pd.Index, names: pd.Index) -> xr.DataArray:
        """Whether each named component is active, over the axis, `name` and `snapshot`."""
        return self.over(
            c.da.active.sel(name=names, snapshot=sns), ("name", "snapshot")
        )

    def ones(self, names: np.ndarray, sns: pd.Index | None = None) -> xr.DataArray:
        """Build a unit array over the axis, `name`, and `snapshot` where one is given.

        A coefficient valued one over a row's whole frame carries a variable
        the row's other dimensions do not reach into it, which is how a
        decision taken once enters a row stated per scenario.
        """
        coords: dict = {}
        if self.names is not None:
            coords["scenario"] = readable(np.asarray(self.names))
        coords["name"] = readable(np.asarray(names))
        dims = (*coords, "snapshot") if sns is not None else tuple(coords)
        shape = tuple(len(coords[d]) for d in coords)
        if sns is not None:
            shape = (*shape, len(sns))
        held = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
        return held if sns is None else held.assign_coords(time_coords(sns))
