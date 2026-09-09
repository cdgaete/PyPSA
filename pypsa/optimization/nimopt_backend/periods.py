# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The investment-period axis a multi-period network carries."""

from __future__ import annotations

from typing import Any

import nimopt as no
import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.nimopt_backend.scenarios import time_coords


class Periods:
    """The investment-period axis of a network, and the weightings over it.

    A single-period network carries an empty axis, whose sets and labels are
    empty tuples: a family writing `(N, *pe.sets, T)` states one time
    dimension without one and two with it.

    The time set is declared over the timesteps a period carries rather than
    over the whole horizon, so a lag over it stops at a period's edge and no
    row is masked out by hand.
    """

    def __init__(
        self, names: pd.Index | None, timesteps: pd.Index, snapshots: pd.Index
    ) -> None:
        """Hold the period labels, the timesteps each period carries, and the horizon."""
        self.names = names
        self.timesteps = timesteps
        self.snapshots = snapshots
        self._set = (
            no.Set("period", np.asarray(names)) if names is not None else None
        )

    @classmethod
    def empty(cls, sns: pd.Index) -> Periods:
        """Return the axis a network optimised over one horizon carries."""
        return cls(None, sns, sns)

    @classmethod
    def of(cls, sns: pd.Index, multi_investment_periods: bool) -> Periods:
        """Return the axis the horizon carries, empty where periods are not asked for.

        The horizon states one timestep set shared by every period, so the
        snapshots are the product of the periods and the timesteps. A horizon
        whose periods carry different timesteps is refused rather than folded
        into a grid with holes.
        """
        if not multi_investment_periods:
            if isinstance(sns, pd.MultiIndex):
                msg = (
                    "The nimopt backend does not state: a horizon indexed by "
                    "investment period optimised without them; optimise with "
                    "multi_investment_periods=True or state a flat horizon."
                )
                raise NotImplementedError(msg)
            return cls.empty(sns)
        if not isinstance(sns, pd.MultiIndex):
            msg = (
                "multi_investment_periods states a horizon whose snapshots carry "
                "a (period, timestep) index; these carry one level."
            )
            raise TypeError(msg)
        periods = sns.unique("period")
        timesteps = sns.unique("timestep")
        product = pd.MultiIndex.from_product(
            [periods, timesteps], names=["period", "timestep"]
        )
        if not sns.equals(product):
            msg = (
                "The nimopt backend states one timestep set shared by every "
                "investment period, so the snapshots are the periods crossed "
                "with the timesteps; this horizon states "
                f"{len(sns)} snapshots where that product states {len(product)}."
            )
            raise NotImplementedError(msg)
        return cls(periods, timesteps, sns)

    def __bool__(self) -> bool:
        """Whether the horizon is resolved by investment period."""
        return self._set is not None

    @property
    def sets(self) -> tuple:
        """The sets a family states between its own and the time set."""
        return (self._set,) if self._set is not None else ()

    @property
    def labels(self) -> tuple:
        """The label arrays a family states between its own and the timesteps."""
        return (np.asarray(self.names),) if self.names is not None else ()

    @property
    def time_dim(self) -> str:
        """The dimension the time set is read back under before folding."""
        return "timestep" if self else "snapshot"

    def declare(self, m: Any) -> None:
        """Declare the period set and the time set, folded back into `snapshot`."""
        if self._set is not None:
            m.add_set("period", self.names, dim="period")
        m.add_set("snapshot", self.timesteps, dim=self.time_dim)
        if self._set is not None:
            m.fold(("period", "timestep"), "snapshot", self.snapshots)

    @property
    def later(self) -> pd.Index:
        """The snapshots a row reaching one step back stands at.

        A lag over the time set drops the first timestep of every period
        rather than the first snapshot of the horizon, so a family narrowing
        its rows to those the lag leaves standing narrows within each period.
        """
        if not self:
            return pd.Index(self.snapshots.to_numpy()[1:], name="snapshot")
        return pd.MultiIndex.from_product(
            [self.names, self.timesteps[1:]], names=["period", "timestep"]
        )

    @property
    def later_members(self) -> dict:
        """The labels `later` narrows a family's frame to, keyed by dimension."""
        if not self:
            return {"snapshot": self.later}
        return {"timestep": self.timesteps[1:]}

    def back(self, step: int) -> tuple:
        """Return the period set read `step` periods back, wrapping at the first.

        A level continuing over a period boundary reads the period it was last
        active in, and a lag of the whole axis reads the period itself.
        """
        return (self._set.cyclic - step,)

    @property
    def ends(self) -> list:
        """The position of the last snapshot of each period, with the period's own place."""
        if not self:
            return [(len(self.snapshots) - 1, 0)]
        step = len(self.timesteps)
        return [(k * step + step - 1, k) for k in range(len(self.names))]

    def reaches(self, period: Any) -> bool:
        """Whether the horizon carries the investment period a limit names.

        A limit naming no period reaches the whole horizon; one naming a
        period the horizon does not carry states no row at all, as PyPSA
        states none.
        """
        if period is None or not np.isfinite(period):
            return True
        return bool(self) and period in set(self.names)

    def within(self, period: Any) -> np.ndarray:
        """Mark the snapshots of one investment period, or all where none is named."""
        if not self or period is None or not np.isfinite(period):
            return np.ones(len(self.snapshots), dtype=bool)
        return self.snapshots.get_level_values("period").to_numpy() == period

    def weighting(self, n: Any, kind: str) -> Any:
        """Return the period weighting of each snapshot, as an array a reader multiplies.

        `objective` discounts a period's operating cost and `years` states how
        many years a period stands for. An empty axis answers one, which scales
        nothing.
        """
        if not self:
            return 1.0
        weights = n.investment_period_weightings[kind].reindex(self.names)
        return xr.DataArray(
            np.asarray(
                weights.reindex(self.snapshots.get_level_values("period")),
                dtype=np.float64,
            ),
            coords=time_coords(self.snapshots),
            dims=("snapshot",),
        )

    def per_period(self, n: Any, kind: str) -> np.ndarray:
        """Return the period weighting of each period, in the axis's own order."""
        return np.asarray(
            n.investment_period_weightings[kind].reindex(self.names), dtype=np.float64
        )
