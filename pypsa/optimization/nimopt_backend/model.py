# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""A nimopt model standing where PyPSA reads a linopy model."""

from __future__ import annotations

from typing import Any

import nimopt as no
import numpy as np
import pandas as pd
import xarray as xr

from pypsa.optimization.nimopt_backend.scenarios import readable

# nimopt status -> (linopy solver status, linopy termination condition)
STATUS: dict[str, tuple[str, str]] = {
    "optimal": ("ok", "optimal"),
    "infeasible": ("warning", "infeasible"),
    "unbounded": ("warning", "unbounded"),
    "unbounded_or_infeasible": ("warning", "infeasible_or_unbounded"),
    "time_limit": ("warning", "time_limit"),
    "iteration_limit": ("warning", "iteration_limit"),
    "solution_limit": ("warning", "terminated_by_limit"),
    "objective_bound": ("warning", "terminated_by_limit"),
    "objective_target": ("warning", "terminated_by_limit"),
    "memory_limit": ("warning", "resource_interrupt"),
    "interrupt": ("warning", "user_interrupt"),
    "empty": ("warning", "unknown"),
}

SOLVERS = ("highs", "gurobi", "mosek")


def _symbol(name: str) -> str:
    """Return `name` as a name a nimopt expression can address.

    A family is named with hyphens, `Generator-ext-p_nom-lower`, and a set,
    parameter or variable is addressed in an expression, so its name is a
    Python identifier. The families keep their own names: a constraint is
    keyed by one and never stands in an expression.
    """
    return name.replace("-", "_")


# linopy.Model.solve keyword arguments that have no nimopt counterpart
LINOPY_SOLVE_KWARGS = {
    "io_api",
    "problem_fn",
    "solution_fn",
    "keep_files",
    "env",
    "sanitize_zeros",
    "sanitize_infinities",
    "slice_size",
    "remote",
    "progress",
    "explicit_coordinate_names",
    "warmstart_fn",
    "basis_fn",
}


def _translate_options(solver_name: str, solver_options: dict[str, Any]) -> dict:
    """Solver options under nimopt's names.

    PyPSA hands options through under the solver's own names, and nimopt names
    each option once for every solver; an option nimopt's vocabulary lacks is
    refused rather than dropped.
    """
    by_name = {option.name: option for option in no.options()}
    by_native = {option.native: option for option in no.options(solver_name)}
    translated: dict[str, Any] = {}
    for key, value in solver_options.items():
        if key == "log_to_console":
            translated["log"] = bool(value)
            continue
        if key in LINOPY_SOLVE_KWARGS:
            msg = f"linopy's solve argument {key!r} has no nimopt counterpart."
            raise ValueError(msg)
        option = by_name.get(key, by_native.get(key))
        if option is None:
            msg = (
                f"Solver option {key!r} is not in nimopt's vocabulary; "
                f"see nimopt.options({solver_name!r})."
            )
            raise ValueError(msg)
        if option.native_choices:
            back = {native: ours for ours, native in option.native_choices}
            value = back.get(value, value)
        translated[option.name] = value
    return translated


def _narrowed(owner: NimoptModel, frame: xr.DataArray, members: dict) -> xr.DataArray:
    """Narrow `frame` to `members` and fold it, as PyPSA reads the family back.

    A dimension the frame carries is narrowed before the fold; one it gains by
    folding is narrowed after, which is how a family standing at every snapshot
    but the horizon's first states rows no product of labels reaches.
    """
    held = {d: v for d, v in members.items() if d in frame.dims}
    da = owner.fold_frame(frame.sel(held) if held else frame)
    rest = {d: v for d, v in members.items() if d not in held}
    return _ordered(da.sel(rest) if rest else da)


def _ordered(da: xr.DataArray) -> xr.DataArray:
    """Order the dimensions as PyPSA reads a linopy family's.

    PyPSA reads some solutions through `to_pandas`, which places the first
    dimension on the frame's index, so a family read back in another order
    aligns against the network's own frames on nothing.
    """
    if "snapshot" not in da.dims:
        return da
    if "scenario" in da.dims:
        rest = [d for d in da.dims if d not in ("scenario", "snapshot")]
        return da.transpose("scenario", *rest, "snapshot")
    return da.transpose("snapshot", *[d for d in da.dims if d != "snapshot"])


class NimoptVariable:
    """One nimopt variable, read back as linopy's `Variable` is read.

    `solution` is an `xarray.DataArray` over the dimensions linopy declares
    the family over, with `NaN` where the variable states no column.
    """

    def __init__(
        self,
        owner: NimoptModel,
        name: str,
        variable: Any,
        *,
        dims: tuple,
        coords: dict,
        members: dict | None = None,
    ) -> None:
        """Record the variable and the frame it is read back over.

        `members` narrows the frame's labels to those the variable carries
        along a dimension, as linopy declares a variable over its own coords.
        """
        self._owner = owner
        self.name = name
        self.variable = variable
        self._frame = xr.DataArray(
            np.nan, coords={d: coords[d] for d in dims}, dims=dims
        )
        self._members = members or {}
        self._template = _narrowed(owner, self._frame, self._members)

    @property
    def dims(self) -> tuple:
        """Dimension names, as PyPSA reads them back."""
        return self._template.dims

    @property
    def coords(self) -> xr.Coordinates:
        """Coordinates over the dimensions."""
        return self._template.coords

    @property
    def indexes(self) -> Any:
        """Index of each dimension."""
        return self._template.indexes

    @property
    def solution(self) -> xr.DataArray:
        """Primal values, `NaN` where the variable states no column."""
        if self.variable is None:
            return self._template.copy()
        values = (
            self._owner.solution_of().primal(_symbol(self.name)).to_dense(fill=np.nan)
        )
        da = self._frame.copy(data=np.asarray(values, dtype=np.float64))
        return _narrowed(self._owner, da, self._members)

    def __repr__(self) -> str:
        """Name and dimensions."""
        return f"NimoptVariable({self.name!r}, {self.dims})"


class NimoptConstraint:
    """One nimopt constraint, read back as linopy's `Constraint` is read.

    `"dual" in constraint` answers whether the last solve produced duals, and
    `dual` is an `xarray.DataArray` over the constraint's row dimensions.
    """

    def __init__(
        self,
        owner: NimoptModel,
        name: str,
        constraint: Any,
        *,
        dims: tuple,
        coords: dict,
        members: dict | None = None,
    ) -> None:
        """Record the constraint and the frame its duals are read back over.

        `members` narrows the frame's labels to those the constraint states
        rows for along a dimension, as linopy's constraint coords do.
        """
        self._owner = owner
        self.name = name
        self.constraint = constraint
        self._frame = xr.DataArray(
            np.nan, coords={d: coords[d] for d in dims}, dims=dims
        )
        self._members = members or {}
        self._template = _narrowed(owner, self._frame, self._members)

    @property
    def dims(self) -> tuple:
        """Dimension names, as PyPSA reads them back."""
        return self._template.dims

    @property
    def coords(self) -> xr.Coordinates:
        """Coordinates over the dimensions."""
        return self._template.coords

    @property
    def indexes(self) -> Any:
        """Index of each dimension."""
        return self._template.indexes

    def __contains__(self, key: str) -> bool:
        """Answer `"dual" in constraint` as linopy does: whether duals exist."""
        return key == "dual" and self._owner.has_duals

    @property
    def dual(self) -> xr.DataArray:
        """Dual values over the constraint's row dimensions."""
        values = self._owner.solution_of().dual(self.name).to_dense(fill=np.nan)
        da = self._frame.copy(data=np.asarray(values, dtype=np.float64))
        return _narrowed(self._owner, da, self._members)

    def __repr__(self) -> str:
        """Name and dimensions."""
        return f"NimoptConstraint({self.name!r}, {self.dims})"


class _Objective:
    """The objective as linopy exposes it: a `value` once solved."""

    def __init__(self, owner: NimoptModel) -> None:
        """Bind to the model whose solve the value is read from."""
        self._owner = owner

    @property
    def value(self) -> float:
        return float(self._owner.solution_of().objective)

    @property
    def sense(self) -> str:
        return self._owner.model.sense


class NimoptModel:
    """A nimopt model with the registries PyPSA reads from a linopy model.

    Sets are declared once and shared by every variable and constraint; each
    is recorded with the dimension name PyPSA reads it back under, so a set
    named after a component is read back as `name`.
    """

    def __init__(self, name: str = "pypsa", sense: str = "min") -> None:
        """Open an empty nimopt model with empty registries."""
        self.model = no.Model(name, sense)
        self.sets: dict[str, Any] = {}
        self.variables: dict[str, NimoptVariable] = {}
        self.constraints: dict[str, NimoptConstraint] = {}
        self.parameters = xr.Dataset()
        self.objective = _Objective(self)
        self.solution: Any = None
        self.solver_model: Any = None
        self.solver_name: str | None = None
        self.status: str | None = None
        self.termination_condition: str | None = None
        self._dim_of: dict[str, str] = {}
        self._labels: dict[str, Any] = {}
        self._folding: tuple | None = None

    def fold(self, dims: tuple, into: str, index: pd.Index) -> None:
        """State that `dims` are read back as the single dimension `into`.

        A model over investment periods declares a period set and a timestep
        set, and PyPSA reads each family over one `snapshot` dimension whose
        index is the pair. `index` is the order that dimension carries, which
        stacking the pair reproduces entry for entry.
        """
        self._folding = (tuple(dims), into, index)

    def fold_frame(self, da: xr.DataArray) -> xr.DataArray:
        """Return `da` with the folded dimensions stacked into one, where it carries them."""
        if self._folding is None:
            return da
        dims, into, _ = self._folding
        if not set(dims) <= set(da.dims):
            return da
        return da.stack({into: dims})

    def __repr__(self) -> str:
        """Return the wrapped nimopt model."""
        return f"NimoptModel({self.model!r})"

    def add_set(self, name: str, labels: Any, dim: str) -> Any:
        """Declare the set `name` over `labels`, read back under `dim`."""
        if name in self.sets:
            msg = f"Set {name!r} is already declared."
            raise ValueError(msg)
        index = labels if isinstance(labels, pd.Index) else pd.Index(labels)
        held = no.Set(name, readable(index.to_numpy()))
        self.sets[name] = held
        self._dim_of[name] = dim
        self._labels[name] = index
        return held

    def labels(self, name: str) -> pd.Index:
        """Return the labels set `name` was declared over."""
        return self._labels[name]

    def add_variables(
        self, name: str, sets: tuple, members: dict | None = None, **kwargs: Any
    ) -> Any:
        """Declare a variable over `sets`, returning the nimopt variable.

        `members` names, per dimension, the labels the variable is read back
        over where it carries columns for a subset of a set.

        The family keeps PyPSA's name here, which the solution assignment
        reads it back by; the nimopt variable takes the identifier form of it,
        because a variable stands in an expression.
        """
        variable = self.model.var(_symbol(name), sets, **kwargs)
        dims, coords = self._frame(sets)
        self.variables[name] = NimoptVariable(
            self, name, variable, dims=dims, coords=coords, members=members
        )
        return variable

    def add_empty_variables(self, name: str, dims: tuple, coords: dict) -> None:
        """Register a variable with no columns, as linopy declares one over no coords.

        PyPSA's solution assignment then writes zeros for every component of
        the family, as it does when linopy's variable is empty.
        """
        empty = {d: coords.get(d, pd.Index([], name=d)) for d in dims}
        self.variables[name] = NimoptVariable(self, name, None, dims=dims, coords=empty)

    def add_constraints(
        self, name: str, relation: Any, members: dict | None = None, **kwargs: Any
    ) -> Any:
        """Declare a constraint from a relation, returning the nimopt constraint.

        `members` names, per dimension, the labels the constraint states rows
        for where they are a subset of a set.
        """
        constraint = self.model.constraint(name, relation, **kwargs)
        sets = tuple(self.sets[d] for d in constraint.frame)
        dims, coords = self._frame(sets)
        self.constraints[name] = NimoptConstraint(
            self, name, constraint, dims=dims, coords=coords, members=members
        )
        return constraint

    def add_objective(self, expression: Any) -> None:
        """Set the objective the model minimises."""
        self.model.set_objective(expression)

    def var(self, name: str) -> Any:
        """Return the nimopt variable declared under `name`."""
        return self.variables[name].variable

    def __getitem__(self, name: str) -> NimoptVariable:
        """Return the variable declared under `name`, as `m[name]` does in linopy."""
        return self.variables[name]

    def _frame(self, sets: tuple) -> tuple[tuple, dict]:
        dims = tuple(self._dim_of[s.name] for s in sets)
        if len(set(dims)) != len(dims):
            msg = f"Sets {[s.name for s in sets]} share a dimension name."
            raise ValueError(msg)
        coords = {self._dim_of[s.name]: self._labels[s.name] for s in sets}
        return dims, coords

    def solution_of(self) -> Any:
        """Return the last solve's answer, refusing before a solve."""
        if self.solution is None:
            msg = "The model has not been solved."
            raise ValueError(msg)
        return self.solution

    @property
    def has_duals(self) -> bool:
        """Whether the last solve produced duals PyPSA can assign."""
        return (
            self.solution is not None
            and self.solution.status == "optimal"
            and not self.model.integrality().any()
        )

    def solve(
        self, solver_name: str = "highs", progress: bool = False, **kwargs: Any
    ) -> tuple[str, str]:
        """Solve through nimopt, answering linopy's `(status, condition)` pair."""
        if solver_name not in SOLVERS:
            msg = f"nimopt solves through {SOLVERS}; got {solver_name!r}."
            raise ValueError(msg)
        options = _translate_options(solver_name, kwargs)
        self.solver_name = solver_name
        self.solution = self.model.solve(solver_name, options or None, progress)
        self.status, self.termination_condition = STATUS[self.solution.status]
        return self.status, self.termination_condition
