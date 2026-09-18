# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""State a network's optimisation problem in nimopt.

Each function restates one family of PyPSA's linopy formulation under the same
variable and constraint names, so the solution and dual assignment read the
model as they read linopy's. A feature the restatement does not cover raises
before anything is built, rather than building a model that omits it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy.constants import BREAKPOINT_DIM
from nimopt import Param, Set, Sum, subset

from pypsa.common import as_index
from pypsa.constants import PYPSA_DATA_DIR
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.common import get_bus_counts
from pypsa.optimization.constraints import _get_delay_config
from pypsa.optimization.nimopt_backend.model import NimoptModel, _symbol
from pypsa.optimization.nimopt_backend.periods import Periods
from pypsa.optimization.nimopt_backend.piecewise import (
    SIGNS,
    breakpoint_param,
    option_groups,
    resolve_method,
)
from pypsa.optimization.nimopt_backend.scenarios import (
    Scenarios,
    columns_of,
    readable,
    time_coords,
)
from pypsa.optimization.piecewise import _get_breakpoints, get_piecewise_names
from pypsa.optimization.window import SnapshotWindow

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)

FREE = {"lower": -np.inf, "upper": np.inf}
KIRCHHOFF_SCALE = 1e5
DEG_TO_RAD = np.pi / 180.0

lookup = pd.read_csv(
    PYPSA_DATA_DIR / "variables.csv", index_col=["component", "variable"]
)
OPERATIONAL = list(lookup.query("not nominal and not handle_separately").index)
NOMINAL = list(nominal_attrs.items())
# the components PyPSA states a ramp limit for, of those this backend carries
RAMPING = (("Generator", "p"), ("Link", "p"))
COST_TYPES = ("marginal_cost", "marginal_cost_storage", "spill_cost")

# (component, variable attribute, bus column, sign), as PyPSA's nodal balance
BALANCE_TERMS = (
    ("Generator", "p", "bus", 1.0),
    ("Store", "p", "bus", 1.0),
    ("StorageUnit", "p_dispatch", "bus", 1.0),
    ("StorageUnit", "p_store", "bus", -1.0),
    ("Line", "s", "bus0", -1.0),
    ("Line", "s", "bus1", 1.0),
    ("Transformer", "s", "bus0", -1.0),
    ("Transformer", "s", "bus1", 1.0),
    ("Link", "p", "bus0", -1.0),
)

# components whose ports each carry a rate the balance reads at that bus
MULTIPORT = ("Link", "Process")

GLOBAL_TYPES = (
    "primary_energy",
    "operational_limit",
    "tech_capacity_expansion_limit",
    "transmission_volume_expansion_limit",
    "transmission_expansion_cost_limit",
)
# global constraint types PyPSA states and the nimopt backend does not
UNSTATED_GLOBAL_TYPES = ()

# families the backend does not yet state under a scenario dimension
SCENARIO_UNSTATED = ()

# families the backend does not yet state under an investment-period dimension
PERIOD_UNSTATED = ()


# --- refusing what is not stated ------------------------------------------


def _refuse_unsupported(
    n: Network,
    pe: Periods,
    transmission_losses: Any,
) -> None:
    """Raise for every feature of the network the nimopt backend does not state."""
    reasons = []
    if pe and PERIOD_UNSTATED:
        reasons.append("under investment periods: " + ", ".join(PERIOD_UNSTATED))
    if n.has_scenarios and SCENARIO_UNSTATED:
        reasons.append("under scenarios: " + ", ".join(SCENARIO_UNSTATED))
    if transmission_losses and not _tangent_segments(transmission_losses):
        reasons.append(
            "transmission losses by secants; state a whole number of tangent "
            "segments instead"
        )
    for c_name in sorted({c for c, _ in NOMINAL} | {c for c, _ in OPERATIONAL}):
        c = n.c[c_name]
        if c.static.empty:
            continue
        stated = {name for name, _ in COMMITTABLE}
        committable = c.committables.intersection(c.active_assets)
        if not committable.empty and c.name not in stated:
            reasons.append(f"committable {c.name} components")
        modular = c.modulars.intersection(c.active_assets)
        held = modular.intersection(c.committables).difference(c.extendables)
        if not held.empty:
            reasons.append(f"committable modular {c.name} components of fixed capacity")
        if not c.maintainables.empty:
            reasons.append(f"maintainable {c.name} components")
        for attr, frame in c.piecewise.items():
            if frame is None or frame.empty:
                continue
            if n.has_scenarios:
                reasons.append(f"piecewise {c.name} {attr} under scenarios")
            elif attr not in ("marginal_cost", "capital_cost"):
                reasons.append(f"piecewise {c.name} {attr}")
        if c.name not in {name for name, _ in RAMPING}:
            for col in ("ramp_limit_up", "ramp_limit_down"):
                if col in c.static and c.static[col].notna().any():
                    reasons.append(f"{c.name} {col}")
                if col in c.dynamic and not c.dynamic[col].empty:
                    reasons.append(f"{c.name} time-varying {col}")
    glcs = n.c.global_constraints.static
    if not glcs.empty:
        other = sorted(set(glcs["type"]) & set(UNSTATED_GLOBAL_TYPES))
        if other:
            reasons.append(f"global constraints of type {other}")
        if not pe and glcs["investment_period"].notna().any():
            reasons.append(
                "a global constraint bound to an investment period on a horizon "
                "carrying none"
            )
    buses = n.c.buses.static
    bound = [c for c in buses.columns if c.startswith(("nom_min_", "nom_max_"))]
    if bound:
        reasons.append(
            f"per-bus-carrier capacity bounds {bound}, which PyPSA deprecates in "
            f"favour of a global constraint of type 'tech_capacity_expansion_limit'"
        )
    if reasons:
        msg = "The nimopt backend does not state: " + "; ".join(reasons) + "."
        raise NotImplementedError(msg)


# --- reading the network ---------------------------------------------------


def _names(index: pd.Index) -> np.ndarray:
    """Component names as the label array a nimopt set or parameter reads.

    The labels carry a string dtype rather than object, because a file saves
    them with `numpy.savez`, which pickles an object array silently and reads
    nothing back under `allow_pickle=False`.
    """
    return np.asarray(index, dtype=str)


def _per_component(c: Any, column: str, names: pd.Index, dtype: Any) -> np.ndarray:
    """Read a static column stating one value per component, whatever the scenario.

    A coefficient placing a component's column in a row carries no scenario
    dimension, and neither does the bus a component sits on, so a column
    reaching one of those is refused where it varies by scenario.
    """
    held = c.static[column]
    if not isinstance(held.index, pd.MultiIndex):
        return np.asarray(held.loc[names], dtype=dtype)
    frame = held.unstack("scenario").loc[names]
    if not frame.eq(frame.iloc[:, 0], axis=0).all().all():
        msg = (
            f"{c.name}.{column} varies by scenario; it states one value per component."
        )
        raise NotImplementedError(msg)
    return np.asarray(frame.iloc[:, 0], dtype=dtype)


def _bus_of(c: Any, names: pd.Index, column: str) -> np.ndarray:
    """Read the bus each named component sits on, as one label per component."""
    return _per_component(c, column, names, str)


def _active(c: Any, sns: pd.Index, names: pd.Index, sc: Scenarios) -> xr.DataArray:
    """Where each named component is active, as the array a row domain reads.

    A component carries a build year and a lifetime, so it is active in some
    investment periods and not others, and its rows exist over the periods it
    is active in. A horizon stating no periods answers ones, because a
    component inactive there is not among the active assets at all.
    """
    return sc.active(c, sns, names).astype(np.float64)


# --- parameters and row domains --------------------------------------------


# the dimensions a reader answers each kind of set over; a set named after a
# component is answered over `name`. The time set reads `timestep` where the
# horizon states investment periods and `snapshot` where it does not, because
# a reader answers the snapshots as a pair of levels under periods.
DIM_OF_SET = {
    "scenario": ("scenario",),
    "snapshot": ("timestep", "snapshot"),
    "period": ("period",),
    "cycle": ("cycle",),
}


def _by_set(sets: tuple, columns: dict) -> dict:
    """Key a reader's label columns by the set each dimension is declared over.

    A reader names its axes `scenario`, `name`, `period` and `timestep`, and a
    set is named for the component it stands over, so the two vocabularies are
    paired by dimension name rather than by position.
    """
    held = {}
    for s in sets:
        dims = DIM_OF_SET.get(s.name, ("name",))
        found = [d for d in dims if d in columns]
        if not found:
            msg = f"A reader answered no {dims} dimension for set {s.name!r}."
            raise ValueError(msg)
        held[s.name] = columns[found[0]]
    return held


def _sparse_of(name: str, sets: tuple, da: xr.DataArray) -> Param:
    """Build a coefficient carrying the array's nonzeros; a zero is absent."""
    columns, values = columns_of(da, np.asarray(da.to_numpy()) != 0)
    return Param.from_long(_symbol(name), sets, _by_set(sets, columns), values)


def _long_of(
    name: str, sets: tuple, da: xr.DataArray, keep: np.ndarray | None = None
) -> Param:
    """Build a parameter valued at every cell of the array, or at the cells `keep` marks."""
    mask = np.ones(da.shape, dtype=bool) if keep is None else np.asarray(keep)
    columns, values = columns_of(da, mask)
    return Param.from_long(_symbol(name), sets, _by_set(sets, columns), values)


def _at_snapshots(pe: Periods, T: Any, sns: pd.Index, at: np.ndarray) -> dict:
    """Label columns naming the snapshots `at` positions, keyed by time set."""
    if not pe:
        return {T.name: readable(sns.to_numpy())[at]}
    return {
        "period": sns.get_level_values("period").to_numpy()[at],
        T.name: readable(sns.get_level_values("timestep").to_numpy())[at],
    }


def _dense_of(name: str, sets: tuple, values: np.ndarray, pe: Periods) -> Param:
    """Build a parameter from a dense grid whose last axis runs over the snapshots.

    A horizon resolved by investment period states the snapshots as the
    periods crossed with the timesteps, in that order, so that axis splits in
    two by reshaping and every value stays where it stood.
    """
    if pe:
        values = values.reshape(*values.shape[:-1], len(pe.names), len(pe.timesteps))
    return Param.from_dense(name, sets, np.ascontiguousarray(values))


def _domain_of(name: str, sets: tuple, da: xr.DataArray) -> Any:
    """Return the coordinates the array marks, as something a file can address."""
    columns, _ = columns_of(da, np.asarray(da.to_numpy()).astype(bool))
    return _named_domain(name, sets, _by_set(sets, columns))


def _named_domain(name: str, sets: tuple, columns: dict) -> Any:
    """Return the coordinates `columns` names, as something a file can address.

    A membership reaching every cell of the product is the sets themselves,
    which a file states as their names. One reaching fewer is a parameter
    valued 1.0 where it reaches, so the file names it and carries its members
    beside the rest of the data. A domain states the same coordinates and
    carries no name, so a model holding one cannot be written.
    """
    held = subset(sets, columns)
    if held.is_full:
        return sets
    return Param(_symbol(name), sets, held.array(np.ones(held.size)))


# --- variables ---------------------------------------------------------------


def define_sets(n: Network, m: NimoptModel, sc: Scenarios, pe: Periods) -> None:
    """One set per component carrying a variable, one for buses, one for time, one for scenarios."""
    sc.declare(m)
    pe.declare(m)
    m.add_set("Bus", n.c.buses.static.index.unique("name"), dim="name")
    for c_name in sorted({c for c, _ in NOMINAL} | {c for c, _ in OPERATIONAL}):
        c = n.c[c_name]
        if c.static.empty or c.active_assets.empty:
            continue
        m.add_set(c_name, c.active_assets, dim="name")


def _empty_frame(sns: pd.Index, sc: Scenarios) -> tuple[tuple, dict]:
    """Return the dimensions and coordinates a variable with no columns is read back over."""
    coords: dict = {"snapshot": sns}
    if sc:
        coords["scenario"] = sc.names
        return ("scenario", "name", "snapshot"), coords
    return ("name", "snapshot"), coords


def define_variables(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    linearized: bool,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Every variable PyPSA declares for the LP core, under its own name."""
    T = m.sets["snapshot"]
    for c_name, attr in NOMINAL:
        c = n.c[c_name]
        if c.static.empty:
            continue
        ext = c.extendables.intersection(c.active_assets)
        if ext.empty:
            continue
        N = m.sets[c_name]
        name = f"{c_name}-{attr}"
        members = _named_domain(f"{name}-members", (N,), {N.name: _names(ext)})
        m.add_variables(name, (N,), subset=members, members={"name": ext}, **FREE)

    for c_name, attr in [*OPERATIONAL, ("Store", "p")]:
        c = n.c[c_name]
        if c.static.empty:
            continue
        name = f"{c_name}-{attr}"
        if c.active_assets.empty:
            m.add_empty_variables(name, *_empty_frame(sns, sc))
            continue
        names = c.active_assets
        sets = sc.sets + (m.sets[c_name], *pe.sets, T)
        m.add_variables(
            name,
            sets,
            subset=_domain_of(f"{name}-members", sets, _active(c, sns, names, sc)),
            **FREE,
        )

    define_spillage_variables(n, m, sns, sc, pe)
    define_phase_shift_variables(n, m, sns, sc, pe)
    define_commitment_variables(n, m, sns, linearized, sc=sc, pe=pe)
    define_modular_variables(n, m)


def define_cvar_variables(n: Network, m: NimoptModel, sc: Scenarios) -> None:
    """Declare the columns a risk-averse objective prices.

    `CVaR-a` carries each scenario's operating cost above the value-at-risk
    level, `CVaR-theta` is that level, and `CVaR` is the tail average the
    objective blends in. A network stating no risk preference declares none.
    """
    if not n.has_risk_preference:
        return
    if not sc:
        msg = "A risk preference states a tail over scenarios; this network has none."
        raise ValueError(msg)
    m.add_variables("CVaR-a", sc.sets, lower=0.0, upper=np.inf)
    m.add_variables("CVaR-theta", (), **FREE)
    m.add_variables("CVaR", (), **FREE)


def _shifting(n: Network) -> Any:
    """Return the active transformers whose phase shift is chosen."""
    c = n.c.transformers
    if c.static.empty:
        return c, c.static.index[:0]
    active = c.static.loc[c.active_assets]
    varying = active["phase_shift_min"] < active["phase_shift_max"]
    return c, active.index[varying]


def define_phase_shift_variables(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Declare a phase shift column per shifting transformer and snapshot.

    A transformer whose shift is bounded below its maximum states the angle as
    a column the optimisation chooses, in degrees. It enters the cycle rows and
    states no row of its own.
    """
    c, names = _shifting(n)
    if names.empty:
        return
    TR, T = m.sets["Transformer"], m.sets["snapshot"]
    sets = sc.sets + (TR, *pe.sets, T)
    m.add_variables(
        "Transformer-phase_shift",
        sets,
        subset=_domain_of(
            "Transformer-phase_shift-members", sets, sc.ones(_names(names), sns)
        ),
        lower=_long_of(
            "Transformer-phase_shift-lower",
            sets,
            sc.grid(c, "phase_shift_min", sns, names),
        ),
        upper=_long_of(
            "Transformer-phase_shift-upper",
            sets,
            sc.grid(c, "phase_shift_max", sns, names),
        ),
        members={"name": names},
    )


def define_spillage_variables(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Declare a spill column wherever inflow reaches a storage unit, bounded by it."""
    c = n.c.storage_units
    if c.static.empty or c.active_assets.empty:
        return
    names = c.active_assets
    inflow = sc.grid(c, "inflow", sns, names)
    live = inflow > 0
    if not bool(live.any()):
        return
    U, T = m.sets["StorageUnit"], m.sets["snapshot"]
    sets = sc.sets + (U, *pe.sets, T)
    members = _domain_of("StorageUnit-spill-members", sets, live)
    upper = _long_of("StorageUnit-spill-upper", sets, inflow, keep=live.to_numpy())
    m.add_variables("StorageUnit-spill", sets, subset=members, lower=0.0, upper=upper)


# --- capacity rows -----------------------------------------------------------


def define_nominal_constraints(n: Network, m: NimoptModel, sc: Scenarios) -> None:
    """Bounds on each extendable capacity, an upper row only where finite."""
    for c_name, attr in NOMINAL:
        c = n.c[c_name]
        if c.static.empty:
            continue
        ext = c.extendables.intersection(c.active_assets)
        if ext.empty:
            continue
        N = m.sets[c_name]
        var = m.var(f"{c_name}-{attr}")
        sets = sc.sets + (N,)
        axes = (*sc.sets, N)
        one = _long_of(f"{c_name}-{attr}-unit", sets, sc.ones(_names(ext)))
        lower = sc.static(c, attr + "_min", ext)
        upper = sc.static(c, attr + "_max", ext)
        name = f"{c_name}-ext-{attr}-lower"
        m.add_constraints(
            name,
            one[*axes] * var[N] >= _long_of(name, sets, lower)[*axes],
            members={"name": ext},
        )
        finite = np.isfinite(upper.to_numpy())
        if finite.any():
            name = f"{c_name}-ext-{attr}-upper"
            bound = _long_of(name, sets, upper, keep=finite)
            held = finite.any(axis=0) if sc else finite
            m.add_constraints(
                name,
                one[*axes] * var[N] <= bound[*axes],
                members={"name": ext[held]},
            )


def define_fixed_nominal_constraints(n: Network, m: NimoptModel, sc: Scenarios) -> None:
    """Fix an extendable capacity to the value its `_set` attribute states."""
    for c_name, attr in NOMINAL:
        c = n.c[c_name]
        if c.static.empty or attr + "_set" not in c.static:
            continue
        ext = c.extendables.intersection(c.active_assets)
        if ext.empty:
            continue
        values = sc.static(c, attr + "_set", ext)
        stated = np.isfinite(values.to_numpy())
        at = stated.all(axis=0) if sc else stated
        if not at.any():
            continue
        if sc and not np.array_equal(stated.any(axis=0), at):
            msg = (
                f"{c.name} states {attr}_set in some scenarios and not others; "
                f"the capacity it fixes is decided once, so it is stated in "
                f"every scenario or in none."
            )
            raise NotImplementedError(msg)
        N = m.sets[c_name]
        sets = sc.sets + (N,)
        axes = (*sc.sets, N)
        name = f"{c_name}-{attr}_set"
        held = ext[at]
        one = _long_of(f"{name}-unit", sets, sc.ones(_names(held)))
        value = _long_of(name, sets, sc.static(c, attr + "_set", held))
        m.add_constraints(
            name,
            one[*axes] * m.var(f"{c_name}-{attr}")[N] == value[*axes],
            members={"name": held},
        )


# --- operating rows ----------------------------------------------------------


def _loss_term(m: NimoptModel, c_name: str, names: Any, losses: bool) -> Any:
    """Return the loss variable a branch carries, or None where none is stated."""
    if not losses or c_name not in LOSSY or names.empty:
        return None
    return m.var(f"{c_name}-loss")


def define_operational_constraints_for_non_extendables(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    losses: bool,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Bound a fixed component's operation by per-unit shares of its capacity."""
    T = m.sets["snapshot"]
    for c_name, attr in OPERATIONAL:
        c = n.c[c_name]
        if c.static.empty:
            continue
        fix = c.fixed.difference(c.committables).intersection(c.active_assets)
        if fix.empty:
            continue
        N = m.sets[c_name]
        sets = sc.sets + (N, *pe.sets, T)
        axes = (*sc.sets, N, *pe.sets, T)
        nominal = sc.static(c, c._operational_attrs["nom"], fix)
        min_pu, max_pu = sc.bounds_pu(c, attr, sns, fix)
        is_inf = np.isinf(nominal)
        with np.errstate(invalid="ignore"):
            lower = sc.over(
                xr.where(is_inf & (min_pu == 0), 0.0, min_pu * nominal),
                ("name", "snapshot"),
            )
            upper = sc.over(
                xr.where(is_inf & (max_pu == 0), 0.0, max_pu * nominal),
                ("name", "snapshot"),
            )
        var = m.var(f"{c_name}-{attr}")
        loss = _loss_term(m, c_name, fix, losses)
        members = {"name": fix}
        name = f"{c_name}-fix-{attr}-lower"
        bound = _long_of(name, sets, lower)
        body = var[*axes] if loss is None else var[*axes] - loss[*axes]
        m.add_constraints(name, body >= bound[*axes], members=members)
        name = f"{c_name}-fix-{attr}-upper"
        bound = _long_of(name, sets, upper)
        body = var[*axes] if loss is None else var[*axes] + loss[*axes]
        m.add_constraints(name, body <= bound[*axes], members=members)


def define_operational_constraints_for_extendables(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    losses: bool,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Bound an extendable component's operation by per-unit shares of its capacity."""
    T = m.sets["snapshot"]
    for c_name, attr in OPERATIONAL:
        c = n.c[c_name]
        if c.static.empty:
            continue
        ext = c.extendables.intersection(c.active_assets).difference(c.committables)
        if ext.empty:
            continue
        N = m.sets[c_name]
        sets = sc.sets + (N, *pe.sets, T)
        axes = (*sc.sets, N, *pe.sets, T)
        min_pu, max_pu = sc.bounds_pu(c, attr, sns, ext)
        var = m.var(f"{c_name}-{attr}")
        cap = m.var(f"{c_name}-{nominal_attrs[c_name]}")
        loss = _loss_term(m, c_name, ext, losses)
        rows = _domain_of(f"{c_name}-ext-{attr}-rows", sets, _active(c, sns, ext, sc))
        for side, grid in (("lower", min_pu), ("upper", max_pu)):
            name = f"{c_name}-ext-{attr}-{side}"
            share = _sparse_of(name, sets, grid)
            body = var[*axes] - share[*axes] * cap[N]
            if loss is not None:
                body = body - loss[*axes] if side == "lower" else body + loss[*axes]
            relation = body >= 0.0 if side == "lower" else body <= 0.0
            m.add_constraints(name, relation, over=rows, members={"name": ext})


COMMITTABLE = (("Generator", "p"), ("Link", "p"))


def _committable(n: Network, c_name: str) -> Any:
    """Return the component and the members whose commitment this backend states.

    An extendable, modular or maintainable committable states its status
    against a capacity column or a second status, which the refusal names, so
    every member reaching here commits against a capacity it already carries.
    """
    c = n.c[c_name]
    if c.static.empty:
        return c, c.static.index[:0]
    com = c.committables.intersection(c.active_assets)
    return c, com.difference(c.modulars.difference(c.extendables))


def _modular_committable(c: Any, com: pd.Index) -> pd.Index:
    """Return the members of `com` whose status counts modules rather than standing at one."""
    return com.intersection(c.modulars)


def _modular(n: Network, c_name: str) -> Any:
    """Return the component and the members whose capacity comes in modules."""
    c = n.c[c_name]
    if c.static.empty:
        return c, c.static.index[:0]
    held = c.extendables.intersection(c.modulars).intersection(c.active_assets)
    return c, held


LOSSY = ("Line", "Transformer")


def _tangent_segments(transmission_losses: Any) -> int:
    """Return the number of tangent segments a loss setting states, or zero.

    PyPSA takes a whole number as that many tangents and anything else as its
    secant approximation, whose tolerances this backend does not state.
    """
    if isinstance(transmission_losses, dict):
        if transmission_losses.get("mode") != "tangents":
            return 0
        return int(transmission_losses.get("segments", 1))
    if isinstance(transmission_losses, bool) or not isinstance(
        transmission_losses, (int, np.integer)
    ):
        return 0
    return int(transmission_losses)


def _lossy(n: Network, c_name: str) -> Any:
    """Return the component and the branches a loss is stated for."""
    c = n.c[c_name]
    if c.static.empty:
        return c, c.static.index[:0]
    return c, c.active_assets


def define_loss_variables(
    n: Network, m: NimoptModel, sc: Scenarios, pe: Periods
) -> None:
    """Declare the loss each passive branch carries at each snapshot."""
    for c_name in LOSSY:
        c, names = _lossy(n, c_name)
        if names.empty:
            continue
        N, T = m.sets[c_name], m.sets["snapshot"]
        m.add_variables(
            f"{c_name}-loss",
            sc.sets + (N, *pe.sets, T),
            lower=0.0,
            upper=np.inf,
            members={"name": names},
        )


def _loss_ceiling(c: Any, sns: pd.Index, names: Any, sc: Scenarios) -> xr.DataArray:
    """Return the greatest loss a branch can carry, from its own rating."""
    nom = c._operational_attrs["nom"]
    rating = xr.where(
        sc.static(c, nom + "_extendable", names).astype(bool),
        sc.static(c, nom + "_max", names),
        sc.static(c, nom, names),
    )
    if not np.isfinite(rating.to_numpy()).all():
        msg = (
            f"A loss approximation reads a finite maximum rating; {c.name} "
            f"components state an infinite one."
        )
        raise NotImplementedError(msg)
    return sc.over(sc.grid(c, "s_max_pu", sns, names) * rating, ("name", "snapshot"))


def define_loss_constraints(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    segments: int,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Bound each branch's loss above and below the quadratic it approximates.

    A loss is the resistance times the square of the flow, which is not a row
    an LP states. Every tangent of that curve is, and the loss standing above
    all of them and below the curve's greatest value approximates it from
    inside; the flow's sign is covered by taking each tangent both ways.
    """
    for c_name in LOSSY:
        c, names = _lossy(n, c_name)
        if names.empty:
            continue
        N, T = m.sets[c_name], m.sets["snapshot"]
        sets = sc.sets + (N, *pe.sets, T)
        axes = (*sc.sets, N, *pe.sets, T)
        loss = m.var(f"{c_name}-loss")
        flow = m.var(f"{c_name}-s")
        resistance = sc.static(c, "r_pu_eff", names)
        ceiling = _loss_ceiling(c, sns, names, sc)
        held = ("name", "snapshot")

        upper = _long_of(
            f"{c_name}-loss_upper-bound", sets, sc.over(resistance * ceiling**2, held)
        )
        m.add_constraints(
            f"{c_name}-loss_upper",
            loss[*axes] <= upper[*axes],
            members={"name": names},
        )
        for k in range(1, segments + 1):
            at = k / segments * ceiling
            slope = 2 * resistance * at
            offset = resistance * at**2 - slope * at
            for sign in (-1.0, 1.0):
                name = f"{c_name}-loss_tangents-{k}-{int(sign)}"
                share = _sparse_of(f"{name}-slope", sets, sc.over(sign * slope, held))
                bound = _long_of(f"{name}-offset", sets, sc.over(offset, held))
                body = loss[*axes]
                if share.nnz:
                    body = body + share[*axes] * flow[*axes]
                m.add_constraints(name, body >= bound[*axes], members={"name": names})


def define_modular_variables(n: Network, m: NimoptModel) -> None:
    """Declare the whole number of modules a modular capacity is built in."""
    for c_name, _attr in NOMINAL:
        c, mod = _modular(n, c_name)
        if mod.empty:
            continue
        N = m.sets[c_name]
        m.add_variables(
            f"{c_name}-n_mod",
            (N,),
            subset=_named_domain(
                f"{c_name}-n_mod-members", (N,), {N.name: _names(mod)}
            ),
            lower=0.0,
            upper=np.inf,
            integer=True,
            members={"name": mod},
        )


def define_modular_constraints(n: Network, m: NimoptModel, sc: Scenarios) -> None:
    """Tie a modular capacity to a whole number of its module size."""
    for c_name, attr in NOMINAL:
        c, mod = _modular(n, c_name)
        if mod.empty:
            continue
        N = m.sets[c_name]
        sets = sc.sets + (N,)
        axes = (*sc.sets, N)
        name = f"{c_name}-{attr}_modularity"
        one = _long_of(f"{name}-unit", sets, sc.ones(_names(mod)))
        size = _long_of(
            f"{name}-module", sets, sc.static(c, c._operational_attrs["nom_mod"], mod)
        )
        body = (
            one[*axes] * m.var(f"{c_name}-{attr}")[N]
            - size[*axes] * m.var(f"{c_name}-n_mod")[N]
        )
        m.add_constraints(name, body == 0.0, members={"name": mod})


def define_commitment_variables(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    linearized: bool,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Declare the status a committable runs under and its two transitions.

    A status is one where the component runs and zero where it stands, and a
    start-up or a shut-down marks the snapshot the status changes at. All
    three are binary, which is what makes a committed model an integer one.
    """
    for c_name, _attr in COMMITTABLE:
        c, com = _committable(n, c_name)
        if com.empty:
            continue
        N, T = m.sets[c_name], m.sets["snapshot"]
        sets = sc.sets + (N, *pe.sets, T)
        ones = sc.ones(_names(com), sns)
        live = _active(c, sns, com, sc)
        members = _domain_of(f"{c_name}-com-members", sets, live)
        mod = _modular_committable(c, com)
        ceiling = xr.DataArray(
            np.where(np.isin(_names(com), _names(mod)), np.inf, 1.0),
            coords={"name": _names(com)},
            dims=("name",),
        )
        upper = _long_of(
            f"{c_name}-com-ceiling",
            sets,
            sc.over(ceiling * ones, ("name", "snapshot")),
        )
        for kind in ("status", "start_up", "shut_down"):
            m.add_variables(
                f"{c_name}-{kind}",
                sets,
                subset=members,
                lower=0.0,
                upper=upper,
                integer=not linearized,
                members={"name": com},
            )


def _flat_later(sns: pd.Index) -> pd.Index:
    """Return the horizon's snapshots but its first.

    A commitment's minimum times run on across a period boundary, so the one
    row they cannot state is the horizon's own first snapshot rather than each
    period's.
    """
    return sns[1:]


def _before_period(da: xr.DataArray, pe: Periods, step: int) -> Any:
    """Mark the cells whose reference `step` snapshots back stands in an earlier period.

    A commitment reads the snapshot before it whatever period that snapshot
    belongs to, so a reference reaching past a period's first timesteps reads
    the previous period's last ones.
    """
    if not pe:
        return 1.0
    index = da.indexes["snapshot"]
    at = pe.timesteps.get_indexer(index.get_level_values("timestep"))
    place = pe.names.get_indexer(index.get_level_values("period"))
    return xr.DataArray(
        ((at < step) & (place > 0)).astype(float),
        coords=time_coords(index),
        dims=("snapshot",),
    )


def _lagged_terms(
    name: str,
    grid: xr.DataArray,
    variable: Any,
    step: int,
    *,
    sets: tuple,
    sc: Scenarios,
    pe: Periods,
    N: Any,
    T: Any,
) -> list:
    """One term per reference a lag of `step` snapshots in the whole horizon reads.

    The horizon runs on across a period boundary here, so a reference the
    period's own lag does not reach is read at the previous period's last
    timesteps instead.
    """
    held = ("name", "snapshot")
    axes = (*sc.sets, N, *pe.sets, T)
    terms = []
    within = _sparse_of(name, sets, sc.over(grid, held))
    if within.nnz:
        terms.append(within[*axes] * variable[*sc.sets, N, *pe.sets, T - step])
    if pe:
        across = _sparse_of(
            f"{name}-before", sets, sc.over(grid * _before_period(grid, pe, step), held)
        )
        if across.nnz:
            terms.append(
                across[*axes] * variable[*sc.sets, N, *pe.back(1), T.cyclic - step]
            )
    return terms


def _window_terms(
    name: str,
    sets: tuple,
    labels: tuple,
    sc: Scenarios,
    pe: Periods,
    *,
    variable: Any,
    lengths: np.ndarray,
) -> Any:
    """Return the rolling sum of `variable` over each member's window length.

    A member's window reaches back `lengths` snapshots, and a lag reaching
    before the horizon contributes nothing, which is the sum over what the
    window carries. One term per step carries the members that step reaches,
    so the members' windows differ and the sum is still written once.
    """
    N, T = sets
    held, sns = labels
    sets_of = sc.sets + (N, *pe.sets, T)
    total = None
    for step in range(int(lengths.max())):
        reached = lengths > step
        if not reached.any():
            continue
        grid = sc.ones(held[reached], sns)
        for term in _lagged_terms(
            f"{name}-{step}",
            grid,
            variable,
            step,
            sets=sets_of,
            sc=sc,
            pe=pe,
            N=N,
            T=T,
        ):
            total = term if total is None else total + term
    return total


def define_commitment_constraints(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    linearized: bool,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Bind a committable's output, its transitions and its minimum times."""
    for c_name, attr in COMMITTABLE:
        c, com = _committable(n, c_name)
        if com.empty:
            continue
        N, T = m.sets[c_name], m.sets["snapshot"]
        sets = sc.sets + (N, *pe.sets, T)
        axes = (*sc.sets, N, *pe.sets, T)
        labels = (_names(com), sns.to_numpy())
        rows = _domain_of(f"{c_name}-com-rows", sets, _active(c, sns, com, sc))
        var = m.var(f"{c_name}-{attr}")
        status = m.var(f"{c_name}-status")
        start_up = m.var(f"{c_name}-start_up")
        shut_down = m.var(f"{c_name}-shut_down")
        members = {"name": com}
        held_dims = ("name", "snapshot")

        nominal = sc.static(c, c._operational_attrs["nom"], com)
        min_pu, max_pu = sc.bounds_pu(c, attr, sns, com)
        mod = _modular_committable(c, com)
        ext = com.intersection(c.extendables).difference(mod)
        fix = com.difference(ext).difference(mod)
        if not mod.empty:
            _define_modular_commitment(
                m, c, mod, sns, sc=sc, pe=pe, sets=(N, T), attr=attr, status=status
            )
        if not fix.empty:
            fixed_rows = _domain_of(
                f"{c_name}-com-fix-rows", sets, _active(c, sns, fix, sc)
            )
            for side, grid in (
                ("lower", (min_pu * nominal).sel(name=fix)),
                ("upper", (max_pu * nominal).sel(name=fix)),
            ):
                name = f"{c_name}-com-{attr}-{side}"
                share = _sparse_of(f"{name}-share", sets, sc.over(grid, held_dims))
                body = var[*axes] - share[*axes] * status[*axes]
                relation = body >= 0.0 if side == "lower" else body <= 0.0
                m.add_constraints(
                    name, relation, over=fixed_rows, members={"name": fix}
                )
        if not ext.empty:
            _define_big_m_commitment(
                n, m, c, ext, sns, sc=sc, pe=pe, sets=(N, T), attr=attr, status=status
            )

        up_before = _per_component(c, "up_time_before", com, np.float64) > 0
        first = np.zeros((len(com), len(sns)))
        first[up_before, 0] = 1.0
        starts = _over_time(first, com, sns)
        ones = sc.ones(labels[0], sns)
        previous = _lagged_terms(
            f"{c_name}-com-transition-previous",
            ones,
            status,
            1,
            sets=sets,
            sc=sc,
            pe=pe,
            N=N,
            T=T,
        )
        change = status[*axes]
        for term in previous:
            change = change - term
        for kind, transition, sign in (
            ("start-up", start_up, -1.0),
            ("shut-down", shut_down, 1.0),
        ):
            name = f"{c_name}-com-transition-{kind}"
            bound = _long_of(
                f"{name}-bound", sets, sc.over(sign * starts * ones, held_dims)
            )
            m.add_constraints(
                name,
                transition[*axes] + sign * change >= bound[*axes],
                over=rows,
                members=members,
            )

        later = _flat_later(sns)
        for kind, variable, minimum, sense in (
            ("up", start_up, "min_up_time", 0.0),
            ("down", shut_down, "min_down_time", 1.0),
        ):
            lengths = _per_component(c, minimum, com, np.float64).astype(int)
            if not (lengths > 0).any() or len(sns) < 2:
                continue
            keep = lengths > 0
            name = f"{c_name}-com-{kind}-time"
            window = _window_terms(
                name,
                (N, T),
                (labels[0][keep], later),
                sc,
                pe,
                variable=variable,
                lengths=lengths[keep],
            )
            sign = -1.0 if kind == "up" else 1.0
            body = sign * status[*axes] + window
            m.add_constraints(
                name,
                body <= sense,
                over=_domain_of(f"{name}-rows", sets, sc.ones(labels[0][keep], later)),
                members={"name": com[keep], "snapshot": later},
            )

        _define_initial_status(m, c, com, sns, sc=sc, pe=pe, sets=(N, T), status=status)
        if linearized:
            _tighten_relaxation(
                m,
                c,
                com,
                sns,
                sc=sc,
                pe=pe,
                sets=(N, T),
                attr=attr,
                nominal=nominal,
            )


def _define_modular_commitment(
    m: NimoptModel,
    c: Any,
    mod: Any,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    sets: tuple,
    attr: str,
    status: Any,
) -> None:
    """Bind a modular committable's output and its status to the modules it runs.

    A module's capacity is a constant, so the output is bounded by per-unit
    shares of the module size times the number of modules running, and no big
    M is needed. The count of running modules is held below the number built,
    which is the capacity column divided by the module size.
    """
    N, T = sets
    sets_of = sc.sets + (N, *pe.sets, T)
    axes = (*sc.sets, N, *pe.sets, T)
    held = ("name", "snapshot")
    rows = _domain_of(f"{c.name}-com-mod-rows", sets_of, _active(c, sns, mod, sc))
    var = m.var(f"{c.name}-{attr}")
    size = sc.static(c, c._operational_attrs["nom_mod"], mod)
    min_pu, max_pu = sc.bounds_pu(c, attr, sns, mod)
    members = {"name": mod}

    for side, share in (("lower", min_pu * size), ("upper", max_pu * size)):
        name = f"{c.name}-com-mod-{attr}-{side}"
        coefficient = _sparse_of(f"{name}-share", sets_of, sc.over(share, held))
        body = var[*axes]
        if coefficient.nnz:
            body = body - coefficient[*axes] * status[*axes]
        relation = body >= 0.0 if side == "lower" else body <= 0.0
        m.add_constraints(name, relation, over=rows, members=members)

    count = m.var(f"{c.name}-n_mod")
    one = _long_of(f"{c.name}-com-mod-unit", sets_of, sc.ones(_names(mod), sns))
    nominal = c._operational_attrs["nom"]
    for kind in ("status", "start_up", "shut_down"):
        m.add_constraints(
            f"{c.name}-{kind}-{nominal}-variable-upper",
            m.var(f"{c.name}-{kind}")[*axes] - one[*axes] * count[N] <= 0.0,
            over=rows,
            members=members,
        )


def _define_big_m_commitment(
    n: Network,
    m: NimoptModel,
    c: Any,
    ext: Any,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    sets: tuple,
    attr: str,
    status: Any,
) -> None:
    """Bind an extendable committable's output to its status through a big M.

    A capacity that is itself a column cannot multiply a status and stay
    linear, so a bound large enough to hold any output stands in its place:
    a unit that is off is held at zero by that bound, and one that is on is
    held by its capacity instead.
    """
    N, T = sets
    held_dims = ("name", "snapshot")
    sets_of = sc.sets + (N, *pe.sets, T)
    axes = (*sc.sets, N, *pe.sets, T)
    rows = _domain_of(f"{c.name}-com-ext-rows", sets_of, _active(c, sns, ext, sc))
    var = m.var(f"{c.name}-{attr}")
    cap = m.var(f"{c.name}-{c._operational_attrs['nom']}")
    members = {"name": ext}
    min_pu, max_pu = sc.bounds_pu(c, attr, sns, ext)
    raw = c.get_committable_big_m_values(
        names=ext,
        max_pu=c.get_bounds_pu(attr=attr)[1].sel(name=ext),
        committable_big_m=getattr(n, "_committable_big_m", None),
    )
    if "snapshot" not in raw.dims:
        raw = raw.expand_dims(snapshot=sns)
    big_m = sc.over(raw.sel(snapshot=sns) * sc.ones(_names(ext), sns), held_dims)
    if not np.isfinite(big_m.to_numpy()).all():
        msg = (
            f"An extendable committable {c.name} states no finite bound to "
            f"hold its output by; give it a maximum capacity."
        )
        raise NotImplementedError(msg)

    name = f"{c.name}-com-ext-{attr}-lower"
    share = _sparse_of(f"{name}-share", sets_of, min_pu)
    hold = _sparse_of(f"{name}-bound", sets_of, big_m)
    body = var[*axes]
    if share.nnz:
        body = body - share[*axes] * cap[N]
    if hold.nnz:
        body = body - hold[*axes] * status[*axes]
    m.add_constraints(
        name,
        body >= _long_of(f"{name}-rhs", sets_of, -big_m)[*axes],
        over=rows,
        members=members,
    )

    name = f"{c.name}-com-ext-{attr}-upper-bigM"
    hold = _sparse_of(f"{name}-share", sets_of, big_m)
    body = var[*axes]
    if hold.nnz:
        body = body - hold[*axes] * status[*axes]
    m.add_constraints(name, body <= 0.0, over=rows, members=members)

    name = f"{c.name}-com-ext-{attr}-upper-cap"
    share = _sparse_of(f"{name}-share", sets_of, max_pu)
    body = var[*axes]
    if share.nnz:
        body = body - share[*axes] * cap[N]
    m.add_constraints(name, body <= 0.0, over=rows, members=members)

    # the big M holds the output down where the unit is off and states no
    # floor there, so a unit that never runs backwards states its own
    floor = min_pu.to_numpy()
    nonneg = ext[(floor >= 0).all(axis=(0, 2) if sc else 1)]
    if not nonneg.empty:
        m.add_constraints(
            f"{c.name}-com-ext-{attr}-lower-nonneg",
            var[*axes] >= 0.0,
            over=_domain_of(
                f"{c.name}-com-ext-nonneg-rows", sets_of, _active(c, sns, nonneg, sc)
            ),
            members={"name": nonneg},
        )


def _tighten_relaxation(
    m: NimoptModel,
    c: Any,
    com: Any,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    sets: tuple,
    attr: str,
    nominal: Any,
) -> None:
    """Bind a relaxed status to the output the unit could actually reach.

    A binary status admits only a running unit or a still one; a relaxed one
    admits every fraction between, and a fraction states an output no unit
    reaches. These rows hold the relaxed status against what a start-up or a
    shut-down could ramp to, so the relaxation states a bound worth solving.

    They hold where a start-up costs what a shut-down does, because only then
    does the pair of transitions carry one price and the tightening stays
    exact.
    """
    fixed = np.isin(_names(com), _names(com.difference(c.extendables)))
    equal = fixed & (
        _per_component(c, "start_up_cost", com, np.float64)
        == _per_component(c, "shut_down_cost", com, np.float64)
    )
    if not equal.any() or len(sns) < 2:
        return
    N, T = sets
    names = com[equal]
    later = _flat_later(sns)
    sets_of = sc.sets + (N, *pe.sets, T)
    axes = (*sc.sets, N, *pe.sets, T)
    rows = _domain_of(
        f"{c.name}-com-relaxed-rows",
        sets_of,
        _active(c, sns, names, sc).sel(snapshot=later),
    )
    var = m.var(f"{c.name}-{attr}")
    status = m.var(f"{c.name}-status")
    start_up = m.var(f"{c.name}-start_up")
    held = nominal.sel(name=names)
    min_pu, max_pu = sc.bounds_pu(c, attr, sns, names)
    lower_p, upper_p = min_pu * held, max_pu * held

    def share(name: str, grid: Any) -> Any:
        return _sparse_of(
            name, sets_of, sc.over(grid.sel(snapshot=later), ("name", "snapshot"))
        )

    ramp = {
        kind: held * sc.grid(c, f"ramp_limit_{kind}", sns, names)
        for kind in ("start_up", "shut_down", "up", "down")
    }
    ramp = {
        kind: xr.where(np.isfinite(value), value, held) for kind, value in ramp.items()
    }

    families = {
        "p-before": [
            (var, 1, 1.0, None),
            (status, 1, -1.0, ramp["shut_down"]),
            (status, 0, -1.0, upper_p - ramp["shut_down"]),
            (start_up, 0, 1.0, upper_p - ramp["shut_down"]),
        ],
        "p-current": [
            (var, 0, 1.0, None),
            (status, 0, -1.0, upper_p),
            (start_up, 0, 1.0, upper_p - ramp["start_up"]),
        ],
        "partly-start-up": [
            (var, 0, 1.0, None),
            (var, 1, -1.0, None),
            (status, 0, -1.0, lower_p + ramp["up"]),
            (status, 1, 1.0, lower_p),
            (start_up, 0, 1.0, lower_p + ramp["up"] - ramp["start_up"]),
        ],
        "partly-shut-down": [
            (var, 1, 1.0, None),
            (var, 0, -1.0, None),
            (status, 1, -1.0, ramp["shut_down"]),
            (status, 0, 1.0, ramp["shut_down"] - ramp["down"]),
            (start_up, 0, -1.0, lower_p + ramp["down"] - ramp["shut_down"]),
        ],
    }
    for family, terms in families.items():
        name = f"{c.name}-com-{family}"
        body = None
        for k, (variable, lag, sign, grid) in enumerate(terms):
            if lag == 0:
                if grid is None:
                    held_terms = [variable[*axes]]
                else:
                    coefficient = share(f"{name}-{k}", grid)
                    if not coefficient.nnz:
                        continue
                    held_terms = [coefficient[*axes] * variable[*axes]]
            else:
                over = (
                    sc.ones(_names(names), later)
                    if grid is None
                    else grid.sel(snapshot=later)
                )
                held_terms = _lagged_terms(
                    f"{name}-{k}",
                    over,
                    variable,
                    lag,
                    sets=sets_of,
                    sc=sc,
                    pe=pe,
                    N=N,
                    T=T,
                )
            for held_term in held_terms:
                term = sign * held_term
                body = term if body is None else body + term
        if body is None:
            continue
        m.add_constraints(
            name,
            body <= 0.0,
            over=rows,
            members={"name": names, "snapshot": later},
        )


def _define_initial_status(
    m: NimoptModel,
    c: Any,
    com: Any,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    sets: tuple,
    status: Any,
) -> None:
    """Hold a committable at the status its history has not yet released it from.

    A component that ran for fewer snapshots than its minimum up time stays
    up for the remainder, and one that stood for fewer than its minimum down
    time stays down; the rest are free from the first snapshot.
    """
    N, T = sets
    steps = np.arange(1, len(sns) + 1)[None, :]
    for kind, minimum, before, value in (
        ("up", "min_up_time", "up_time_before", 1.0),
        ("down", "min_down_time", "down_time_before", 0.0),
    ):
        held = _per_component(c, before, com, np.float64)
        must = np.clip(_per_component(c, minimum, com, np.float64) - held, 0, None)
        stays = (must[:, None] >= steps) & (held > 0)[:, None]
        if not stays.any():
            continue
        name = f"{c.name}-com-status-min_{kind}_time_must_stay_up"
        marked = xr.DataArray(
            stays.astype(float),
            coords={"name": _names(com), "snapshot": sns},
            dims=("name", "snapshot"),
        )
        rows = _domain_of(
            f"{name}-rows",
            sc.sets + (N, *pe.sets, T),
            sc.ones(_names(com), sns) * marked,
        )
        m.add_constraints(name, status[*sc.sets, N, *pe.sets, T] == value, over=rows)


def define_ramp_limit_constraints(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Bound the change in an operating variable between consecutive snapshots.

    A row stands at every snapshot but the first, where the reference to the
    previous one reaches outside the horizon. A fixed component bounds the
    change by its own capacity and an extendable one by the capacity column.

    A committable component ramps against its commitment: it may reach its
    start-up limit in the snapshot it starts, its shut-down limit in the one
    it stops, and its ordinary limit while it runs. A committable whose
    capacity is also chosen states no ramp row, as PyPSA states none.
    """
    if len(sns) < 2:
        return
    T = m.sets["snapshot"]
    for c_name, attr in RAMPING:
        c = n.c[c_name]
        if c.static.empty:
            continue
        active = c.active_assets
        if active.empty or {"ramp_limit_up", "ramp_limit_down"}.isdisjoint(c.static):
            continue
        committable = active.intersection(c.committables)
        names = active.difference(committable.intersection(c.extendables))
        if names.empty:
            continue
        N = m.sets[c_name]
        var = m.var(f"{c_name}-{attr}")
        ext = c.extendables.intersection(names)
        commits = committable.intersection(names)
        held_dims = ("name", "snapshot")
        after = pe.later
        sets = sc.sets + (N, *pe.sets, T)
        axes = (*sc.sets, N, *pe.sets, T)
        ones = _active(c, sns, names, sc).sel(snapshot=after)
        by_name = {"name": _names(names)}
        extendable = xr.DataArray(
            np.isin(_names(names), _names(ext)).astype(float),
            coords=by_name,
            dims=("name",),
        )
        commits_at = xr.DataArray(
            np.isin(_names(names), _names(commits)).astype(float),
            coords=by_name,
            dims=("name",),
        )
        p_nom = sc.static(c, nominal_attrs[c_name], names)
        for side in ("up", "down"):
            limit = sc.grid(c, f"ramp_limit_{side}", sns, names).sel(snapshot=after)
            over_rows = sc.over(limit * ones, held_dims)
            stated = np.isfinite(over_rows.to_numpy())
            if not stated.any():
                continue
            name = f"{c_name}-{attr}-ramp_limit_{side}"
            sign = 1.0 if side == "up" else -1.0
            body = var[*axes] - var[*sc.sets, N, *pe.sets, T - 1]

            share = _sparse_of(
                f"{name}-share",
                sets,
                sc.over(xr.where(extendable > 0, limit, 0.0) * ones, held_dims),
            )
            if share.nnz:
                cap = m.var(f"{c_name}-{nominal_attrs[c_name]}")
                body = body - sign * share[*axes] * cap[N]

            transition = "start_up" if side == "up" else "shut_down"
            edge = sc.grid(c, f"ramp_limit_{transition}", sns, names).sel(
                snapshot=after
            )
            edge = xr.where(np.isfinite(edge), edge, 1.0)
            running = xr.where(commits_at > 0, (limit - edge) * p_nom, 0.0)
            switching = xr.where(commits_at > 0, edge * p_nom, 0.0)
            # the ordinary limit holds against the status the unit had, and
            # the transition limit against the one it takes; the two swap
            # ends between a rise and a fall
            pairs = (
                ((1, running), (0, switching))
                if side == "up"
                else ((0, running), (1, switching))
            )
            status = None
            for lag, grid in pairs:
                carried = _sparse_of(
                    f"{name}-status-{lag}", sets, sc.over(grid * ones, held_dims)
                )
                if not carried.nnz:
                    continue
                if status is None:
                    status = m.var(f"{c_name}-status")
                reference = (
                    status[*axes]
                    if lag == 0
                    else status[*sc.sets, N, *pe.sets, T - lag]
                )
                body = body - sign * (carried[*axes] * reference)

            free = (extendable > 0) | (commits_at > 0)
            bound = _long_of(
                f"{name}-bound",
                sets,
                sc.over(sign * xr.where(free, 0.0, limit * p_nom) * ones, held_dims),
                keep=stated,
            )
            rows = _domain_of(
                f"{name}-rows",
                sets,
                xr.DataArray(
                    stated.astype(float),
                    coords=over_rows.coords,
                    dims=over_rows.dims,
                ),
            )
            relation = body <= bound[*axes] if side == "up" else body >= bound[*axes]
            reached = stated.any(axis=(0, 2) if sc else 1)
            m.add_constraints(
                name,
                relation,
                over=rows,
                members={"name": names[reached], **pe.later_members},
            )


def define_fixed_operation_constraints(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Fix an operating variable wherever its `_set` attribute states a value."""
    T = m.sets["snapshot"]
    for c_name, attr in [*OPERATIONAL, ("StorageUnit", "p"), ("Store", "p")]:
        c = n.c[c_name]
        attr_set = f"{attr}_set"
        if c.static.empty or c.active_assets.empty or attr_set not in c.dynamic:
            continue
        names = c.active_assets
        fix = sc.grid(c, attr_set, sns, names)
        live = ~np.isnan(fix.to_numpy())
        if not live.any():
            continue
        N = m.sets[c_name]
        sets = sc.sets + (N, *pe.sets, T)
        axes = (*sc.sets, N, *pe.sets, T)
        if c_name == "StorageUnit" and attr == "p":
            name = "StorageUnit-p_set"
            lhs = (
                m.var("StorageUnit-p_dispatch")[*axes]
                - m.var("StorageUnit-p_store")[*axes]
            )
        else:
            name = f"{c_name}-{attr_set}"
            lhs = m.var(f"{c_name}-{attr}")[*axes]
        m.add_constraints(name, lhs == _long_of(name, sets, fix, keep=live)[*axes])


# --- network rows ------------------------------------------------------------


def _incidence(
    name: str,
    B: Any,
    N: Any,
    *,
    at_bus: np.ndarray,
    names: np.ndarray,
    values: np.ndarray,
    sc: Scenarios,
) -> Param:
    """Build a coefficient placing each member's column in the bus row it sits on.

    It carries the scenario dimension so that every term of the balance states
    the axis in the same position, which is the order the rows are stated over.
    """
    if not sc:
        return Param.from_long(
            _symbol(name), (B, N), {B.name: at_bus, N.name: names}, values
        )
    labels = np.asarray(sc.names, dtype=str)
    return Param.from_long(
        _symbol(name),
        sc.sets + (B, N),
        {
            "scenario": np.repeat(labels, len(names)),
            B.name: np.tile(at_bus, len(labels)),
            N.name: np.tile(names, len(labels)),
        },
        np.tile(values, len(labels)),
    )


def _delay_groups(
    delays: dict, suffix: str, names: Any, reaches: np.ndarray, T: Any
) -> list:
    """Each group of links sharing a delay, with the axis its power arrives on.

    Power a link takes in at one snapshot leaves at a later one where the link
    states a delay, so the flow arriving at a bus is read at a lag. A cyclic
    delay wraps at the end of the horizon and states every row; a plain one
    drops the rows whose source reaches before the horizon starts.
    """
    held, cyclic = delays.get(suffix, (0, True))
    steps = np.asarray(pd.Series(held, index=names).fillna(0)).astype(int)
    wraps = np.asarray(pd.Series(cyclic, index=names).fillna(True)).astype(bool)
    groups = []
    for step in np.unique(steps):
        for cycles in (True, False):
            at = reaches & (steps == step) & (wraps == cycles)
            if not at.any():
                continue
            if step == 0:
                groups.append(("", at, T))
                continue
            axis = (T.cyclic - int(step)) if cycles else T - int(step)
            groups.append((f"-delay{int(step)}{'c' if cycles else ''}", at, axis))
    return groups


def _arrivals(
    sets: tuple,
    name: str,
    sc: Scenarios,
    pe: Periods,
    *,
    efficiency: np.ndarray,
    member: np.ndarray,
    reaching: np.ndarray,
    snapshots: pd.Index,
    flow: Any,
    arrival: Any,
) -> list:
    """Return the terms one group of links contributes where its power arrives.

    A link whose efficiency holds across the horizon places one entry per
    link; only one whose efficiency varies needs an entry per snapshot.
    """
    B, K, T = sets
    terms = []
    if not len(member):
        return terms
    values = np.asarray(efficiency, dtype=np.float64)
    if sc:
        steady = np.all(values == values[:1, :, :1], axis=(0, 2))
        first = values[0, :, 0]
    else:
        steady = (values == values[:, :1]).all(axis=1)
        first = values[:, 0]
    live = steady & (first != 0)
    if live.any():
        coefficient = _incidence(
            name,
            B,
            K,
            at_bus=reaching[live],
            names=member[live],
            values=first[live],
            sc=sc,
        )
        terms.append(
            Sum(
                K,
                coefficient[*sc.sets, B, K] * flow[*sc.sets, K, *pe.sets, arrival],
            )
        )
    if (~steady).any():
        varying = np.flatnonzero(~steady)
        held = values[:, varying] if sc else values[varying]
        at = np.nonzero(held)
        columns = {}
        if sc:
            columns["scenario"] = np.asarray(sc.names, dtype=str)[at[0]]
        member_at, snapshot_at = (at[1], at[2]) if sc else (at[0], at[1])
        columns[B.name] = reaching[varying][member_at]
        columns[K.name] = member[varying][member_at]
        columns.update(_at_snapshots(pe, T, snapshots, snapshot_at))
        coefficient = Param.from_long(
            _symbol(f"{name}-varying"),
            sc.sets + (B, K, *pe.sets, T),
            columns,
            held[at],
        )
        terms.append(
            Sum(
                K,
                coefficient[*sc.sets, B, K, *pe.sets, T]
                * flow[*sc.sets, K, *pe.sets, arrival],
            )
        )
    return terms


def define_nodal_balance_constraints(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    meshed_thresholds: Any,
    losses: bool,
    *,
    sc: Scenarios,
    pe: Periods,
) -> None:
    """Power balance at every bus and snapshot, grouped by bus connectivity.

    The balance is one expression over every bus; each connectivity group
    states its rows from it under the name PyPSA gives that group.
    """
    B, T = m.sets["Bus"], m.sets["snapshot"]
    buses = m.labels("Bus")
    bus_labels = _names(buses)
    terms = []
    attached: set = set()

    balance = list(BALANCE_TERMS)
    if losses:
        balance += [
            (c_name, "loss", column, -0.5)
            for c_name in LOSSY
            for column in ("bus0", "bus1")
        ]
    for c_name, attr, column, sign in balance:
        c = n.c[c_name]
        if c.static.empty or c.active_assets.empty:
            continue
        names = c.active_assets
        N = m.sets[c_name]
        values = np.full(len(names), sign, dtype=np.float64)
        if "sign" in c.static:
            values = values * _per_component(c, "sign", names, np.float64)
        at_bus = _bus_of(c, names, column)
        keep = np.array([b != "" and b in buses for b in at_bus], dtype=bool)
        if not keep.any():
            continue
        attached.update(at_bus[keep])
        coefficient = _incidence(
            f"{c_name}-{attr}-{column}",
            B,
            N,
            at_bus=at_bus[keep],
            names=_names(names)[keep],
            values=values[keep],
            sc=sc,
        )
        terms.append(
            Sum(
                N,
                coefficient[*sc.sets, B, N]
                * m.var(f"{c_name}-{attr}")[*sc.sets, N, *pe.sets, T],
            )
        )

    for c_name in MULTIPORT:
        c = n.c[c_name]
        if c.static.empty or c.active_assets.empty:
            continue
        names = c.active_assets
        K = m.sets[c_name]
        flow = m.var(f"{c_name}-p")
        delays = _get_delay_config(c)
        for port in c._output_ports:
            column = f"bus{port}"
            at_bus = _bus_of(c, names, column)
            reaches = np.array([b != "" and b in buses for b in at_bus], dtype=bool)
            if not reaches.any():
                continue
            attached.update(at_bus[reaches])
            grid = np.asarray(
                sc.grid(c, c._port_coefficient_attr(port), sns, names).to_numpy(),
                dtype=np.float64,
            )
            for tag, keep, arrival in _delay_groups(
                delays, c._port_suffix(port), names, reaches, T
            ):
                terms.extend(
                    _arrivals(
                        (B, K, T),
                        f"{c_name}-p-{column}{tag}",
                        sc,
                        pe,
                        efficiency=grid[:, keep] if sc else grid[keep],
                        member=_names(names)[keep],
                        reaching=at_bus[keep],
                        snapshots=sns,
                        flow=flow,
                        arrival=arrival,
                    )
                )

    balance = terms[0]
    for term in terms[1:]:
        balance = balance + term

    leading = (len(sc.names),) if sc else ()
    load = np.zeros((*leading, len(buses), len(sns)))
    loads = n.c.loads
    if not loads.static.empty and not loads.active_assets.empty:
        names = loads.active_assets
        drawn = sc.over(
            -sc.grid(loads, "p_set", sns, names) * sc.static(loads, "sign", names),
            ("name", "snapshot"),
        )
        at = buses.get_indexer(_bus_of(loads, names, "bus"))
        if (at < 0).any():
            msg = "A load sits on a bus the network does not carry."
            raise ValueError(msg)
        np.add.at(load, (slice(None), at) if sc else (at,), drawn.to_numpy())
    rhs = _dense_of("Bus_load", sc.sets + (B, *pe.sets, T), load, pe)

    counts = get_bus_counts(n).reindex(buses, fill_value=0)
    has_terms = np.array([b in attached for b in buses], dtype=bool)
    drawn_at = (load != 0).any(axis=0) if sc else (load != 0)
    if ((~has_terms)[:, None] & drawn_at).any():
        msg = "Empty LHS with non-zero RHS in nodal balance constraint."
        raise ValueError(msg)

    thresholds = (
        sorted(set(meshed_thresholds))
        if meshed_thresholds is not None
        else [30, 100, 400]
    )
    prev: float = 0
    for t in [*thresholds, float("inf")]:
        mask = (counts > prev) if t == float("inf") else (counts > prev) & (counts <= t)
        group = mask.to_numpy() & has_terms
        suffix = f"-meshed-{prev}" if prev > 0 else ""
        if group.any():
            rows = _domain_of(
                f"Bus{suffix}-nodal_balance-rows",
                sc.sets + (B, *pe.sets, T),
                sc.ones(bus_labels[group], sns),
            )
            m.add_constraints(
                f"Bus{suffix}-nodal_balance",
                balance == rhs[*sc.sets, B, *pe.sets, T],
                over=rows,
                members={"name": buses[group]},
            )
        prev = t


def _cycle_law(
    name: str,
    C: Any,
    N: Any,
    *,
    cycles: np.ndarray,
    names: np.ndarray,
    grid: np.ndarray,
    sc: Scenarios,
) -> Param:
    """Build a cycle coefficient, carrying the scenario dimension where one is stated."""
    at = np.nonzero(grid)
    columns: dict = {}
    if sc:
        labels = np.asarray(sc.names, dtype=str)
        held = len(at[0])
        columns["scenario"] = np.repeat(labels, held)
        columns[C.name] = np.tile(cycles[at[0]], len(labels))
        columns[N.name] = np.tile(names[at[1]], len(labels))
        return Param.from_long(
            _symbol(name),
            sc.sets + (C, N),
            columns,
            np.tile(grid[at], len(labels)),
        )
    columns[C.name] = cycles[at[0]]
    columns[N.name] = names[at[1]]
    return Param.from_long(_symbol(name), (C, N), columns, grid[at])


def _require_invariant_impedance(n: Network, weighted: Any, sc: Scenarios) -> None:
    """Refuse a cycle whose weights a scenario would change.

    `cycle_matrix` answers one matrix for the whole network, so an impedance
    differing by scenario would be represented by one scenario's value.
    """
    if not sc:
        return
    for c_name in weighted.index.unique("type"):
        c = n.c[c_name]
        for attr in ("x_pu_eff", "r_pu_eff"):
            if attr not in c.static:
                continue
            values = np.asarray(
                sc.static(c, attr, c.active_assets).to_numpy(), dtype=np.float64
            )
            if not (values == values[:1]).all():
                msg = (
                    f"{c_name}.{attr} varies by scenario; Kirchhoff's voltage "
                    f"law reads one cycle matrix for the whole network."
                )
                raise NotImplementedError(msg)


def define_kirchhoff_voltage_constraints(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Kirchhoff's voltage law on every cycle of every sub-network."""
    n.calculate_dependent_values()
    weighted = n.cycle_matrix(apply_weights=True)
    if weighted.empty:
        return
    _require_invariant_impedance(n, weighted, sc)
    T = m.sets["snapshot"]
    C = m.add_set("cycle", np.arange(weighted.shape[1]), dim="cycle")
    cycles = np.arange(weighted.shape[1])
    terms = []
    for c_name in weighted.index.unique("type"):
        block = weighted.loc[c_name]
        names = block.index.intersection(n.c[c_name].active_assets)
        block = block.loc[names]
        N = m.sets[c_name]
        law = _cycle_law(
            f"kirchhoff-{c_name}",
            C,
            N,
            cycles=cycles,
            names=_names(names),
            grid=block.to_numpy().T * KIRCHHOFF_SCALE,
            sc=sc,
        )
        terms.append(
            Sum(N, law[*sc.sets, C, N] * m.var(f"{c_name}-s")[*sc.sets, N, *pe.sets, T])
        )
    lhs = terms[0]
    for term in terms[1:]:
        lhs = lhs + term

    rhs: Any = 0.0
    if "Transformer" in weighted.index.unique("type"):
        plain = n.cycle_matrix(apply_weights=False).loc["Transformer"]
        tr = n.c.transformers
        names = plain.index.intersection(tr.active_assets)
        _, shifting = _shifting(n)
        shifting = plain.index.intersection(shifting)
        if not shifting.empty:
            TR = m.sets["Transformer"]
            angle = _cycle_law(
                "kirchhoff-phase_shift-varying",
                C,
                TR,
                cycles=cycles,
                names=_names(shifting),
                grid=plain.loc[shifting].to_numpy().T * DEG_TO_RAD * KIRCHHOFF_SCALE,
                sc=sc,
            )
            lhs = lhs + Sum(
                TR,
                angle[*sc.sets, C, TR]
                * m.var("Transformer-phase_shift")[*sc.sets, TR, *pe.sets, T],
            )
        names = names.difference(shifting)
        shift = _per_component(tr, "phase_shift", names, np.float64)
        if (shift != 0).any():
            constant = (
                plain.loc[names].to_numpy().T @ shift * DEG_TO_RAD * KIRCHHOFF_SCALE
            )
            leading = (len(sc.names),) if sc else ()
            grid = np.broadcast_to(
                -constant[:, None], (*leading, len(cycles), len(sns))
            )
            rhs = _dense_of(
                "kirchhoff_phase_shift", sc.sets + (C, *pe.sets, T), grid, pe
            )
    m.add_constraints(
        "Kirchhoff-Voltage-Law", lhs == rhs, over=sc.sets + (C, *pe.sets, T)
    )


# --- temporal rows -----------------------------------------------------------


def _weights(n: Network, sns: pd.Index, kind: str) -> xr.DataArray:
    """Return a snapshot weighting as an array a reader's answer multiplies.

    A reader answers a labelled array, so the weighting carries the snapshot
    dimension by name and the product aligns on it.
    """
    return xr.DataArray(
        np.asarray(n.snapshot_weightings[kind].loc[sns], dtype=np.float64),
        coords=time_coords(sns),
        dims=("snapshot",),
    )


def _hours(n: Network, sns: pd.Index) -> xr.DataArray:
    """Return the hours each snapshot stands for."""
    return _weights(n, sns, "stores")


def _carry_previous(cyclic: np.ndarray, names: pd.Index, sns: pd.Index) -> xr.DataArray:
    """Whether each row reaches back to the level before it.

    A cyclic level reaches back at every snapshot, wrapping at the first; one
    stating an initial value reaches back at every snapshot but the first,
    where the initial value stands on the right-hand side instead.
    """
    held = np.ones((len(names), len(sns)))
    held[~cyclic, 0] = 0.0
    return xr.DataArray(
        held,
        coords={"name": _names(names), "snapshot": sns},
        dims=("name", "snapshot"),
    )


class Carry:
    """How a level reaches the step before it, within a period and across periods.

    `within` is the coefficient on the level one timestep back, wrapping to the
    period's last timestep where a cyclic level states one. `across` pairs a
    whole number of periods with the coefficient on the level at the last
    timestep of that many periods back, which is how a level continues over a
    period boundary. `initial` marks the rows starting from the level the
    component states, because no earlier level reaches them.
    """

    def __init__(self, within: np.ndarray, across: list, initial: np.ndarray) -> None:
        """Hold the two coefficient grids and the rows starting from an initial level."""
        self.within = within
        self.across = across
        self.initial = initial


def _reaches_back(on: np.ndarray, cyclic: np.ndarray) -> np.ndarray:
    """Return the periods back to the previous period each component is active in.

    Zero where none reaches: the component's first active period, whose row
    starts from the initial level instead. A cyclic component wraps, so its
    first active period reads its last -- and a component active in one period
    alone reads that period's own last timestep, which a lag of the whole axis
    states.
    """
    held, periods = on.shape
    gap = np.zeros((held, periods), dtype=int)
    for k in range(held):
        live = np.flatnonzero(on[k])
        for at in live:
            earlier = live[live < at]
            if earlier.size:
                gap[k, at] = at - earlier[-1]
            elif cyclic[k]:
                gap[k, at] = at + periods - live[-1]
    return gap


def _over_time(values: np.ndarray, names: pd.Index, sns: pd.Index) -> xr.DataArray:
    """Label a grid of one value per component and snapshot."""
    return xr.DataArray(
        values, coords={"name": _names(names)}, dims=("name", "snapshot")
    ).assign_coords(time_coords(sns))


def _carry(
    c: Any,
    names: pd.Index,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    cyclic: str,
    cyclic_per_period: str,
    initial_per_period: str,
) -> Carry:
    """Return how each named component's level reaches the level before it.

    A component cycling within a period wraps at the period's edge and one
    starting afresh in each period takes its initial level there; every other
    component continues across the boundary, reading the last timestep of the
    previous period it was active in.
    """
    wraps = _per_component(c, cyclic, names, bool)
    if not pe:
        within = _carry_previous(wraps, names, sns).to_numpy()
        return Carry(within, [], 1.0 - within)

    per_period = _per_component(c, cyclic_per_period, names, bool)
    afresh = _per_component(c, initial_per_period, names, bool)
    apart = per_period | afresh
    on = np.asarray(sc.active(c, sns, names).to_numpy(), dtype=bool)
    if sc:
        if not (on == on[:1]).all():
            msg = (
                f"{c.name} components are active in different investment "
                f"periods by scenario; activity reads a build year and a "
                f"lifetime, which state one value per component."
            )
            raise NotImplementedError(msg)
        on = on[0]
    shape = (len(names), len(pe.names), len(pe.timesteps))
    on = on.reshape(shape).any(axis=2)

    within = np.zeros(shape)
    within[:, :, 1:] = 1.0
    within[per_period, :, 0] = 1.0
    within[~on] = 0.0

    gap = _reaches_back(on, wraps)
    reaching = (gap > 0) & ~apart[:, None] & on
    across = []
    for step in np.unique(gap[reaching]):
        grid = np.zeros(shape)
        grid[(gap == step) & reaching, 0] = 1.0
        across.append((int(step), grid.reshape(len(names), -1)))

    reaches = (within[:, :, 0] > 0) | reaching
    initial = np.zeros(shape)
    initial[:, :, 0] = np.where(on & ~reaches, 1.0, 0.0)
    return Carry(
        within.reshape(len(names), -1), across, initial.reshape(len(names), -1)
    )


def _previous_level(
    name: str,
    level: Any,
    carry: Carry,
    *,
    sets: tuple,
    sc: Scenarios,
    pe: Periods,
    names: pd.Index,
    sns: pd.Index,
    standing: xr.DataArray,
    N: Any,
    T: Any,
) -> Any:
    """Return the terms carrying a level from the step before it, standing loss applied.

    One term reads the previous timestep within the period, and one term per
    distinct gap reads the last timestep of that many periods back, which is
    how a level continues over a period boundary.
    """
    held = ("name", "snapshot")
    axes = (*sc.sets, N, *pe.sets, T)
    coefficient = _sparse_of(
        f"{name}-previous",
        sets,
        sc.over(standing * _over_time(carry.within, names, sns), held),
    )
    body = coefficient[*axes] * level[*sc.sets, N, *pe.sets, T.cyclic - 1]
    for step, grid in carry.across:
        reaching = _sparse_of(
            f"{name}-previous-period-{step}",
            sets,
            sc.over(standing * _over_time(grid, names, sns), held),
        )
        if not reaching.nnz:
            continue
        body = body + reaching[*axes] * level[*sc.sets, N, *pe.back(step), T.cyclic - 1]
    return body


def _warn_initial_ignored(
    c: Any,
    names: pd.Index,
    initial: xr.DataArray,
    pe: Periods,
    *,
    cyclic: str,
    cyclic_per_period: str,
    initial_per_period: str,
) -> None:
    """Warn where a cyclic level overrules the initial level a component states."""
    held = np.asarray(initial.to_numpy()) != 0
    stated = held.any(axis=tuple(range(held.ndim - 1))) if held.ndim > 1 else held
    ignored = _per_component(c, cyclic, names, bool) & stated
    if pe:
        ignored = ignored | (
            _per_component(c, cyclic_per_period, names, bool)
            & _per_component(c, initial_per_period, names, bool)
            & stated
        )
    if ignored.any():
        logger.warning(
            "%s %s: a cyclic level overrules the initial level; %s is ignored.",
            c.name,
            list(names[ignored]),
            initial_per_period.removesuffix("_per_period"),
        )


def define_storage_unit_constraints(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """State of charge carried from each snapshot to the next."""
    c = n.c.storage_units
    if c.static.empty or c.active_assets.empty:
        return
    names = c.active_assets
    U, T = m.sets["StorageUnit"], m.sets["snapshot"]
    sets = sc.sets + (U, *pe.sets, T)
    axes = (*sc.sets, U, *pe.sets, T)
    eh = _hours(n, sns)

    eff_stand = (1 - sc.grid(c, "standing_loss", sns, names)) ** eh
    eff_dispatch = sc.grid(c, "efficiency_dispatch", sns, names)
    eff_store = sc.grid(c, "efficiency_store", sns, names)
    soc_init = sc.static(c, "state_of_charge_initial", names)
    carry = _carry(
        c,
        names,
        sns,
        sc=sc,
        pe=pe,
        cyclic="cyclic_state_of_charge",
        cyclic_per_period="cyclic_state_of_charge_per_period",
        initial_per_period="state_of_charge_initial_per_period",
    )
    _warn_initial_ignored(
        c,
        names,
        soc_init,
        pe,
        cyclic="cyclic_state_of_charge",
        cyclic_per_period="cyclic_state_of_charge_per_period",
        initial_per_period="state_of_charge_initial_per_period",
    )

    soc = m.var("StorageUnit-state_of_charge")
    held = ("name", "snapshot")
    dispatch = _sparse_of("su-dispatch", sets, sc.over(-1.0 / eff_dispatch * eh, held))
    store = _sparse_of("su-store", sets, sc.over(eff_store * eh, held))
    body = (
        -soc[*axes]
        + dispatch[*axes] * m.var("StorageUnit-p_dispatch")[*axes]
        + store[*axes] * m.var("StorageUnit-p_store")[*axes]
        + _previous_level(
            "su",
            soc,
            carry,
            sets=sets,
            sc=sc,
            pe=pe,
            names=names,
            sns=sns,
            standing=eff_stand,
            N=U,
            T=T,
        )
    )
    inflow = sc.grid(c, "inflow", sns, names)
    if "StorageUnit-spill" in m.variables:
        spilled = _sparse_of(
            "su-spill", sets, sc.over(xr.where(inflow > 0, -eh, 0.0), held)
        )
        body = body + spilled[*axes] * m.var("StorageUnit-spill")[*axes]

    rhs = sc.over(-inflow * eh - soc_init * _over_time(carry.initial, names, sns), held)
    m.add_constraints(
        "StorageUnit-energy_balance",
        body == _dense_of("su_inflow", sets, rhs.to_numpy(), pe)[*axes],
        over=_domain_of("su-rows", sets, _active(c, sns, names, sc)),
    )


def define_store_constraints(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Energy carried from each snapshot to the next in a store."""
    c = n.c.stores
    if c.static.empty or c.active_assets.empty:
        return
    names = c.active_assets
    E, T = m.sets["Store"], m.sets["snapshot"]
    sets = sc.sets + (E, *pe.sets, T)
    axes = (*sc.sets, E, *pe.sets, T)
    eh = _hours(n, sns)

    eff_stand = (1 - sc.grid(c, "standing_loss", sns, names)) ** eh
    e_init = sc.static(c, "e_initial", names)
    carry = _carry(
        c,
        names,
        sns,
        sc=sc,
        pe=pe,
        cyclic="e_cyclic",
        cyclic_per_period="e_cyclic_per_period",
        initial_per_period="e_initial_per_period",
    )
    _warn_initial_ignored(
        c,
        names,
        e_init,
        pe,
        cyclic="e_cyclic",
        cyclic_per_period="e_cyclic_per_period",
        initial_per_period="e_initial_per_period",
    )

    e = m.var("Store-e")
    held = ("name", "snapshot")
    ones = sc.ones(_names(names), sns)
    power = _sparse_of("store-power", sets, sc.over(-eh * ones, held))
    body = (
        -e[*axes]
        + power[*axes] * m.var("Store-p")[*axes]
        + _previous_level(
            "store",
            e,
            carry,
            sets=sets,
            sc=sc,
            pe=pe,
            names=names,
            sns=sns,
            standing=eff_stand,
            N=E,
            T=T,
        )
    )

    rhs = sc.over(-e_init * _over_time(carry.initial, names, sns) * ones, held)
    m.add_constraints(
        "Store-energy_balance",
        body == _dense_of("store_initial", sets, rhs.to_numpy(), pe)[*axes],
        over=_domain_of("store-rows", sets, _active(c, sns, names, sc)),
    )


def define_total_supply_constraints(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    component: str = "Generator",
) -> None:
    """Bound each generator's energy over the horizon where a bound is finite."""
    c = n.c[component]
    if c.static.empty or c.active_assets.empty:
        return
    N, T = m.sets[component], m.sets["snapshot"]
    weight = _weights(n, sns, "generators")
    active = c.active_assets
    for side, sense in (("min", ">="), ("max", "<=")):
        bound = sc.static(c, f"e_sum_{side}", active)
        finite = np.isfinite(bound.to_numpy())
        if not finite.any():
            continue
        held = finite.all(axis=0) if sc else finite
        if sc and not np.array_equal(finite.any(axis=0), held):
            msg = (
                f"{component} states e_sum_{side} in some scenarios and not "
                f"others; the limit is stated in every scenario or in none."
            )
            raise NotImplementedError(msg)
        names = active[held]
        name = f"{component}-e_sum_{side}"
        sets = sc.sets + (N, *pe.sets, T)
        energy = _sparse_of(name, sets, sc.ones(_names(names), sns) * weight)
        lhs = Sum(
            *pe.sets,
            T,
            energy[*sc.sets, N, *pe.sets, T]
            * m.var(f"{component}-p")[*sc.sets, N, *pe.sets, T],
        )
        limit = _long_of(
            name + "-bound", sc.sets + (N,), sc.static(c, f"e_sum_{side}", names)
        )
        axes = (*sc.sets, N)
        relation = lhs >= limit[*axes] if sense == ">=" else lhs <= limit[*axes]
        m.add_constraints(name, relation, members={"name": names})


# --- global rows -------------------------------------------------------------


def _carrier_values(n: Network, attr: str) -> pd.Series:
    """Read a carrier attribute, as one value per carrier.

    A carrier states the same rate whatever the scenario, so the attribute is
    read once and indexed by carrier name.
    """
    c = n.c.carriers
    names = c.static.index.unique("name")
    return pd.Series(_per_component(c, attr, names, np.float64), index=names)


def _global_limits(n: Network, kind: str, sc: Scenarios) -> list:
    """Each global constraint of `kind`, with the limit it states in each scenario.

    A limit binds within a scenario rather than in expectation, so a constraint
    appears once under its own name and carries one constant per scenario.
    """
    static = n.c.global_constraints.static
    if static.empty:
        return []
    held = static[static["type"] == kind]
    if held.empty:
        return []
    if not sc:
        return [
            (name, glc, np.array([float(glc.constant)]))
            for name, glc in held.iterrows()
        ]
    out = []
    for name in held.index.unique("name"):
        rows = held.xs(name, level="name")
        for column in ("sense", "carrier_attribute", "type"):
            if column in rows and rows[column].nunique(dropna=False) > 1:
                msg = (
                    f"Global constraint {name!r} states a different {column} in "
                    f"each scenario; only its constant may differ."
                )
                raise NotImplementedError(msg)
        constants = np.asarray(rows.loc[list(sc.names), "constant"], dtype=np.float64)
        out.append((name, rows.iloc[0], constants))
    return out


def _limit(name: str, constants: np.ndarray, sc: Scenarios) -> Any:
    """Build the right-hand side a global limit states, per scenario where one is stated."""
    if not sc:
        return float(constants[0])
    da = xr.DataArray(
        constants,
        coords={"scenario": np.asarray(sc.names, dtype=str)},
        dims=("scenario",),
    )
    return _long_of(name, sc.sets, da)[*sc.sets]


def _per_scenario(values: Any, sc: Scenarios) -> np.ndarray:
    """Reduce a reader's answer over `name`, leaving one value per scenario."""
    held = values.sum("name")
    return (
        np.asarray(held.to_numpy(), dtype=np.float64).reshape(-1)
        if sc
        else np.array([float(held)])
    )


def _carrier_of(c: Any, names: pd.Index) -> pd.Series:
    """Read the carrier each named component belongs to, as one label per component."""
    return pd.Series(_per_component(c, "carrier", names, str), index=names)


def _final_level(
    m: NimoptModel,
    c: Any,
    variable: str,
    *,
    sc: Scenarios,
    pe: Periods,
    grid: xr.DataArray,
) -> Any:
    """Read a storage level at the snapshots `grid` marks, under its coefficients."""
    N, T = m.sets[c.name], m.sets["snapshot"]
    sets = sc.sets + (N, *pe.sets, T)
    coefficient = _sparse_of(
        f"{variable}-final", sets, sc.over(grid, ("name", "snapshot"))
    )
    return Sum(
        N,
        *pe.sets,
        T,
        coefficient[*sc.sets, N, *pe.sets, T]
        * m.var(variable)[*sc.sets, N, *pe.sets, T],
    )


def _level_ends(
    names: pd.Index,
    sns: pd.Index,
    pe: Periods,
    *,
    afresh: np.ndarray,
    years: np.ndarray,
    inside: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Where a level's final value is read, and what weight its initial value carries.

    A level starting afresh in each investment period is read at the last
    snapshot of every period the limit reaches, under that period's year
    weighting; one depleting continuously across the horizon is read once at
    its end.
    """
    grid = np.zeros((len(names), len(sns)))
    weight = np.zeros(len(names))
    reached = [(at, k) for at, k in pe.ends if inside[at]]
    for at, k in reached:
        grid[afresh, at] = years[k]
        weight[afresh] += years[k]
    last = max((at for at, _ in reached), default=len(sns) - 1)
    grid[~afresh, last] = 1.0
    weight[~afresh] = 1.0
    return grid, weight


def _require_continuous_weighting(
    c: Any, names: pd.Index, afresh: np.ndarray, years: np.ndarray
) -> None:
    """Refuse a level depleting across periods weighted by unequal years.

    A level read once at the end states one period's worth of depletion
    whatever the weightings say, so a limit reaching periods weighted
    differently -- `years` carries the weighting of those it reaches -- states
    an inconsistent limit rather than a wrong one.
    """
    if (~afresh).any() and not np.allclose(years, 1.0):
        msg = (
            f"{c.name} components {list(names[~afresh])} deplete continuously "
            f"across investment periods weighted by unequal years; state them "
            f"per period or weight every period by one year."
        )
        raise NotImplementedError(msg)


LEVELS = (
    (
        "storage_units",
        "StorageUnit-state_of_charge",
        "cyclic_state_of_charge",
        "state_of_charge_initial",
        "state_of_charge_initial_per_period",
    ),
    ("stores", "Store-e", "e_cyclic", "e_initial", "e_initial_per_period"),
)


def _level_terms(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
    inside: np.ndarray,
    years: np.ndarray,
    rate_of: Any,
) -> tuple[list, Any]:
    """Return the terms every storage level contributes to a limit, and their constant.

    A level a limit counts enters as its own depletion: the level at the end
    less the level it started from, so the end stands in the rows and the
    start stands in the limit.
    """
    terms = []
    constant: Any = 0.0
    for attribute, variable, cyclic, initial, initial_per_period in LEVELS:
        c = getattr(n.c, attribute)
        if c.static.empty:
            continue
        active = c.active_assets
        rate = rate_of(c, active)
        held = _per_component(c, cyclic, active, bool)
        names = active[(rate != 0).to_numpy() & ~held]
        if names.empty:
            continue
        afresh = (
            _per_component(c, initial_per_period, names, bool)
            if pe
            else np.zeros(len(names), dtype=bool)
        )
        reached = [k for at, k in pe.ends if inside[at]]
        _require_continuous_weighting(c, names, afresh, years[reached])
        ends, weight = _level_ends(
            names, sns, pe, afresh=afresh, years=years, inside=inside
        )
        counted = xr.DataArray(
            np.asarray(rate.loc[names], dtype=np.float64),
            coords={"name": _names(names)},
            dims=("name",),
        )
        terms.append(
            _final_level(
                m,
                c,
                variable,
                sc=sc,
                pe=pe,
                grid=-counted * _over_time(ends, names, sns),
            )
        )
        started = xr.DataArray(weight, coords={"name": _names(names)}, dims=("name",))
        constant = constant - _per_scenario(
            counted * started * sc.static(c, initial, names), sc
        )
    return terms, constant


def define_primary_energy_limit(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Limit a carrier attribute's total, such as emissions."""
    stated = _global_limits(n, "primary_energy", sc)
    if not stated:
        return
    T = m.sets["snapshot"]
    years = pe.per_period(n, "years") if pe else np.ones(1)
    weight = _weights(n, sns, "generators") * pe.weighting(n, "years")
    for name, glc, constants in stated:
        emissions = _carrier_values(n, glc.carrier_attribute)
        emissions = emissions[emissions != 0]
        if emissions.empty:
            continue
        named = glc.get("investment_period")
        if not pe.reaches(named):
            continue
        inside = pe.within(named)
        terms = []
        constant = constants.copy()

        gens = n.c.generators
        if not gens.static.empty:
            active = gens.active_assets
            names = active[_carrier_of(gens, active).isin(emissions.index).to_numpy()]
            if not names.empty:
                G = m.sets["Generator"]
                rate = xr.DataArray(
                    np.asarray(
                        _carrier_of(gens, names).map(emissions), dtype=np.float64
                    ),
                    coords={"name": _names(names)},
                    dims=("name",),
                )
                efficiency = sc.grid(gens, "efficiency", sns, names)
                reached = _over_time(
                    np.broadcast_to(inside, (len(names), len(sns))), names, sns
                )
                grid = sc.over(
                    rate * weight * reached / efficiency, ("name", "snapshot")
                )
                coefficient = _sparse_of(
                    f"{name}-generators", sc.sets + (G, *pe.sets, T), grid
                )
                terms.append(
                    Sum(
                        G,
                        *pe.sets,
                        T,
                        coefficient[*sc.sets, G, *pe.sets, T]
                        * m.var("Generator-p")[*sc.sets, G, *pe.sets, T],
                    )
                )

        held, adjusted = _level_terms(
            n,
            m,
            sns,
            sc=sc,
            pe=pe,
            inside=inside,
            years=years,
            rate_of=lambda c, at, held=emissions: (
                _carrier_of(c, at).map(held).fillna(0.0)
            ),
        )
        terms.extend(held)
        constant = constant + adjusted

        if not terms:
            continue
        lhs = terms[0]
        for term in terms[1:]:
            lhs = lhs + term
        limit = _limit(f"GlobalConstraint-{name}-limit", constant, sc)
        m.add_constraints(f"GlobalConstraint-{name}", _relation(lhs, glc.sense, limit))


def define_operational_limit(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Limit the net production of one carrier over the horizon."""
    stated = _global_limits(n, "operational_limit", sc)
    if not stated:
        return
    T = m.sets["snapshot"]
    years = pe.per_period(n, "years") if pe else np.ones(1)
    weight = _weights(n, sns, "generators") * pe.weighting(n, "years")
    for name, glc, constants in stated:
        named = glc.get("investment_period")
        if not pe.reaches(named):
            continue
        inside = pe.within(named)
        terms = []
        constant = constants.copy()
        gens = n.c.generators
        if not gens.static.empty:
            active = gens.active_assets
            names = active[
                (_carrier_of(gens, active) == glc.carrier_attribute).to_numpy()
            ]
            if not names.empty:
                G = m.sets["Generator"]
                reached = _over_time(
                    np.broadcast_to(inside, (len(names), len(sns))), names, sns
                )
                grid = sc.over(
                    weight * reached * sc.ones(_names(names), sns), ("name", "snapshot")
                )
                coefficient = _sparse_of(
                    f"{name}-generators", sc.sets + (G, *pe.sets, T), grid
                )
                terms.append(
                    Sum(
                        G,
                        *pe.sets,
                        T,
                        coefficient[*sc.sets, G, *pe.sets, T]
                        * m.var("Generator-p")[*sc.sets, G, *pe.sets, T],
                    )
                )
        held, adjusted = _level_terms(
            n,
            m,
            sns,
            sc=sc,
            pe=pe,
            inside=inside,
            years=years,
            rate_of=lambda c, at, held=glc.carrier_attribute: (
                _carrier_of(c, at) == held
            ).astype(float),
        )
        terms.extend(held)
        constant = constant + adjusted
        if not terms:
            continue
        lhs = terms[0]
        for term in terms[1:]:
            lhs = lhs + term
        limit = _limit(f"GlobalConstraint-{name}-limit", constant, sc)
        m.add_constraints(f"GlobalConstraint-{name}", _relation(lhs, glc.sense, limit))


def _capacity_terms(m: NimoptModel, name: str, chosen: list, sc: Scenarios) -> list:
    """One summed capacity term per component, from `(component, members, weights)`.

    Each entry states the extendable members a limit counts and the weight
    each carries, so a limit over a carrier, a length or a capital cost picks
    its own members and the sum is written once.
    """
    terms = []
    for c_name, members, weights in chosen:
        if members.empty or not np.any(weights):
            continue
        N = m.sets[c_name]
        held = xr.DataArray(
            np.asarray(weights, dtype=np.float64),
            coords={"name": _names(members)},
            dims=("name",),
        )
        coefficient = _long_of(
            f"{name}-{c_name}",
            sc.sets + (N,),
            sc.over(held * sc.ones(_names(members)), ("name",)),
        )
        variable = m.var(f"{c_name}-{nominal_attrs[c_name]}")
        terms.append(Sum(N, coefficient[*sc.sets, N] * variable[N]))
    return terms


def _extendable(n: Network, c_name: str) -> Any:
    """Return the active extendable members of a component, or none."""
    c = n.c[c_name]
    if c.static.empty:
        return None, None
    ext = c.extendables.intersection(c.active_assets)
    return (None, None) if ext.empty else (c, ext)


def _carriers(glc: Any) -> list:
    """Return the carriers a global constraint names, as a list."""
    return [held.strip(" []()") for held in str(glc.carrier_attribute).split(",")]


def define_tech_capacity_expansion_limit(
    n: Network, m: NimoptModel, sc: Scenarios
) -> None:
    """Limit the capacity a carrier expands to, over all buses or at one."""
    stated = _global_limits(n, "tech_capacity_expansion_limit", sc)
    if not stated:
        return
    for name, glc, constants in stated:
        at_bus = glc.get("bus") or None
        chosen = []
        for c_name in nominal_attrs:
            c, ext = _extendable(n, c_name)
            if c is None or "carrier" not in c.static:
                continue
            keep = (_carrier_of(c, ext) == glc.carrier_attribute).to_numpy()
            if at_bus is not None:
                bus = "bus0" if c_name in n.branch_components else "bus"
                keep = keep & (_bus_of(c, ext, bus) == str(at_bus))
            members = ext[keep]
            chosen.append((c_name, members, np.ones(len(members))))
        terms = _capacity_terms(m, name, chosen, sc)
        if terms:
            _add_global(
                m,
                name,
                terms,
                glc.sense,
                _limit(f"GlobalConstraint-{name}-limit", constants, sc),
            )


TRANSMISSION_LIMITS = (
    ("transmission_volume_expansion_limit", "length"),
    ("transmission_expansion_cost_limit", "capital_cost"),
)


def define_transmission_expansion_limit(
    n: Network, m: NimoptModel, sc: Scenarios
) -> None:
    """Limit the transmission a network expands, by volume or by its cost."""
    static = n.c.global_constraints.static
    if static.empty:
        return
    for kind, attribute in TRANSMISSION_LIMITS:
        for name, glc, constants in _global_limits(n, kind, sc):
            carriers = _carriers(glc)
            chosen = []
            for c_name in ("Line", "Link"):
                c, ext = _extendable(n, c_name)
                if c is None or "carrier" not in c.static:
                    continue
                members = ext[_carrier_of(c, ext).isin(carriers).to_numpy()]
                chosen.append(
                    (c_name, members, _per_component(c, attribute, members, np.float64))
                )
            terms = _capacity_terms(m, name, chosen, sc)
            if terms:
                _add_global(
                    m,
                    name,
                    terms,
                    glc.sense,
                    _limit(f"GlobalConstraint-{name}-limit", constants, sc),
                )


def _growth_limits(n: Network, sc: Scenarios) -> tuple[pd.Series, pd.Series]:
    """Return the absolute and relative growth each carrier limits its capacity by.

    A scenario states its own limit and the capacity is decided once, so the
    strictest limit binds.
    """
    static = n.c.carriers.static
    if sc:
        absolute = static.groupby(level="name")["max_growth"].min()
        relative = static.groupby(level="name")["max_relative_growth"].min()
    else:
        absolute = static["max_growth"]
        relative = static["max_relative_growth"]
    held = absolute[np.isfinite(absolute)].index
    return absolute.loc[held], relative.loc[held].clip(lower=0)


def define_growth_limit(
    n: Network, m: NimoptModel, sns: pd.Index, sc: Scenarios, pe: Periods
) -> None:
    """Bound the capacity a carrier first builds in each investment period.

    A component counts in the period it first becomes active, and a carrier
    limiting its relative growth counts the period before it too.
    """
    if not pe:
        return
    absolute, relative = _growth_limits(n, sc)
    if absolute.empty:
        return
    carriers = pd.Index(absolute.index, name="Carrier")
    C = m.add_set("Carrier", carriers, dim="Carrier")
    P = pe.sets[0]
    labels = _names(carriers)
    terms = []
    for c_name, attr in NOMINAL:
        c = n.c[c_name]
        if c.static.empty or "carrier" not in c.static:
            continue
        ext = c.extendables.intersection(c.active_assets)
        if ext.empty:
            continue
        of = _carrier_of(c, ext)
        names = ext[of.isin(carriers).to_numpy()]
        if names.empty:
            continue
        N = m.sets[c_name]
        on = np.asarray(sc.active(c, sns, names).to_numpy(), dtype=bool)
        if sc:
            on = on[0]
        on = on.reshape(len(names), len(pe.names), len(pe.timesteps)).any(axis=2)
        first = (on.cumsum(axis=1) == 1) & on
        at = carriers.get_indexer(_carrier_of(c, names))
        grid = np.zeros((len(carriers), len(names), len(pe.names)))
        grid[at, np.arange(len(names))] = first
        share = np.asarray(relative.loc[carriers], dtype=np.float64)
        grid[:, :, 1:] = grid[:, :, 1:] - share[:, None, None] * grid[:, :, :-1]
        held = np.nonzero(grid)
        coefficient = Param.from_long(
            f"Carrier_growth_{_symbol(c_name)}",
            (C, N, P),
            {
                C.name: labels[held[0]],
                N.name: _names(names)[held[1]],
                P.name: np.asarray(pe.names)[held[2]],
            },
            grid[held],
        )
        terms.append(Sum(N, coefficient[C, N, P] * m.var(f"{c_name}-{attr}")[N]))
    if not terms:
        return
    lhs = terms[0]
    for term in terms[1:]:
        lhs = lhs + term
    over = np.asarray(pe.names)
    bound = Param.from_long(
        "Carrier_growth_limit",
        (C, P),
        {
            C.name: np.repeat(labels, len(over)),
            P.name: np.tile(over, len(labels)),
        },
        np.repeat(np.asarray(absolute.loc[carriers], dtype=np.float64), len(over)),
    )
    m.add_constraints("Carrier-growth_limit", lhs <= bound[C, P])


def _add_global(
    m: NimoptModel, name: str, terms: list, sense: str, constant: Any
) -> None:
    """State one global constraint from the terms its limit sums."""
    lhs = terms[0]
    for term in terms[1:]:
        lhs = lhs + term
    m.add_constraints(f"GlobalConstraint-{name}", _relation(lhs, sense, constant))


def _relation(lhs: Any, sense: str, constant: Any) -> Any:
    """Build `lhs sense constant` for the senses a global constraint states."""
    if sense == "<=":
        return lhs <= constant
    if sense == ">=":
        return lhs >= constant
    if sense == "==":
        return lhs == constant
    msg = f"Unknown global constraint sense {sense!r}."
    raise ValueError(msg)


# --- objective ---------------------------------------------------------------


def _active_periods(
    n: Network,
    c: Any,
    sns: pd.Index,
    names: pd.Index,
    *,
    sc: Scenarios,
    pe: Periods,
) -> Any:
    """Return the objective weighting of every period a component is active in.

    A capacity is decided once and stands through each period the asset lives
    in, so its periodized cost is charged once per such period. A horizon
    stating no periods answers one.
    """
    if not pe:
        return 1.0
    active = sc.active(c, sns, names).groupby("period").any("snapshot")
    weights = xr.DataArray(
        pe.per_period(n, "objective"),
        coords={"period": np.asarray(pe.names)},
        dims=("period",),
    )
    return (active * weights).sum("period")


def _sum_terms(terms: list) -> Any:
    """Add the terms together, or answer None where there are none."""
    if not terms:
        return None
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _expected(terms: list, sc: Scenarios) -> list:
    """Each per-scenario term summed under the scenario probabilities."""
    if not sc or not terms:
        return terms
    weight = _long_of("scenario-probability", sc.sets, sc.probability)
    return [Sum(*sc.sets, weight[*sc.sets] * term) for term in terms]


def define_cvar_constraints(
    n: Network, m: NimoptModel, operating: list, sc: Scenarios
) -> Any:
    """Tie the tail average to the operating cost each scenario carries.

    A scenario's excess stands above its operating cost less the
    value-at-risk level, and the tail average stands above that level plus the
    expected excess over the tail's probability mass. Both rows bound the tail
    from below, and the objective prices it, so each holds with equality at the
    optimum.
    """
    alpha = float(n.risk_preference["alpha"])
    excess = m.var("CVaR-a")
    level = m.var("CVaR-theta")
    tail = m.var("CVaR")

    body = excess[*sc.sets] + level
    for term in operating:
        body = body - term
    m.add_constraints("CVaR-excess", body >= 0.0)

    weight = _long_of("cvar-probability", sc.sets, sc.probability)
    m.add_constraints(
        "CVaR-def",
        level
        + (1.0 / (1.0 - alpha)) * Sum(*sc.sets, weight[*sc.sets] * excess[*sc.sets])
        - tail
        <= 0.0,
    )
    return tail


def _options_for(options: Any, c: Any, attr: str) -> list:
    """Return the piecewise options for one component class and attribute."""
    return [o for o in options if o.component == c.name and o.attribute == attr]


def declare_piecewise(
    n: Network,
    m: NimoptModel,
    c: Any,
    *,
    attr: str,
    names: pd.Index,
    sign: str,
    cumulative: bool,
    invert: bool = False,
    status: bool = False,
    timed: bool = True,
    options: Any = (),
    linearized: bool = False,
    sns: pd.Index,
    sc: Scenarios,
    pe: Periods,
) -> pd.Index:
    """Declare the piecewise curves of one PyPSA call and return the names with a curve.

    The auxiliary variable takes PyPSA's name for `attr`, and `assign_solution`
    reads it. Each option group declares one `Model.piecewise` over the names
    without a status, and one over the names with a status that passes the
    status as `active`. A `where` parameter restricts each declaration to its
    names. `timed` declares the curves over the snapshots, and otherwise over
    the components only.
    """
    curved = get_piecewise_names(c, attr, names)
    if curved.empty:
        return curved
    x_points, y_points, valid = _get_breakpoints(c, attr, curved, cumulative, invert)
    aux = c._piecewise_aux_var(attr)
    N = m.sets[c.name]
    B = Set(_symbol(f"{aux}-breakpoint"), np.asarray(x_points.indexes[BREAKPOINT_DIM]))
    xp = breakpoint_param(f"{aux}-x_points", (N, B), x_points, valid)
    yp = breakpoint_param(f"{aux}-y_points", (N, B), y_points, valid)
    if timed:
        frame = sc.sets + (N, *pe.sets, m.sets["snapshot"])
        members = _domain_of(f"{aux}-subset", frame, _active(c, sns, curved, sc))
    else:
        frame = (N,)
        members = _named_domain(f"{aux}-subset", frame, {N.name: _names(curved)})
    m.add_variables(aux, frame, subset=members, members={"name": curved}, **FREE)
    x = m.var(c._piecewise_x_var(attr))[*frame]
    y = m.var(aux)[*frame]
    committed = pd.Index([], name="name")
    if status and c.name in {name for name, _ in COMMITTABLE}:
        committed = _committable(n, c.name)[1]
    owner = f"piecewise {attr!r} of {c.name}"
    for suffix, covered, requested, held in option_groups(curved, options, sign):
        with_status = covered.intersection(committed)
        method = resolve_method(
            requested,
            SIGNS[held],
            has_status=not with_status.empty,
            x_points=x_points.sel(name=covered),
            y_points=y_points.sel(name=covered),
            owner=owner,
        )
        for part, tag in (
            (covered.difference(with_status), ""),
            (with_status, "-status"),
        ):
            if part.empty:
                continue
            name = f"{aux}{suffix}{tag}"
            m.model.piecewise(
                _symbol(name),
                x,
                xp[N, B],
                y,
                yp[N, B],
                SIGNS[held],
                method,
                active=m.var(f"{c.name}-status")[*frame] if tag else None,
                relaxed=bool(linearized) if tag else False,
                where=_named_domain(f"{name}-where", (N,), {N.name: _names(part)}),
            )
    return curved


def _require_no_overnight_cost(c: Any, curved: pd.Index) -> None:
    """Raise ValueError where a component with a capital cost curve has an overnight cost."""
    overnight = c.static["overnight_cost"].loc[curved]
    if overnight.notna().any():
        bad = overnight[overnight.notna()].index.tolist()
        msg = (
            f"Components {bad} of type {c.name} define both a piecewise "
            "'capital_cost' curve and 'overnight_cost'. The piecewise "
            "curve must already be periodized; remove 'overnight_cost'."
        )
        raise ValueError(msg)


def define_objective(
    n: Network,
    m: NimoptModel,
    sns: pd.Index,
    include_objective_constant: bool,
    *,
    sc: Scenarios,
    pe: Periods,
    piecewise_options: Any = (),
    linearized: bool = False,
) -> None:
    """Capital cost of capacity built and operating cost of energy run.

    An operating term is stated per scenario and unweighted, so the objective
    reads it under the probabilities while a risk preference reads the same
    term scenario by scenario. A term stated per period is weighted once by
    the period's objective weighting: an operating term through the snapshot
    it stands at, a capacity through every period the asset is active in.

    PyPSA states no constant for investment already done where the horizon
    carries periods, so this states none either.
    """
    T = m.sets["snapshot"]
    weight = _weights(n, sns, "objective") * pe.weighting(n, "objective")
    terms = []
    operating: list = []

    constant = 0.0
    for c_name, attr in NOMINAL:
        c = n.c[c_name]
        if c.static.empty:
            continue
        ext = c.extendables.intersection(c.active_assets)
        if ext.empty:
            continue
        curved = pd.Index([], name="name")
        if not c._piecewise_schema("capital_cost").empty:
            curved = declare_piecewise(
                n,
                m,
                c,
                attr="capital_cost",
                names=ext,
                sign=">=",
                cumulative=True,
                timed=False,
                options=_options_for(piecewise_options, c, "capital_cost"),
                sns=sns,
                sc=sc,
                pe=pe,
            )
        if not curved.empty:
            _require_no_overnight_cost(c, curved)
            N = m.sets[c_name]
            aux = c._piecewise_aux_var("capital_cost")
            ones = xr.DataArray(
                np.ones(len(curved)), coords={"name": _names(curved)}, dims=("name",)
            )
            active_weight = sc.over(
                ones * _active_periods(n, c, sns, curved, sc=sc, pe=pe), ("name",)
            )
            price = _long_of(
                f"{aux}-weight", sc.sets + (N,), active_weight * sc.probability
            )
            terms.append(Sum(*sc.sets, N, price[*sc.sets, N] * m.var(aux)[N]))
        held = c.periodized_cost.sel(name=ext)
        if held.size == 0:
            continue
        cost = sc.over(held * _active_periods(n, c, sns, ext, sc=sc, pe=pe), ("name",))
        cost = cost * sc.probability
        if not pe:
            constant += float((cost * sc.static(c, attr, ext)).sum())
        live = (cost.to_numpy() != 0) & ~np.isin(_names(ext), _names(curved))
        if live.any():
            N = m.sets[c_name]
            price = _long_of(
                f"{c_name}-{attr}-capital", sc.sets + (N,), cost, keep=live
            )
            terms.append(
                Sum(*sc.sets, N, price[*sc.sets, N] * m.var(f"{c_name}-{attr}")[N])
            )

    if include_objective_constant:
        n._objective_constant = constant
    else:
        n._objective_constant = 0.0
        constant = 0.0

    for cost_type in COST_TYPES:
        for c_name, attr in lookup.query(cost_type).index:
            c = n.c[c_name]
            variable = f"{c_name}-{attr}"
            if c.static.empty or variable not in m.variables:
                continue
            names = c.active_assets
            if c.has_piecewise(cost_type):
                curved = declare_piecewise(
                    n,
                    m,
                    c,
                    attr=cost_type,
                    names=names,
                    sign=">=",
                    cumulative=True,
                    status=True,
                    options=_options_for(piecewise_options, c, cost_type),
                    linearized=linearized,
                    sns=sns,
                    sc=sc,
                    pe=pe,
                )
                if not curved.empty:
                    aux = c._piecewise_aux_var(cost_type)
                    N = m.sets[c_name]
                    grid = sc.over(
                        sc.ones(_names(curved), sns) * weight, ("name", "snapshot")
                    )
                    price = _sparse_of(
                        f"{aux}-weight", sc.sets + (N, *pe.sets, T), grid
                    )
                    operating.append(
                        Sum(
                            N,
                            *pe.sets,
                            T,
                            price[*sc.sets, N, *pe.sets, T]
                            * m.var(aux)[*sc.sets, N, *pe.sets, T],
                        )
                    )
                    names = names.difference(curved)
            cost = sc.over(
                sc.grid(c, cost_type, sns, names) * weight, ("name", "snapshot")
            )
            if not (cost.to_numpy() != 0).any():
                continue
            N = m.sets[c_name]
            price = _sparse_of(
                f"{variable}-{cost_type}", sc.sets + (N, *pe.sets, T), cost
            )
            operating.append(
                Sum(
                    N,
                    *pe.sets,
                    T,
                    price[*sc.sets, N, *pe.sets, T]
                    * m.var(variable)[*sc.sets, N, *pe.sets, T],
                )
            )

    for c_name, _attr in COMMITTABLE:
        c, com = _committable(n, c_name)
        if com.empty:
            continue
        N = m.sets[c_name]
        ones = sc.ones(_names(com), sns)
        for kind, cost in (
            ("start_up", sc.static(c, "start_up_cost", com)),
            ("shut_down", sc.static(c, "shut_down_cost", com)),
            ("status", sc.grid(c, "stand_by_cost", sns, com) * weight),
        ):
            grid = sc.over(cost * ones, ("name", "snapshot"))
            if not (grid.to_numpy() != 0).any():
                continue
            variable = f"{c_name}-{kind}"
            price = _sparse_of(f"{variable}-cost", sc.sets + (N, *pe.sets, T), grid)
            operating.append(
                Sum(
                    N,
                    *pe.sets,
                    T,
                    price[*sc.sets, N, *pe.sets, T]
                    * m.var(variable)[*sc.sets, N, *pe.sets, T],
                )
            )

    for c_name, _attr in lookup.query("marginal_cost_quadratic").index:
        c = n.c[c_name]
        if c.static.empty or "marginal_cost_quadratic" not in c.static:
            continue
        held = sc.grid(c, "marginal_cost_quadratic", sns, c.active_assets)
        if (held.to_numpy() != 0).any():
            msg = "The nimopt backend does not state quadratic marginal costs."
            raise NotImplementedError(msg)

    if not terms and not operating:
        msg = (
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )
        raise ValueError(msg)

    expected = _sum_terms(terms)
    if n.has_risk_preference:
        omega = float(n.risk_preference["omega"])
        tail = define_cvar_constraints(n, m, operating, sc)
        blended = _sum_terms(
            [(1.0 - omega) * held for held in _expected(operating, sc)] + [omega * tail]
        )
        objective = _sum_terms(
            [expected, blended] if expected is not None else [blended]
        )
    else:
        objective = _sum_terms(
            ([expected] if expected is not None else []) + _expected(operating, sc)
        )
    m.add_objective(objective - constant)


# --- the model ---------------------------------------------------------------


def create_model(
    n: Network,
    snapshots: Any = None,
    *,
    multi_investment_periods: bool = False,
    transmission_losses: Any = False,
    linearized_unit_commitment: bool = False,
    include_objective_constant: bool = True,
    meshed_thresholds: Any = None,
    piecewise_options: Any = None,
    **kwargs: Any,
) -> NimoptModel:
    """Build the network's optimisation problem as a nimopt model, stored at `n.model`."""
    if kwargs:
        msg = f"linopy model arguments {sorted(kwargs)} have no nimopt counterpart."
        raise ValueError(msg)
    sns = as_index(n, snapshots, "snapshots")
    pe = Periods.of(sns, multi_investment_periods)
    _refuse_unsupported(n, pe, transmission_losses)
    n._optimize_window = SnapshotWindow(n, sns, sns)
    m = NimoptModel("pypsa")
    n._model = m
    sc = Scenarios.of(n)
    options = list(piecewise_options or [])
    linearized = bool(linearized_unit_commitment)

    define_sets(n, m, sc, pe)
    segments = _tangent_segments(transmission_losses)
    define_variables(n, m, sns, linearized_unit_commitment, sc=sc, pe=pe)
    define_cvar_variables(n, m, sc)
    if segments:
        define_loss_variables(n, m, sc, pe)

    define_nominal_constraints(n, m, sc)
    define_fixed_nominal_constraints(n, m, sc)
    define_operational_constraints_for_non_extendables(
        n, m, sns, bool(segments), sc=sc, pe=pe
    )
    define_operational_constraints_for_extendables(
        n, m, sns, bool(segments), sc=sc, pe=pe
    )
    define_modular_constraints(n, m, sc)
    define_commitment_constraints(n, m, sns, linearized_unit_commitment, sc=sc, pe=pe)
    define_ramp_limit_constraints(n, m, sns, sc, pe)
    define_fixed_operation_constraints(n, m, sns, sc, pe)
    define_nodal_balance_constraints(
        n, m, sns, meshed_thresholds, bool(segments), sc=sc, pe=pe
    )
    if segments:
        define_loss_constraints(n, m, sns, segments, sc=sc, pe=pe)
    define_kirchhoff_voltage_constraints(n, m, sns, sc, pe)
    define_storage_unit_constraints(n, m, sns, sc, pe)
    define_store_constraints(n, m, sns, sc, pe)
    define_total_supply_constraints(n, m, sns, sc=sc, pe=pe)
    define_primary_energy_limit(n, m, sns, sc, pe)
    define_operational_limit(n, m, sns, sc, pe)
    define_tech_capacity_expansion_limit(n, m, sc)
    define_transmission_expansion_limit(n, m, sc)
    define_growth_limit(n, m, sns, sc, pe)
    define_objective(
        n,
        m,
        sns,
        include_objective_constant,
        sc=sc,
        pe=pe,
        piecewise_options=options,
        linearized=linearized,
    )
    return m
