# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The nimopt optimisation backend against linopy, family by family.

The backend restates PyPSA's formulation under the same family names, so a
network optimised either way reaches the same objective and assigns the same
solution. What it does not state it refuses by name, and each refusal is
checked here rather than left to a network that silently omits a row.
"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from linopy.constants import BREAKPOINT_DIM

import pypsa
from pypsa.optimization.piecewise import PiecewiseOptions

no = pytest.importorskip("nimopt")

from pypsa.optimization.nimopt_backend.model import _symbol  # noqa: E402
from pypsa.optimization.nimopt_backend.piecewise import (  # noqa: E402
    breakpoint_param,
    option_groups,
    resolve_method,
)
from pypsa.optimization.nimopt_backend.scenarios import (  # noqa: E402
    Scenarios,
    columns_of,
    labels_of,
)

SOLVER = "highs"


def tiny(snapshots=2, **generator):
    """A one-bus network carrying a single generator and a constant load."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(snapshots))
    n.add("Bus", "b")
    n.add("Generator", "g", bus="b", p_nom=100, marginal_cost=10, **generator)
    n.add("Load", "l", bus="b", p_set=50)
    return n


def solved(n, backend):
    """`n` optimised on `backend`, returned for its solution."""
    n.optimize(backend=backend, solver_name=SOLVER)
    return n


def inner(n):
    """The nimopt model PyPSA's backend builds, under its wrapper."""
    return n.optimize.create_model(backend="nimopt").model


# --- the same problem, either backend ---------------------------------------


@pytest.mark.parametrize("network", ["ac_dc_network", "storage_hvdc_network"])
def test_the_backend_reaches_the_objective_linopy_reaches(network, request):
    a = solved(request.getfixturevalue(network), "linopy")
    b = solved(request.getfixturevalue(network), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-9)


def test_the_dispatch_a_cost_prices_agrees_with_linopy(ac_dc_network):
    # a variable the objective prices is pinned by the optimum; one it does
    # not -- a store's energy level -- is free to differ between two optima
    a = solved(ac_dc_network, "linopy")
    b = solved(pypsa.examples.ac_dc_meshed(), "nimopt")
    p_a = a.c["Generator"].dynamic["p"].to_numpy()
    p_b = b.c["Generator"].dynamic["p"].to_numpy()
    assert np.allclose(p_a, p_b, rtol=1e-6, atol=1e-4)


def test_the_backend_states_the_families_linopy_states(ac_dc_network):
    # linopy carries the objective's constant as a column of its own, which
    # holds no matrix entry; nimopt states the constant as a number
    a = ac_dc_network.optimize.create_model(backend="linopy")
    b = pypsa.examples.ac_dc_meshed().optimize.create_model(backend="nimopt")
    assert set(b.variables) == set(a.variables) - {"objective_constant"}
    assert set(b.constraints) <= set(a.constraints)


def test_a_solved_network_carries_duals_from_either_backend(ac_dc_network):
    b = solved(ac_dc_network, "nimopt")
    prices = b.c["Bus"].dynamic["marginal_price"]
    assert prices.shape == (len(b.snapshots), len(b.c["Bus"].static))
    assert np.isfinite(prices.to_numpy()).all()


# --- the model writes itself and reads back ---------------------------------


def test_the_model_writes_a_file_and_reads_back_the_same_model(ac_dc_network, tmp_path):
    model = inner(ac_dc_network)
    path = tmp_path / "network.yaml"
    no.save(model, path)
    assert path.read_text().splitlines()[-1] == "data: network.npz"

    back = no.load(path)
    assert (back.n_rows, back.n_columns, back.nnz) == (
        model.n_rows,
        model.n_columns,
        model.nnz,
    )
    a, b = model.assemble(), back.assemble()
    for field in ("indptr", "indices", "values", "col_cost", "col_lower", "col_upper"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_every_symbol_is_a_name_an_expression_can_address(ac_dc_network):
    # a set, alias, parameter or variable stands in a relation, so a file
    # addresses it by a Python identifier; a constraint is keyed and never
    # stands in one
    explained = inner(ac_dc_network).explain()
    for held in (*explained.sets, *explained.parameters, *explained.variables):
        assert held.name.isidentifier(), held.name
    for alias, base in explained.aliases:
        assert alias.isidentifier(), alias
        assert base.isidentifier(), base


def test_the_families_keep_the_names_pypsa_reads_them_by(ac_dc_network):
    wrapper = ac_dc_network.optimize.create_model(backend="nimopt")
    assert "Generator-p" in wrapper.variables
    assert any("-" in name for name in wrapper.constraints)


def test_no_domain_the_file_cannot_name_reaches_the_model(ac_dc_network):
    from nimblend import Domain

    model = inner(ac_dc_network)
    nameless = [n for n, v in model.variables.items() if isinstance(v.subset, Domain)]
    nameless += [n for n, c in model.constraints.items() if isinstance(c.over, Domain)]
    assert nameless == []


# --- ramp limits ------------------------------------------------------------


def ramping(**generator):
    """Two generators meeting a load that swings faster than the cheap one can."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(4))
    n.add("Bus", "b")
    n.add("Generator", "slow", bus="b", marginal_cost=10, **generator)
    n.add("Generator", "fast", bus="b", p_nom=100, marginal_cost=50)
    n.add("Load", "l", bus="b", p_set=[20, 90, 30, 80])
    return n


RAMPS = [
    ("fixed", {"p_nom": 100, "ramp_limit_up": 0.3, "ramp_limit_down": 0.3}),
    ("up only", {"p_nom": 100, "ramp_limit_up": 0.3}),
    ("down only", {"p_nom": 100, "ramp_limit_down": 0.3}),
    (
        "extendable",
        {
            "p_nom_extendable": True,
            "capital_cost": 5,
            "ramp_limit_up": 0.3,
            "ramp_limit_down": 0.3,
        },
    ),
]


@pytest.mark.parametrize(("label", "generator"), RAMPS)
def test_a_ramp_limit_reaches_the_optimum_linopy_reaches(label, generator):
    a = solved(ramping(**generator), "linopy")
    b = solved(ramping(**generator), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-9)
    assert np.allclose(
        a.c["Generator"].dynamic["p"].to_numpy(),
        b.c["Generator"].dynamic["p"].to_numpy(),
        atol=1e-6,
    )


def test_a_ramp_limit_binds_the_change_between_snapshots():
    n = solved(ramping(p_nom=100, ramp_limit_up=0.3, ramp_limit_down=0.3), "nimopt")
    slow = n.c["Generator"].dynamic["p"]["slow"].to_numpy()
    assert np.all(np.diff(slow) <= 30 + 1e-6)
    assert np.all(np.diff(slow) >= -30 - 1e-6)


def test_a_time_varying_ramp_limit_is_read_per_snapshot():
    def build():
        n = ramping(p_nom=100)
        n.c.generators.dynamic["ramp_limit_up"] = pd.DataFrame(
            {"slow": [0.9, 0.2, 0.5, 0.9]}, index=n.snapshots
        )
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-9)


COMMITTED_RAMPS = [
    ("ramp both ways", {"ramp_limit_up": 0.5, "ramp_limit_down": 0.5}),
    ("ramp up only", {"ramp_limit_up": 0.5}),
    ("ramp down only", {"ramp_limit_down": 0.5}),
    (
        "ramp with transition limits",
        {
            "ramp_limit_up": 0.5,
            "ramp_limit_down": 0.5,
            "ramp_limit_start_up": 0.6,
            "ramp_limit_shut_down": 0.7,
        },
    ),
]


@pytest.mark.parametrize(("label", "generator"), COMMITTED_RAMPS)
def test_a_committed_unit_ramps_against_its_status(label, generator):
    # a committed unit may reach its start-up limit in the snapshot it starts
    # and its ordinary limit only while it runs, so the row reads the status
    a = solved(committed(**generator), "linopy")
    b = solved(committed(**generator), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_committable_whose_capacity_is_chosen_states_no_ramp_row():
    n = sized_and_committed(ramp_limit_up=0.5, ramp_limit_down=0.5)
    model = inner(n)
    assert not [name for name in model.constraints if "ramp_limit" in name]


def test_a_link_carries_a_ramp_limit_too():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "a")
        n.add("Bus", "b")
        n.add("Generator", "g", bus="a", p_nom=200, marginal_cost=10)
        n.add("Generator", "local", bus="b", p_nom=200, marginal_cost=90)
        n.add(
            "Link",
            "k",
            bus0="a",
            bus1="b",
            p_nom=100,
            ramp_limit_up=0.25,
            ramp_limit_down=0.25,
        )
        n.add("Load", "l", bus="b", p_set=[10, 90, 20, 80])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-9)


def test_a_single_snapshot_states_no_ramp_row():
    n = ramping(p_nom=100, ramp_limit_up=0.3)
    n.set_snapshots(pd.RangeIndex(1))
    model = inner(n)
    assert not [name for name in model.constraints if "ramp_limit" in name]


# --- global constraints -----------------------------------------------------


def expandable():
    """Two carriers and a line, every capacity free to expand."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(3))
    n.add("Bus", "a")
    n.add("Bus", "b")
    for carrier in ("wind", "gas", "AC"):
        n.add("Carrier", carrier)
    n.add(
        "Generator",
        "w",
        bus="a",
        carrier="wind",
        p_nom_extendable=True,
        capital_cost=100,
        marginal_cost=0,
    )
    n.add(
        "Generator",
        "g",
        bus="b",
        carrier="gas",
        p_nom_extendable=True,
        capital_cost=50,
        marginal_cost=40,
    )
    n.add(
        "Line",
        "l",
        bus0="a",
        bus1="b",
        carrier="AC",
        x=0.1,
        s_nom_extendable=True,
        capital_cost=20,
        length=10,
    )
    n.add("Load", "d", bus="b", p_set=[50, 80, 60])
    return n


GLOBAL_LIMITS = [
    ("tech_capacity_expansion_limit", "wind", 30.0),
    ("transmission_volume_expansion_limit", "AC", 200.0),
    ("transmission_expansion_cost_limit", "AC", 500.0),
]


@pytest.mark.parametrize(("kind", "carrier", "constant"), GLOBAL_LIMITS)
def test_a_global_limit_reaches_the_optimum_linopy_reaches(kind, carrier, constant):
    def build():
        n = expandable()
        n.add(
            "GlobalConstraint",
            "limit",
            type=kind,
            carrier_attribute=carrier,
            sense="<=",
            constant=constant,
        )
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-8)
    # the limit binds, so the test would pass for a constraint stated as well
    # as for one silently omitted only if the two agreed by accident
    assert b.objective > solved(expandable(), "nimopt").objective


def test_a_capacity_limit_at_one_bus_counts_only_that_bus():
    def build(bus):
        n = expandable()
        n.add(
            "GlobalConstraint",
            "limit",
            type="tech_capacity_expansion_limit",
            carrier_attribute="gas",
            sense="<=",
            constant=10.0,
            bus=bus,
        )
        return n

    # the gas generator sits at bus b, so a limit at bus a leaves it free
    at_b = solved(build("b"), "nimopt").objective
    at_a = solved(build("a"), "nimopt").objective
    assert at_b > at_a
    assert at_a == pytest.approx(solved(expandable(), "nimopt").objective)
    assert at_b == pytest.approx(solved(build("b"), "linopy").objective, rel=1e-8)


# --- phase shifting transformers --------------------------------------------


def shifting(maximum):
    """A cycle where a phase shift relieves the congested direct line."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(2))
    for bus in ("a", "b", "c"):
        n.add("Bus", bus)
    n.add("Generator", "cheap", bus="a", p_nom=200, marginal_cost=10)
    n.add("Generator", "dear", bus="b", p_nom=200, marginal_cost=100)
    n.add("Line", "ab", bus0="a", bus1="b", x=0.1, s_nom=30)
    n.add("Line", "cb", bus0="c", bus1="b", x=0.1, s_nom=100)
    n.add(
        "Transformer",
        "ac",
        bus0="a",
        bus1="c",
        x=0.1,
        s_nom=100,
        phase_shift_min=-maximum,
        phase_shift_max=maximum,
    )
    n.add("Load", "d", bus="b", p_set=[60, 60])
    return n


@pytest.mark.parametrize("maximum", [0.0, 5.0, 20.0])
def test_an_optimisable_phase_shift_reaches_the_optimum_linopy_reaches(maximum):
    a, b = solved(shifting(maximum), "linopy"), solved(shifting(maximum), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-7)


def test_a_phase_shift_relieves_a_congested_line():
    # a shift the optimisation may choose redistributes the cycle's flow, so
    # the cheap generator reaches the load the congested line cannot carry
    fixed = solved(shifting(0.0), "nimopt").objective
    free = solved(shifting(20.0), "nimopt").objective
    assert free < fixed


def test_the_chosen_phase_shift_is_assigned_back_as_linopy_assigns_it():
    a, b = solved(shifting(20.0), "linopy"), solved(shifting(20.0), "nimopt")
    assert np.allclose(
        a.c["Transformer"].dynamic["phase_shift_opt"].to_numpy(),
        b.c["Transformer"].dynamic["phase_shift_opt"].to_numpy(),
        atol=1e-6,
    )


def test_a_fixed_phase_shift_states_no_column():
    n = shifting(0.0)
    model = inner(n)
    assert "Transformer-phase_shift" not in model.variables


# --- unit commitment --------------------------------------------------------


def committed(**generator):
    """A cheap base unit and a committable peaker the load cycles on and off."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(6))
    n.add("Bus", "b")
    n.add("Generator", "base", bus="b", p_nom=50, marginal_cost=10)
    n.add(
        "Generator",
        "com",
        bus="b",
        p_nom=100,
        marginal_cost=50,
        committable=True,
        p_min_pu=0.4,
        **generator,
    )
    n.add("Load", "l", bus="b", p_set=[45, 95, 45, 95, 45, 45])
    return n


COMMITMENT = [
    ("plain", {}),
    ("start up cost", {"start_up_cost": 400}),
    ("shut down cost", {"shut_down_cost": 400}),
    ("stand by cost", {"stand_by_cost": 20}),
    ("minimum up time", {"min_up_time": 3}),
    ("minimum down time", {"min_down_time": 3}),
    ("both minimum times", {"min_up_time": 2, "min_down_time": 2}),
    ("up time carried in", {"min_up_time": 3, "up_time_before": 1}),
]


@pytest.mark.parametrize(("label", "generator"), COMMITMENT)
def test_a_committed_unit_reaches_the_optimum_linopy_reaches(label, generator):
    a, b = (
        solved(committed(**generator), "linopy"),
        solved(committed(**generator), "nimopt"),
    )
    assert b.objective == pytest.approx(a.objective, rel=1e-7)
    assert np.array_equal(
        a.c["Generator"].dynamic["status"]["com"].to_numpy(),
        b.c["Generator"].dynamic["status"]["com"].to_numpy(),
    )


def test_a_committed_unit_runs_above_its_minimum_or_stands_still():
    n = solved(committed(), "nimopt")
    status = n.c["Generator"].dynamic["status"]["com"].to_numpy()
    output = n.c["Generator"].dynamic["p"]["com"].to_numpy()
    assert np.all(output[status == 0] < 1e-6)
    assert np.all(output[status == 1] >= 40 - 1e-6)


def test_a_minimum_up_time_holds_a_unit_on_and_costs_more():
    plain = solved(committed(), "nimopt")
    held = solved(committed(min_up_time=3), "nimopt")
    assert held.objective > plain.objective
    on = held.c["Generator"].dynamic["status"]["com"].to_numpy()
    # the unit starts once and stays up for its minimum
    assert list(on.astype(int)) == [1, 1, 1, 1, 0, 0]


def test_a_committed_model_carries_integer_columns():
    model = inner(committed())
    assert int(model.integrality().sum()) > 0


def test_a_link_commits_as_a_generator_does():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "a")
        n.add("Bus", "b")
        n.add("Generator", "g", bus="a", p_nom=300, marginal_cost=10)
        n.add("Generator", "local", bus="b", p_nom=300, marginal_cost=90)
        n.add(
            "Link",
            "k",
            bus0="a",
            bus1="b",
            p_nom=100,
            committable=True,
            p_min_pu=0.5,
            start_up_cost=200,
        )
        n.add("Load", "d", bus="b", p_set=[80, 20, 80, 20])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-7)


def test_the_backend_reproduces_pypsas_own_commitment_example():
    # the network and the answer PyPSA's own unit-commitment test states, so
    # the backend is checked against a computed truth and not only linopy
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )
    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=1000,
    )
    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])
    n.optimize(backend="nimopt", solver_name=SOLVER)

    status = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=float).T
    dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T
    assert np.array_equal(n.c.generators.dynamic.status.values, status)
    assert np.allclose(n.c.generators.dynamic.p.values, dispatch)


def test_the_backend_reproduces_pypsas_own_minimum_up_time_example():
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )
    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=0,
        min_up_time=3,
        p_nom=1000,
    )
    n.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])
    n.optimize(backend="nimopt", solver_name=SOLVER)

    status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T
    assert np.array_equal(n.c.generators.dynamic.status.values, status)


def sized_and_committed(**generator):
    """A committable unit whose capacity the optimisation also chooses."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(4))
    n.add("Bus", "b")
    n.add("Generator", "base", bus="b", p_nom=50, marginal_cost=10)
    n.add(
        "Generator",
        "flex",
        bus="b",
        p_nom_extendable=True,
        p_nom_max=120,
        capital_cost=8,
        marginal_cost=50,
        committable=True,
        p_min_pu=0.4,
        **generator,
    )
    n.add("Load", "l", bus="b", p_set=[45, 95, 45, 95])
    return n


BIG_M = [
    ("plain", {}),
    ("a start up cost", {"start_up_cost": 200}),
    ("a minimum up time", {"min_up_time": 2, "start_up_cost": 200}),
]


@pytest.mark.parametrize(("label", "generator"), BIG_M)
def test_an_extendable_committable_reaches_the_optimum_linopy_reaches(label, generator):
    a = solved(sized_and_committed(**generator), "linopy")
    b = solved(sized_and_committed(**generator), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_an_extendable_committable_never_runs_backwards():
    # the big M holds the output down where the unit is off and states no
    # floor there, so the non-negative row is what keeps it at zero
    n = solved(sized_and_committed(start_up_cost=200), "nimopt")
    output = n.c["Generator"].dynamic["p"]["flex"].to_numpy()
    assert np.all(output >= -1e-6)


def test_an_extendable_committable_without_a_stated_maximum_still_builds():
    # PyPSA supplies its own bound where a capacity states no maximum, so the
    # big M is finite either way
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "b")
        n.add("Generator", "base", bus="b", p_nom=50, marginal_cost=10)
        n.add(
            "Generator",
            "flex",
            bus="b",
            p_nom_extendable=True,
            capital_cost=8,
            marginal_cost=50,
            committable=True,
            p_min_pu=0.4,
            start_up_cost=200,
        )
        n.add("Load", "l", bus="b", p_set=[45, 95, 45, 95])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


# --- modular capacity, transmission losses ----------------------------------


def test_a_modular_capacity_is_built_in_whole_modules():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(3))
        n.add("Bus", "b")
        n.add(
            "Generator",
            "g",
            bus="b",
            p_nom_extendable=True,
            p_nom_mod=25.0,
            capital_cost=100,
            marginal_cost=10,
        )
        n.add("Generator", "peak", bus="b", p_nom=200, marginal_cost=500)
        n.add("Load", "l", bus="b", p_set=[60, 60, 60])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-7)
    built = b.c["Generator"].static["p_nom_opt"]["g"]
    # the load asks 60 and the module is 25, so the capacity overshoots to 75
    assert built == pytest.approx(75.0)


def test_a_fixed_capacity_modular_states_no_modularity_row():
    # PyPSA ties a capacity to its module size only where the capacity is
    # itself a column, so a fixed one carries no row and no module count
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(2))
        n.add("Bus", "b")
        n.add("Generator", "g", bus="b", p_nom=100, p_nom_mod=25.0, marginal_cost=10)
        n.add("Load", "l", bus="b", p_set=60)
        return n

    model = inner(build())
    assert not [name for name in model.constraints if "modularity" in name]
    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-7)


def test_a_committable_modular_is_refused():
    # PyPSA states this pair only for an extendable capacity, and the fixed
    # pair it states is infeasible, so the backend refuses rather than differ
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(2))
    n.add("Bus", "b")
    n.add(
        "Generator",
        "g",
        bus="b",
        p_nom=100,
        p_nom_mod=25.0,
        marginal_cost=10,
        committable=True,
    )
    n.add("Load", "l", bus="b", p_set=60)
    with pytest.raises(NotImplementedError, match="committable modular"):
        n.optimize.create_model(backend="nimopt")


def lossy():
    """A resistive line the cheap generator must push its power through."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(3))
    n.add("Bus", "a")
    n.add("Bus", "b")
    n.add("Generator", "cheap", bus="a", p_nom=300, marginal_cost=10)
    n.add("Generator", "local", bus="b", p_nom=300, marginal_cost=60)
    n.add("Line", "l", bus0="a", bus1="b", x=0.1, r=0.05, s_nom=200, s_nom_max=200)
    n.add("Load", "d", bus="b", p_set=[100, 150, 120])
    return n


def tangents(segments):
    return {"mode": "tangents", "segments": segments}


@pytest.mark.parametrize("segments", [1, 2, 4])
def test_tangent_losses_reach_the_optimum_linopy_reaches(segments):
    a, b = lossy(), lossy()
    a.optimize(
        backend="linopy", solver_name=SOLVER, transmission_losses=tangents(segments)
    )
    b.optimize(
        backend="nimopt", solver_name=SOLVER, transmission_losses=tangents(segments)
    )
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_loss_costs_more_than_no_loss_and_tightens_with_segments():
    plain = solved(lossy(), "nimopt").objective
    reached = []
    for segments in (1, 2, 4):
        n = lossy()
        n.optimize(
            backend="nimopt", solver_name=SOLVER, transmission_losses=tangents(segments)
        )
        reached.append(n.objective)
    assert reached[0] > plain
    assert reached[0] < reached[1] < reached[2]


def test_the_secant_loss_approximation_is_refused(ac_dc_network):
    with pytest.raises(NotImplementedError, match="secants"):
        ac_dc_network.optimize.create_model(backend="nimopt", transmission_losses=True)


# --- the linear relaxation of a commitment ----------------------------------


LINEARIZED = [
    ("equal transition costs", {"start_up_cost": 300, "shut_down_cost": 300}),
    ("no transition cost", {}),
    (
        "a minimum up time",
        {"min_up_time": 2, "start_up_cost": 300, "shut_down_cost": 300},
    ),
]


@pytest.mark.parametrize(("label", "generator"), LINEARIZED)
def test_a_relaxed_commitment_reaches_the_optimum_linopy_reaches(label, generator):
    a, b = committed(**generator), committed(**generator)
    a.optimize(backend="linopy", solver_name=SOLVER, linearized_unit_commitment=True)
    b.optimize(backend="nimopt", solver_name=SOLVER, linearized_unit_commitment=True)
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_relaxed_commitment_tightens_only_a_fixed_capacity():
    # the tightening rows hold a relaxed status against what a start-up could
    # ramp to, which reads a capacity that is a number; a capacity that is
    # itself a column is held by its big M instead, so the rows do not apply
    a = sized_and_committed(start_up_cost=300, shut_down_cost=300)
    b = sized_and_committed(start_up_cost=300, shut_down_cost=300)
    a.optimize(backend="linopy", solver_name=SOLVER, linearized_unit_commitment=True)
    b.optimize(backend="nimopt", solver_name=SOLVER, linearized_unit_commitment=True)
    assert b.objective is not None
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_relaxed_commitment_states_no_integer_column():
    n = committed(start_up_cost=300, shut_down_cost=300)
    model = n.optimize.create_model(
        backend="nimopt", linearized_unit_commitment=True
    ).model
    assert int(model.integrality().sum()) == 0


def test_a_relaxation_costs_no_more_than_the_commitment_it_relaxes():
    exact = solved(committed(start_up_cost=300, shut_down_cost=300), "nimopt").objective
    n = committed(start_up_cost=300, shut_down_cost=300)
    n.optimize(backend="nimopt", solver_name=SOLVER, linearized_unit_commitment=True)
    assert n.objective <= exact + 1e-6


# --- links that deliver later than they take in -----------------------------


def delayed(delay, cyclic=True):
    """A link whose power arrives some snapshots after it is taken in."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(5))
    n.add("Bus", "a")
    n.add("Bus", "b")
    n.add("Generator", "g", bus="a", p_nom=200, marginal_cost=10)
    n.add("Generator", "local", bus="b", p_nom=200, marginal_cost=90)
    held = {"delay": delay, "cyclic_delay": cyclic} if delay else {}
    n.add("Link", "k", bus0="a", bus1="b", p_nom=100, efficiency=0.9, **held)
    n.add("Load", "d", bus="b", p_set=[20, 60, 20, 60, 20])
    return n


DELAYS = [(0, True), (1, True), (1, False), (2, True), (2, False)]


@pytest.mark.parametrize(("delay", "cyclic"), DELAYS)
def test_a_delayed_link_reaches_the_optimum_linopy_reaches(delay, cyclic):
    a, b = (
        solved(delayed(delay, cyclic), "linopy"),
        solved(delayed(delay, cyclic), "nimopt"),
    )
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_plain_delay_costs_more_than_one_that_wraps():
    # a cyclic delay states every row, so the horizon's first snapshots are
    # served by power taken in at its end; a plain one leaves them unserved
    # by the link and the dear local unit covers them
    wrapping = solved(delayed(2, cyclic=True), "nimopt").objective
    plain = solved(delayed(2, cyclic=False), "nimopt").objective
    assert plain > wrapping


def test_links_carrying_different_delays_are_grouped():
    def build(varying=False):
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(5))
        for bus in ("a", "b", "c"):
            n.add("Bus", bus)
        n.add("Generator", "g", bus="a", p_nom=300, marginal_cost=10)
        n.add("Generator", "h", bus="c", p_nom=300, marginal_cost=20)
        n.add("Generator", "local", bus="b", p_nom=300, marginal_cost=90)
        n.add("Link", "fast", bus0="a", bus1="b", p_nom=60, efficiency=0.9, delay=0)
        n.add(
            "Link",
            "slow",
            bus0="c",
            bus1="b",
            p_nom=60,
            efficiency=0.95,
            delay=2,
            cyclic_delay=False,
        )
        n.add(
            "Link",
            "wrap",
            bus0="a",
            bus1="b",
            p_nom=40,
            efficiency=0.8,
            delay=1,
            cyclic_delay=True,
        )
        n.add("Load", "d", bus="b", p_set=[20, 60, 20, 60, 20])
        if varying:
            n.c.links.dynamic["efficiency"] = pd.DataFrame(
                {"slow": [0.9, 0.8, 0.95, 0.85, 0.9]}, index=n.snapshots
            )
        return n

    for varying in (False, True):
        a = solved(build(varying), "linopy")
        b = solved(build(varying), "nimopt")
        assert b.objective == pytest.approx(a.objective, rel=1e-6), varying


# --- processes, whose every port carries a rate -----------------------------


def with_process(**process):
    """Three buses and a process converting power between them."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(4))
    for bus in ("a", "b", "c"):
        n.add("Bus", bus)
    n.add("Generator", "g", bus="a", p_nom=300, marginal_cost=10)
    n.add("Generator", "local", bus="b", p_nom=300, marginal_cost=90)
    n.add("Generator", "lc", bus="c", p_nom=300, marginal_cost=80)
    n.add("Load", "d", bus="b", p_set=[30, 50, 40, 60])
    n.add("Load", "e", bus="c", p_set=[10, 20, 10, 20])
    n.add("Process", "p1", bus0="a", p_nom=100, rate0=-1.0, **process)
    return n


PROCESSES = [
    ("two ports", {"bus1": "b", "rate1": 0.8}),
    ("three ports", {"bus1": "b", "bus2": "c", "rate1": 0.6, "rate2": 0.3}),
    (
        "a delayed port",
        {"bus1": "b", "rate1": 0.8, "delay1": 1, "cyclic_delay1": False},
    ),
]


@pytest.mark.parametrize(("label", "process"), PROCESSES)
def test_a_process_reaches_the_optimum_linopy_reaches(label, process):
    a, b = (
        solved(with_process(**process), "linopy"),
        solved(with_process(**process), "nimopt"),
    )
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_process_whose_capacity_is_chosen():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "a")
        n.add("Bus", "b")
        n.add("Generator", "g", bus="a", p_nom=300, marginal_cost=10)
        n.add("Generator", "local", bus="b", p_nom=300, marginal_cost=90)
        n.add(
            "Process",
            "p1",
            bus0="a",
            bus1="b",
            p_nom_extendable=True,
            p_nom_max=200,
            capital_cost=5,
            rate0=-1.0,
            rate1=0.8,
        )
        n.add("Load", "d", bus="b", p_set=[30, 50, 40, 60])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


# --- stores, and limits over the whole horizon -------------------------------


def test_a_store_carries_energy_across_snapshots():
    # one store starts at a level of its own and one closes its cycle, so the
    # balance's right-hand side carries an initial level at some rows and a
    # zero at the rest
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "b")
        n.add("Generator", "g", bus="b", p_nom=80, marginal_cost=[10, 90, 90, 10])
        n.add("Store", "held", bus="b", e_nom=100, e_initial=40, e_cyclic=False)
        n.add("Store", "closed", bus="b", e_nom=50, e_cyclic=True)
        n.add("Load", "l", bus="b", p_set=[50, 50, 50, 50])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


@pytest.mark.parametrize(
    ("limit", "cost", "supplied"),
    [({"e_sum_max": 120.0}, 10, 120.0), ({"e_sum_min": 90.0}, 99, 90.0)],
    ids=["a ceiling over the horizon", "a floor over the horizon"],
)
def test_a_total_supply_limit_binds(limit, cost, supplied):
    # the row sums one generator's dispatch over every snapshot, so it stands
    # over the generator alone; both senses are stated
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(3))
        n.add("Bus", "b")
        n.add("Generator", "g", bus="b", p_nom=100, marginal_cost=cost, **limit)
        n.add("Generator", "other", bus="b", p_nom=100, marginal_cost=50)
        n.add("Load", "l", bus="b", p_set=[50, 50, 50])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)
    # the limit is what pins the total, so it binds at its own value
    assert b.c["Generator"].dynamic["p"]["g"].sum() == pytest.approx(supplied)


# --- families stated together -----------------------------------------------


def test_a_phase_shift_and_a_loss_stand_on_one_transformer():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(2))
        for bus in ("a", "b", "c"):
            n.add("Bus", bus)
        n.add("Generator", "cheap", bus="a", p_nom=200, marginal_cost=10)
        n.add("Generator", "dear", bus="b", p_nom=200, marginal_cost=100)
        n.add("Line", "ab", bus0="a", bus1="b", x=0.1, r=0.02, s_nom=30, s_nom_max=30)
        n.add("Line", "cb", bus0="c", bus1="b", x=0.1, r=0.02, s_nom=100, s_nom_max=100)
        n.add(
            "Transformer",
            "ac",
            bus0="a",
            bus1="c",
            x=0.1,
            r=0.02,
            s_nom=100,
            s_nom_max=100,
            phase_shift_min=-20,
            phase_shift_max=20,
        )
        n.add("Load", "d", bus="b", p_set=[60, 60])
        return n

    a, b = build(), build()
    a.optimize(backend="linopy", solver_name=SOLVER, transmission_losses=tangents(2))
    b.optimize(backend="nimopt", solver_name=SOLVER, transmission_losses=tangents(2))
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_commitment_stands_under_a_global_limit():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "b")
        for carrier in ("gas", "wind"):
            n.add("Carrier", carrier)
        n.add("Generator", "base", bus="b", carrier="wind", p_nom=50, marginal_cost=10)
        n.add(
            "Generator",
            "flex",
            bus="b",
            carrier="gas",
            p_nom=100,
            marginal_cost=50,
            committable=True,
            p_min_pu=0.4,
            start_up_cost=100,
        )
        n.add("Load", "l", bus="b", p_set=[45, 95, 45, 95])
        n.add(
            "GlobalConstraint",
            "lim",
            type="operational_limit",
            carrier_attribute="gas",
            sense="<=",
            constant=120.0,
        )
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_modular_capacity_ramps():
    def build():
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(4))
        n.add("Bus", "b")
        n.add(
            "Generator",
            "g",
            bus="b",
            p_nom_extendable=True,
            p_nom_mod=25.0,
            capital_cost=100,
            marginal_cost=10,
            ramp_limit_up=0.4,
            ramp_limit_down=0.4,
        )
        n.add("Generator", "peak", bus="b", p_nom=200, marginal_cost=500)
        n.add("Load", "l", bus="b", p_set=[30, 60, 30, 60])
        return n

    a, b = solved(build(), "linopy"), solved(build(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


# --- what the backend does not state, it refuses ----------------------------


def test_a_quadratic_marginal_cost_is_refused():
    n = tiny(marginal_cost_quadratic=1.0)
    with pytest.raises(NotImplementedError, match="quadratic"):
        n.optimize.create_model(backend="nimopt")


def test_a_period_horizon_optimised_without_periods_is_refused(ac_dc_periods):
    with pytest.raises(NotImplementedError, match="without them"):
        ac_dc_periods.optimize.create_model(backend="nimopt")


def test_a_horizon_whose_periods_carry_different_timesteps_is_refused(ac_dc_network):
    # the backend states one timestep set shared by every period, so the
    # snapshots are the periods crossed with the timesteps
    n = ac_dc_network
    steps = n.snapshots
    n.snapshots = pd.MultiIndex.from_tuples(
        [(2013, t) for t in steps[:4]] + [(2020, t) for t in steps],
        names=["period", "timestep"],
    )
    n.investment_periods = [2013, 2020]
    with pytest.raises(NotImplementedError, match="crossed with the timesteps"):
        n.optimize.create_model(backend="nimopt", multi_investment_periods=True)


def test_a_stochastic_network_builds(ac_dc_stochastic):
    m = ac_dc_stochastic.optimize.create_model(backend="nimopt")
    assert "scenario" in m.sets
    assert m.variables["Generator-p"].dims == ("scenario", "name", "snapshot")


def test_the_deterministic_model_is_unchanged_by_the_axis(ac_dc_network):
    m = ac_dc_network.optimize.create_model(backend="nimopt")
    assert "scenario" not in m.sets
    for variable in m.variables.values():
        assert "scenario" not in variable.dims
    for constraint in m.constraints.values():
        assert "scenario" not in constraint.dims


def test_an_unknown_backend_is_refused(ac_dc_network):
    with pytest.raises(ValueError, match="Unknown optimisation backend"):
        ac_dc_network.optimize.create_model(backend="glpk")


# --- the investment-period axis ---------------------------------------------


def _periods(n, state=None):
    """Sets, variables and the one family `state` declares, over investment periods."""
    return _one_family(n, state, periods=True)


def solved_periods(n, backend):
    """`n` optimised over investment periods on `backend`."""
    n.optimize(backend=backend, solver_name=SOLVER, multi_investment_periods=True)
    return n


def test_an_empty_axis_states_no_period_set_and_no_label(ac_dc_network):
    from pypsa.optimization.nimopt_backend.periods import Periods

    pe = Periods.of(ac_dc_network.snapshots, False)
    assert pe.sets == ()
    assert pe.labels == ()
    assert not pe
    assert pe.time_dim == "snapshot"


def test_an_axis_reads_its_periods_and_timesteps_from_the_horizon(ac_dc_two_periods):
    from pypsa.optimization.nimopt_backend.periods import Periods

    n = ac_dc_two_periods
    pe = Periods.of(n.snapshots, True)
    assert bool(pe)
    assert list(pe.names) == [2013, 2020]
    assert len(pe.timesteps) == len(n.snapshots) // 2
    assert pe.sets[0].name == "period"
    assert pe.time_dim == "timestep"


def test_a_period_is_a_set_and_the_timesteps_are_its_own(ac_dc_two_periods):
    m = _periods(ac_dc_two_periods)
    assert "period" in m.sets
    assert len(m.sets["snapshot"]) == len(ac_dc_two_periods.snapshots) // 2


def test_a_family_reads_back_over_one_snapshot_dimension(ac_dc_two_periods):
    n = ac_dc_two_periods
    m = n.optimize.create_model(backend="nimopt", multi_investment_periods=True)
    variable = m.variables["Generator-p"]
    assert variable.dims == ("snapshot", "name")
    assert variable.indexes["snapshot"].names == ["period", "timestep"]
    assert list(variable.indexes["snapshot"]) == list(n.snapshots)
    assert m.constraints["Bus-nodal_balance"].dims == ("snapshot", "name")


def test_a_family_reads_back_where_linopy_reads_it(ac_dc_two_periods):
    a = ac_dc_two_periods.optimize.create_model(
        backend="linopy", multi_investment_periods=True
    )
    b = pypsa.examples.ac_dc_meshed()
    b.snapshots = pd.MultiIndex.from_product([[2013, 2020], b.snapshots])
    b.investment_periods = [2013, 2020]
    held = b.optimize.create_model(backend="nimopt", multi_investment_periods=True)
    for name in ("Generator-p", "Line-s", "Generator-p_nom"):
        assert held.variables[name].dims == a.variables[name].dims, name


def test_a_lag_over_the_timesteps_stops_at_a_period_edge(ac_dc_two_periods):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_two_periods
    n.c.generators.static.loc["Manchester Gas", "ramp_limit_up"] = 0.1
    n.c.generators.static.loc["Manchester Gas", "ramp_limit_down"] = 0.1
    m = _periods(
        n,
        lambda n, m, sns, sc, pe: backend.define_ramp_limit_constraints(
            n, m, sns, sc, pe
        ),
    )
    rows = m.constraints["Generator-p-ramp_limit_up"]
    steps = n.snapshots.unique("timestep")
    # one row per period per timestep but the period's first, and no row
    # written to mask the edge away
    assert len(rows.indexes["snapshot"]) == 2 * (len(steps) - 1)
    assert set(rows.indexes["snapshot"].get_level_values("period")) == {2013, 2020}
    assert steps[0] not in set(rows.indexes["snapshot"].get_level_values("timestep"))


def test_an_asset_carries_columns_only_in_the_periods_it_is_active(ac_dc_network):
    n = ac_dc_network
    n.snapshots = pd.MultiIndex.from_product([[2013, 2020, 2030], n.snapshots])
    n.investment_periods = [2013, 2020, 2030]
    n.c.generators.static.loc["Manchester Wind", "build_year"] = 2020
    n.c.generators.static.loc["Manchester Wind", "lifetime"] = 5
    m = _periods(n)
    steps = len(n.snapshots) // 3
    others = len(n.c.generators.static.index.unique("name")) - 1
    # the wind is built in 2020 and retires before 2030, so it carries the
    # columns of one period of the three
    assert m.model.variables["Generator_p"].n_columns == others * 3 * steps + steps
    solution = m.variables["Generator-p"]
    assert solution.dims == ("snapshot", "name")


def test_a_level_carries_across_a_period_boundary(ac_dc_two_periods):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_two_periods
    n.add("Store", "tank", bus="Manchester", e_nom=100, e_initial=40)
    m = _periods(
        n,
        lambda n, m, sns, sc, pe: backend.define_store_constraints(n, m, sns, sc, pe),
    )
    rows = m.constraints["Store-energy_balance"]
    assert rows.dims == ("snapshot", "name")
    # every snapshot states a row, including each period's first
    assert len(rows.indexes["snapshot"]) == len(n.snapshots)
    # a row carries the level, the power and the level before it; the horizon's
    # first row alone reaches back to nothing, so the second period's first
    # timestep reads the first period's last one
    held = m.model.constraints["Store-energy_balance"]
    assert held.nnz == 3 * len(n.snapshots) - 1


def test_the_first_row_of_a_period_reads_the_period_before_it(ac_dc_two_periods):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_two_periods
    n.add("Store", "tank", bus="Manchester", e_nom=100, e_initial=40)
    m = _periods(
        n,
        lambda n, m, sns, sc, pe: backend.define_store_constraints(n, m, sns, sc, pe),
    )
    steps = n.snapshots.unique("timestep").to_numpy()
    row = m.model.row(
        "Store-energy_balance", Store="tank", period=2020, snapshot=steps[0]
    )
    reached = {
        (t.variable, t.coordinate["period"], pd.Timestamp(t.coordinate["snapshot"]))
        for t in row.terms
        if t.variable == "Store_e"
    }
    # the period's first row reads the level at the previous period's last
    # timestep, and no other level
    assert reached == {
        ("Store_e", 2013, pd.Timestamp(steps[-1])),
        ("Store_e", 2020, pd.Timestamp(steps[0])),
    }


def test_a_level_cycling_within_a_period_reaches_back_at_every_row(ac_dc_two_periods):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_two_periods
    n.add("Store", "tank", bus="Manchester", e_nom=100, e_cyclic_per_period=True)
    m = _periods(
        n,
        lambda n, m, sns, sc, pe: backend.define_store_constraints(n, m, sns, sc, pe),
    )
    # a period's first row wraps to that period's last timestep, so no row
    # reaches back to nothing
    held = m.model.constraints["Store-energy_balance"]
    assert held.nnz == 3 * len(n.snapshots)


def test_the_objective_reads_the_period_weightings(ac_dc_two_periods):
    n = ac_dc_two_periods
    n.investment_period_weightings["objective"] = [1.0, 0.5]
    a = solved_periods(n, "linopy")
    b = pypsa.examples.ac_dc_meshed()
    b.snapshots = pd.MultiIndex.from_product([[2013, 2020], b.snapshots])
    b.investment_periods = [2013, 2020]
    b.investment_period_weightings["objective"] = [1.0, 0.5]
    solved_periods(b, "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def _two_period_case(build):
    """The same network optimised over periods either way, for its objectives."""
    a = solved_periods(build(), "linopy")
    b = solved_periods(build(), "nimopt")
    return a, b


def _storage_network(**store):
    n = pypsa.examples.ac_dc_meshed()
    n.snapshots = pd.MultiIndex.from_product([[2013, 2020], n.snapshots])
    n.investment_periods = [2013, 2020]
    n.add("Store", "tank", bus="Frankfurt", e_nom=200, marginal_cost=1.0, **store)
    return n


@pytest.mark.parametrize(
    "store",
    [
        {"e_initial": 80.0},
        {"e_cyclic": True},
        {"e_cyclic_per_period": True},
        {"e_initial": 60.0, "e_initial_per_period": True},
    ],
)
def test_a_level_reaches_the_period_before_it_as_linopy_reads_it(store):
    a, b = _two_period_case(lambda: _storage_network(**store))
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_transition_row_reads_the_status_at_the_period_before_it(
    committable_periods,
):
    from pypsa.optimization.nimopt_backend import build as backend

    n = committable_periods
    m = _periods(
        n,
        lambda n, m, sns, sc, pe: backend.define_commitment_constraints(
            n, m, sns, False, sc=sc, pe=pe
        ),
    )
    row = m.model.row(
        "Generator-com-transition-start-up", Generator="cheap", period=2030, snapshot=0
    )
    reached = {
        (t.variable, t.coordinate["period"], t.coordinate["snapshot"])
        for t in row.terms
        if t.variable == "Generator_status"
    }
    # a commitment runs on across a period's edge, so the first row of the
    # second period reads the status at the last snapshot of the first
    assert reached == {("Generator_status", 2020, 3), ("Generator_status", 2030, 0)}


def test_a_committable_reads_the_snapshot_before_it_across_a_period(
    committable_periods, request
):
    a = solved_periods(committable_periods, "linopy")
    b = solved_periods(request.getfixturevalue("committable_periods"), "nimopt")
    assert a.objective is not None
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_growth_limit_binds_and_agrees_with_linopy():
    def build(**carrier):
        n = pypsa.Network()
        n.set_snapshots(pd.RangeIndex(3))
        n.add("Bus", "b")
        n.add("Carrier", "wind", **carrier)
        n.add("Carrier", "gas")
        for name, year in (("w1", 2020), ("w2", 2030)):
            n.add(
                "Generator",
                name,
                bus="b",
                carrier="wind",
                p_nom_extendable=True,
                capital_cost=100,
                build_year=year,
                lifetime=40,
            )
        n.add("Generator", "peak", bus="b", carrier="gas", p_nom=500, marginal_cost=300)
        n.add("Load", "l", bus="b", p_set=[100, 150, 120])
        n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots])
        n.investment_periods = [2020, 2030]
        return n

    free = solved_periods(build(), "nimopt")
    held = solved_periods(build(max_growth=60.0), "nimopt")
    # the limit costs something: the fallback runs where the carrier cannot grow
    assert held.objective > free.objective
    a = solved_periods(build(max_growth=60.0), "linopy")
    assert held.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_global_limit_bound_to_a_period_on_a_flat_horizon_is_refused(ac_dc_network):
    n = ac_dc_network
    n.c.global_constraints.static.loc["co2_limit", "investment_period"] = 2013.0
    with pytest.raises(NotImplementedError, match="carrying none"):
        n.optimize.create_model(backend="nimopt")


def test_a_global_limit_naming_a_period_outside_the_horizon_states_no_row(
    ac_dc_two_periods,
):
    # PyPSA states no row for it either, so the objective is the one the
    # horizon's own limit reaches
    n = ac_dc_two_periods
    n.add(
        "GlobalConstraint",
        "co2_2040",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=1.0,
        investment_period=2040,
    )
    m = n.optimize.create_model(backend="nimopt", multi_investment_periods=True)
    assert "GlobalConstraint-co2_2040" not in m.constraints


def test_a_global_limit_bound_to_one_period_binds_and_agrees_with_linopy():
    def build(constant=None):
        n = pypsa.examples.ac_dc_meshed()
        n.snapshots = pd.MultiIndex.from_product([[2013, 2020], n.snapshots])
        n.investment_periods = [2013, 2020]
        n.investment_period_weightings["years"] = [7.0, 10.0]
        n.c.global_constraints.static.loc["co2_limit", "constant"] = 6000.0
        if constant is not None:
            n.add(
                "GlobalConstraint",
                "co2_2020",
                type="primary_energy",
                carrier_attribute="co2_emissions",
                sense="<=",
                constant=constant,
                investment_period=2020,
            )
        return n

    free = solved_periods(build(), "nimopt")
    held = solved_periods(build(3000.0), "nimopt")
    assert held.objective > free.objective
    a = solved_periods(build(3000.0), "linopy")
    assert held.objective == pytest.approx(a.objective, rel=1e-6)


# --- the scenario axis -------------------------------------------------------


def test_an_empty_axis_states_no_set_and_no_label():
    sc = Scenarios.empty()
    assert sc.sets == ()
    assert sc.labels == ()
    assert sc.weights is None
    assert not sc


def test_an_axis_reads_its_names_and_weights_from_the_network(ac_dc_stochastic):
    sc = Scenarios.of(ac_dc_stochastic)
    assert bool(sc)
    assert list(sc.labels[0]) == ["low", "high"]
    assert sc.sets[0].name == "scenario"
    assert np.allclose(sc.weights, [0.3, 0.7])


def test_a_reader_answers_the_scenario_dimension(ac_dc_stochastic):
    n = ac_dc_stochastic
    sc = Scenarios.of(n)
    c = n.c.generators
    names = c.active_assets
    assert sc.static(c, "p_nom_min", names).dims == ("scenario", "name")
    assert sc.grid(c, "marginal_cost", n.snapshots, names).dims == (
        "scenario",
        "name",
        "snapshot",
    )
    lower, upper = sc.bounds_pu(c, "p", n.snapshots, names)
    # get_bounds_pu answers its two arrays in different orders; a reader fixes both
    assert lower.dims == ("scenario", "name", "snapshot")
    assert upper.dims == ("scenario", "name", "snapshot")


def test_a_reader_drops_the_dimension_for_a_deterministic_network(ac_dc_network):
    n = ac_dc_network
    sc = Scenarios.empty()
    c = n.c.generators
    names = c.active_assets
    assert sc.static(c, "p_nom_min", names).dims == ("name",)
    assert sc.grid(c, "marginal_cost", n.snapshots, names).dims == ("name", "snapshot")


def test_an_empty_axis_refuses_an_array_carrying_scenarios(ac_dc_stochastic):
    # an empty axis meeting scenario data states no answer, rather than
    # silently keeping one scenario of it
    c = ac_dc_stochastic.c.generators
    with pytest.raises(ValueError, match="disagree"):
        Scenarios.empty().static(c, "p_nom_min", c.active_assets)


def test_columns_are_derived_from_the_array_and_not_from_a_position():
    # An array whose axes are ordered against the canonical order still labels
    # every entry correctly, because each axis is named by the array itself.
    da = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        coords={"snapshot": [0, 1, 2], "name": ["a", "b"]},
        dims=("snapshot", "name"),
    )
    columns, values = columns_of(da, np.ones(da.shape, dtype=bool))
    assert set(columns) == {"snapshot", "name"}
    order = np.lexsort((columns["name"], columns["snapshot"]))
    assert list(values[order]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(columns["name"][order]) == ["a", "b", "a", "b", "a", "b"]
    assert list(columns["snapshot"][order]) == [0, 0, 1, 1, 2, 2]


def test_object_labels_are_read_as_strings():
    da = xr.DataArray(
        np.ones(2), coords={"name": np.array(["a", "b"], dtype=object)}, dims=("name",)
    )
    assert labels_of(da, "name").dtype.kind == "U"


def test_the_probability_carries_the_scenario_dimension_by_name(ac_dc_stochastic):
    # a coefficient multiplies it, so it must align on the dimension rather
    # than on a position
    sc = Scenarios.of(ac_dc_stochastic)
    assert sc.probability.dims == ("scenario",)
    assert list(sc.probability.indexes["scenario"]) == ["low", "high"]
    assert Scenarios.empty().probability == 1.0


def test_a_parameter_from_a_labelled_array_ignores_axis_order():
    from pypsa.optimization.nimopt_backend import build as backend

    S = no.Set("scenario", np.array(["low", "high"]))
    N = no.Set("Generator", np.array(["g0", "g1"]))
    # a reader names the component axis `name`; the array is ordered against
    # the sets and the parameter must still be right
    da = xr.DataArray(
        np.array([[1.0, 3.0], [2.0, 4.0]]),
        coords={"name": ["g0", "g1"], "scenario": ["low", "high"]},
        dims=("name", "scenario"),
    )
    param = backend._long_of("bound", (S, N), da)
    dense = param.materialise().to_dense()
    assert dense.shape == (2, 2)
    # scenario is the parameter's first dimension, Generator its second
    assert dense[0, 0] == 1.0
    assert dense[0, 1] == 2.0
    assert dense[1, 0] == 3.0
    assert dense[1, 1] == 4.0


def test_a_set_named_for_a_component_reads_the_name_dimension():
    # a reader names the component axis `name` while the set is named for the
    # component, so the two are paired by dimension name rather than position
    from pypsa.optimization.nimopt_backend import build as backend

    TR = no.Set("Transformer", np.array(["t0", "t1"]))
    da = xr.DataArray(
        np.array([3.0, 5.0]), coords={"name": ["t0", "t1"]}, dims=("name",)
    )
    dense = backend._long_of("shift", (TR,), da).materialise().to_dense()
    assert list(dense) == [3.0, 5.0]


def _one_family(n, state=None, periods=False):
    """Sets, variables and the one family `state` declares.

    A network whose remaining rows are not yet stated under scenarios or
    investment periods cannot be built whole, so a family is stated on its own
    to check the rows it states and the dimensions they carry.
    """
    from pypsa.common import as_index
    from pypsa.optimization.nimopt_backend import build as backend
    from pypsa.optimization.nimopt_backend.model import NimoptModel
    from pypsa.optimization.nimopt_backend.periods import Periods

    sns = as_index(n, None, "snapshots")
    m = NimoptModel("pypsa")
    sc = Scenarios.of(n)
    pe = Periods.of(sns, periods)
    n._multi_invest = int(bool(pe))
    backend.define_sets(n, m, sc, pe)
    backend.define_variables(n, m, sns, False, sc=sc, pe=pe)
    if state is not None:
        state(n, m, sns, sc, pe)
    return m


def _variables_only(n):
    """The sets and variables alone."""
    return _one_family(n)


def test_capacity_is_first_stage_and_operation_is_second_stage(ac_dc_stochastic):
    n = ac_dc_stochastic
    m = _variables_only(n)
    assert m.variables["Generator-p_nom"].dims == ("name",)
    assert m.variables["Line-s_nom"].dims == ("name",)
    assert m.variables["Generator-p"].dims == ("scenario", "name", "snapshot")
    assert m.variables["Line-s"].dims == ("scenario", "name", "snapshot")


def test_a_capacity_bound_is_stated_per_scenario(ac_dc_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    # a lower bound that differs by scenario: the capacity must carry the larger
    n.c.generators.static.loc[("low", "Manchester Wind"), "p_nom_min"] = 50.0
    n.c.generators.static.loc[("high", "Manchester Wind"), "p_nom_min"] = 200.0
    m = _one_family(
        n, lambda n, m, sns, sc, pe: backend.define_nominal_constraints(n, m, sc)
    )
    assert set(m.constraints["Generator-ext-p_nom-lower"].dims) == {"scenario", "name"}


def test_a_capacity_bounded_above_in_one_scenario_only(ac_dc_stochastic):
    # the upper row is stated where the bound is finite and nowhere else, so
    # the mask reducing over the axis states rows for one scenario and not the
    # other; the reduction is checked here rather than assumed
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    n.c.generators.static.loc[("low", "Manchester Wind"), "p_nom_max"] = 300.0
    n.c.generators.static.loc[("high", "Manchester Wind"), "p_nom_max"] = np.inf
    m = _one_family(
        n, lambda n, m, sns, sc, pe: backend.define_nominal_constraints(n, m, sc)
    )
    assert "scenario" in m.constraints["Generator-ext-p_nom-upper"].dims


def test_a_fixed_capacity_and_a_modular_capacity_are_stated_per_scenario(
    ac_dc_stochastic,
):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    n.c.generators.static.loc[(slice(None), "Manchester Wind"), "p_nom_set"] = 120.0
    n.c.generators.static.loc[(slice(None), "Norway Wind"), "p_nom_mod"] = 25.0
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: (
            backend.define_fixed_nominal_constraints(n, m, sc),
            backend.define_modular_constraints(n, m, sc),
        ),
    )
    assert set(m.constraints["Generator-p_nom_set"].dims) == {"scenario", "name"}
    assert set(m.constraints["Generator-p_nom_modularity"].dims) == {
        "scenario",
        "name",
    }


def test_an_operating_bound_is_stated_per_scenario(ac_dc_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    # a wind availability that differs by scenario bounds dispatch differently
    n.c.generators.dynamic.p_max_pu[("low", "Manchester Wind")] = 0.1
    n.c.generators.dynamic.p_max_pu[("high", "Manchester Wind")] = 0.9
    n.c.generators.static.loc[(slice(None), "Manchester Wind"), "p_nom_extendable"] = (
        False
    )
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: (
            backend.define_operational_constraints_for_non_extendables(
                n, m, sns, False, sc=sc, pe=pe
            ),
            backend.define_operational_constraints_for_extendables(
                n, m, sns, False, sc=sc, pe=pe
            ),
        ),
    )
    assert m.constraints["Generator-fix-p-upper"].dims == (
        "scenario",
        "name",
        "snapshot",
    )
    assert m.constraints["Generator-ext-p-upper"].dims == (
        "scenario",
        "name",
        "snapshot",
    )


def test_a_fixed_operation_is_stated_per_scenario(ac_dc_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    n.c.generators.dynamic.p_set = pd.DataFrame(
        30.0, index=n.snapshots, columns=n.c.generators.dynamic.p_max_pu.columns
    )
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: backend.define_fixed_operation_constraints(
            n, m, sns, sc, pe
        ),
    )
    assert m.constraints["Generator-p_set"].dims == ("scenario", "name", "snapshot")


def test_the_nodal_balance_is_stated_per_scenario(ac_dc_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    n.c.loads.dynamic.p_set[("low", "Manchester")] = 10.0
    n.c.loads.dynamic.p_set[("high", "Manchester")] = 90.0
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: backend.define_nodal_balance_constraints(
            n, m, sns, None, False, sc=sc, pe=pe
        ),
    )
    assert m.constraints["Bus-nodal_balance"].dims == ("scenario", "name", "snapshot")


def test_the_state_of_charge_carries_within_a_scenario(ac_dc_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    n.add("StorageUnit", "store", bus="Manchester", p_nom=100, max_hours=4)
    n.add("Store", "tank", bus="Manchester", e_nom=100)
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: (
            backend.define_storage_unit_constraints(n, m, sns, sc, pe),
            backend.define_store_constraints(n, m, sns, sc, pe),
        ),
    )
    assert m.constraints["StorageUnit-energy_balance"].dims == (
        "scenario",
        "name",
        "snapshot",
    )
    assert m.constraints["Store-energy_balance"].dims == (
        "scenario",
        "name",
        "snapshot",
    )


def test_the_voltage_law_and_a_supply_total_are_stated_per_scenario(ac_dc_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = ac_dc_stochastic
    n.c.generators.static.loc[(slice(None), "Manchester Wind"), "e_sum_max"] = 500.0
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: (
            backend.define_kirchhoff_voltage_constraints(n, m, sns, sc, pe),
            backend.define_total_supply_constraints(n, m, sns, sc=sc, pe=pe),
        ),
    )
    assert set(m.constraints["Kirchhoff-Voltage-Law"].dims) == {
        "scenario",
        "cycle",
        "snapshot",
    }
    assert set(m.constraints["Generator-e_sum_max"].dims) == {"scenario", "name"}


def test_commitment_is_stated_per_scenario(committable_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = committable_stochastic
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: backend.define_commitment_constraints(
            n, m, sns, False, sc=sc, pe=pe
        ),
    )
    for family in ("status", "start_up", "shut_down"):
        assert m.variables[f"Generator-{family}"].dims == (
            "scenario",
            "name",
            "snapshot",
        )
    assert m.constraints["Generator-com-p-upper"].dims == (
        "scenario",
        "name",
        "snapshot",
    )


def test_a_ramp_limit_is_stated_per_scenario(committable_stochastic):
    from pypsa.optimization.nimopt_backend import build as backend

    n = committable_stochastic
    n.c.generators.static.loc[(slice(None), "peak"), "ramp_limit_up"] = 0.1
    n.c.generators.static.loc[(slice(None), "peak"), "ramp_limit_down"] = 0.1
    m = _one_family(
        n,
        lambda n, m, sns, sc, pe: backend.define_ramp_limit_constraints(
            n, m, sns, sc, pe
        ),
    )
    assert m.constraints["Generator-p-ramp_limit_up"].dims == (
        "scenario",
        "name",
        "snapshot",
    )


def test_identical_scenarios_cost_what_one_deterministic_network_costs(ac_dc_network):
    # the probabilities sum to one, so scenarios carrying the same data state
    # the same problem: an unweighted objective would answer their sum instead
    a = ac_dc_network.copy()
    a.optimize(solver_name=SOLVER)
    b = ac_dc_network.copy()
    b.set_scenarios({"low": 0.3, "high": 0.7})
    b.optimize(backend="nimopt", solver_name=SOLVER)
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_a_global_constraint_is_stated_per_scenario(ac_dc_stochastic):
    m = ac_dc_stochastic.optimize.create_model(backend="nimopt")
    name = next(k for k in m.constraints if k.startswith("GlobalConstraint-"))
    assert m.constraints[name].dims == ("scenario",)


# --- what a stochastic answer must satisfy ----------------------------------


def _deterministic(n, scenario, p_nom=None, extendable=None):
    """The network of one scenario alone, as linopy builds it.

    A committable component under scenarios reaches linopy's refusal, so the
    identities a stochastic answer satisfies are checked against deterministic
    solves, which linopy performs.
    """
    d = pypsa.Network()
    d.set_snapshots(n.snapshots)
    for c_name in ("Bus", "Carrier", "Generator", "Load"):
        c = n.c[c_name]
        if c.static.empty:
            continue
        static = c.static.xs(scenario, level="scenario")
        for name, row in static.iterrows():
            d.add(c_name, name, **row.to_dict())
    for c_name, attr in (("Load", "p_set"),):
        frame = n.c[c_name].dynamic[attr]
        if not frame.empty:
            d.c[c_name].dynamic[attr] = frame.xs(scenario, level="scenario", axis=1)
    if p_nom is not None:
        d.c.generators.static.loc["peak", "p_nom"] = p_nom
        d.c.generators.static.loc["peak", "p_nom_extendable"] = False
    if extendable is not None:
        d.c.generators.static.loc["peak", "p_nom_extendable"] = extendable
    return d


def _solved(n, scenario, **kwargs):
    d = _deterministic(n, scenario, **kwargs)
    d.optimize(solver_name=SOLVER)
    return d


def test_a_network_without_a_first_stage_variable_decomposes(committable_stochastic):
    # scenarios couple only through a first-stage variable; carrying none, the
    # problem separates and the objective is the weighted sum of its scenarios
    n = committable_stochastic
    n.optimize(backend="nimopt", solver_name=SOLVER)
    stochastic = n.objective + n.objective_constant
    weighted = sum(
        p * _solved(committable_stochastic, s).objective
        for s, p in (("low", 0.4), ("high", 0.6))
    )
    assert stochastic == pytest.approx(weighted, rel=1e-6)


def test_an_answer_carrying_a_first_stage_capacity_prices_out(extendable_stochastic):
    # with a capacity decided once the problem does not separate, but its own
    # answer prices out: fix that capacity and solve each scenario alone
    n = extendable_stochastic
    n.optimize(backend="nimopt", solver_name=SOLVER)
    stochastic = n.objective + n.objective_constant
    built = float(n.c.generators.static.loc[("low", "peak"), "p_nom_opt"])
    capex = 300.0 * built
    opex = sum(
        p * _solved(extendable_stochastic, s, p_nom=built).objective
        for s, p in (("low", 0.4), ("high", 0.6))
    )
    assert stochastic == pytest.approx(capex + opex, rel=1e-6)


def test_the_value_of_perfect_information_is_not_negative(extendable_stochastic):
    # a scenario choosing its own capacity costs no more than one capacity
    # serving every scenario
    n = extendable_stochastic
    n.optimize(backend="nimopt", solver_name=SOLVER)
    stochastic = n.objective + n.objective_constant
    wait_and_see = sum(
        p * _solved(extendable_stochastic, s, extendable=True).objective
        for s, p in (("low", 0.4), ("high", 0.6))
    )
    assert wait_and_see <= stochastic + 1e-6


# --- risk preference ---------------------------------------------------------


def test_a_risk_averse_network_states_the_cvar_families(extendable_stochastic):
    n = extendable_stochastic
    n.set_risk_preference(alpha=0.5, omega=0.5)
    m = n.optimize.create_model(backend="nimopt")
    assert m.variables["CVaR-a"].dims == ("scenario",)
    assert m.variables["CVaR-theta"].dims == ()
    assert m.variables["CVaR"].dims == ()
    assert m.constraints["CVaR-excess"].dims == ("scenario",)
    assert m.constraints["CVaR-def"].dims == ()


@pytest.mark.parametrize(
    ("alpha", "omega"), [(0.5, 0.5), (0.9, 0.3), (0.95, 1.0), (0.5, 0.0)]
)
def test_the_cvar_objective_agrees_with_linopy(extendable_stochastic, alpha, omega):
    a = extendable_stochastic.copy()
    a.set_risk_preference(alpha=alpha, omega=omega)
    a.optimize(solver_name=SOLVER)
    b = extendable_stochastic.copy()
    b.set_risk_preference(alpha=alpha, omega=omega)
    b.optimize(backend="nimopt", solver_name=SOLVER)
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


def test_no_risk_aversion_reproduces_the_risk_neutral_objective(extendable_stochastic):
    # omega = 0 states the risk-neutral blend, so the CVaR rows stand in the
    # model and price nothing
    plain = extendable_stochastic.copy()
    plain.optimize(backend="nimopt", solver_name=SOLVER)
    averse = extendable_stochastic.copy()
    averse.set_risk_preference(alpha=0.9, omega=0.0)
    averse.optimize(backend="nimopt", solver_name=SOLVER)
    assert averse.objective == pytest.approx(plain.objective, rel=1e-9)


def test_the_conditional_value_is_no_less_than_the_expected_cost(extendable_stochastic):
    # a tail average is never below the mean it is drawn from
    n = extendable_stochastic
    n.set_risk_preference(alpha=0.5, omega=1.0)
    n.optimize(backend="nimopt", solver_name=SOLVER)
    m = n.model
    cvar = float(m.variables["CVaR"].solution)
    operating = 0.0
    for s, p in (("low", 0.4), ("high", 0.6)):
        held = n.c.generators.dynamic.p.xs(s, level="scenario", axis=1)
        cost = n.c.generators.static.xs(s, level="scenario")["marginal_cost"]
        operating += p * float((held * cost).to_numpy().sum())
    assert cvar >= operating - 1e-6


def test_a_risk_averse_commitment_is_stated_where_linopy_raises(committable_stochastic):
    # linopy addresses the status variable without its scenario dimension, so
    # it states no model here; the backend does, and the tail prices out
    n = committable_stochastic
    n.set_risk_preference(alpha=0.5, omega=1.0)
    with pytest.raises(KeyError, match="scenario"):
        n.optimize.create_model()

    averse = committable_stochastic
    averse.set_risk_preference(alpha=0.5, omega=1.0)
    averse.optimize(backend="nimopt", solver_name=SOLVER)
    tail = float(averse.model.variables["CVaR"].solution)
    # a tail mass of 1 - alpha = 0.5 sits inside the high scenario, whose
    # probability is 0.6, so the tail average is that scenario's own cost
    assert averse.objective == pytest.approx(tail, rel=1e-9)

    neutral = committable_stochastic.copy()
    neutral.optimize(backend="nimopt", solver_name=SOLVER)
    # pricing the tail alone costs no less than pricing the mean
    assert tail >= neutral.objective - 1e-6


# --- a committable whose capacity comes in modules ---------------------------


def modular_committable(extendable=True):
    """A unit committed module by module against a load that swings."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(4))
    n.add("Bus", "b")
    held = {
        "bus": "b",
        "marginal_cost": 10,
        "committable": True,
        "p_min_pu": 0.4,
        "p_nom_mod": 25,
        "start_up_cost": 100,
    }
    if extendable:
        held |= {
            "p_nom": 0,
            "p_nom_extendable": True,
            "p_nom_max": 100,
            "capital_cost": 5,
        }
    else:
        held |= {"p_nom": 100}
    n.add("Generator", "g", **held)
    n.add("Generator", "back", bus="b", p_nom=500, marginal_cost=300)
    n.add("Load", "l", bus="b", p_set=[30, 70, 90, 20])
    return n


def test_a_modular_commitment_counts_running_modules():
    n = modular_committable()
    m = n.optimize.create_model(backend="nimopt")
    # the status counts modules rather than standing at one, so it is an
    # integer column bounded by the number of modules built
    assert "Generator-status-p_nom-variable-upper" in m.constraints
    assert m.model.integrality().any()


def test_a_modular_commitment_reaches_the_objective_linopy_reaches():
    a = solved(modular_committable(), "linopy")
    b = solved(modular_committable(), "nimopt")
    assert b.objective == pytest.approx(a.objective, rel=1e-6)
    # the count of running modules is a whole number bounded by those built,
    # and the output sits between the module shares that count allows; the
    # count itself is free to tie between optima, so it is not compared
    status = b.c.generators.dynamic.status["g"].to_numpy()
    output = b.c.generators.dynamic.p["g"].to_numpy()
    built = b.c.generators.static.loc["g", "p_nom_opt"] / 25
    assert status == pytest.approx(np.round(status))
    assert (status <= built + 1e-6).all()
    assert (output >= 0.4 * 25 * status - 1e-6).all()
    assert (output <= 25 * status + 1e-6).all()


def test_a_fixed_modular_commitment_is_refused_by_name():
    # PyPSA states both the capacity bounds and the module bounds for this
    # pair, and together they hold the status at zero, so the unit can never
    # run; the backend refuses rather than answer that
    n = modular_committable(extendable=False)
    with pytest.raises(NotImplementedError, match="committable modular"):
        n.optimize.create_model(backend="nimopt")


# --- piecewise attributes: method, option groups, breakpoints ---------------


def points(rows):
    """Breakpoints over `name` and linopy's breakpoint dimension, one row per name."""
    width = max(len(row) for row in rows.values())
    values = [list(row) + [np.nan] * (width - len(row)) for row in rows.values()]
    return xr.DataArray(
        np.array(values, dtype=float),
        coords={"name": list(rows), BREAKPOINT_DIM: np.arange(width)},
        dims=("name", BREAKPOINT_DIM),
    )


CONVEX = ({"a": [0, 50, 100]}, {"a": [0, 50, 250]})
CONCAVE = ({"a": [0, 50, 100]}, {"a": [0, 30, 55]})
RESOLVED = [
    ("convex under >=", "auto", ">=", False, *CONVEX, "tangent"),
    ("concave under >=", "auto", ">=", False, *CONCAVE, "incremental"),
    ("convex under <=", "auto", "<=", False, *CONVEX, "incremental"),
    ("concave under <=", "auto", "<=", False, *CONCAVE, "tangent"),
    ("a status", "auto", ">=", True, *CONVEX, "incremental"),
    ("equality", "auto", "==", False, *CONVEX, "incremental"),
    # linopy resolves sos2 here: y is not strictly monotonic
    (
        "flat y",
        "auto",
        "==",
        False,
        {"a": [0, 10, 50, 100]},
        {"a": [0, 3, 10, 10]},
        "incremental",
    ),
    (
        "one convex and one concave",
        "auto",
        ">=",
        False,
        {"a": [0, 50, 100], "b": [0, 50, 100]},
        {"a": [0, 50, 250], "b": [0, 30, 55]},
        "incremental",
    ),
    (
        "a two-point curve beside a convex one",
        "auto",
        ">=",
        False,
        {"a": [0, 50, 100], "b": [0, 100]},
        {"a": [0, 50, 250], "b": [0, 20]},
        "tangent",
    ),
    ("lp", "lp", "==", False, *CONVEX, "tangent"),
    ("incremental", "incremental", ">=", False, *CONVEX, "incremental"),
]


@pytest.mark.parametrize(
    ("label", "requested", "sign", "status", "x", "y", "method"), RESOLVED
)
def test_the_method_resolves_as_linopys_auto_resolves(
    label, requested, sign, status, x, y, method
):
    resolved = resolve_method(
        requested,
        sign,
        has_status=status,
        x_points=points(x),
        y_points=points(y),
        owner="the curve",
    )
    assert resolved == method


def test_method_sos2_is_not_supported():
    message = (
        "method 'sos2' of the curve is not supported by the nimopt backend; "
        "use method 'auto', 'lp' or 'incremental'"
    )
    with pytest.raises(NotImplementedError, match=re.escape(message)):
        resolve_method(
            "sos2",
            ">=",
            has_status=False,
            x_points=points(CONVEX[0]),
            y_points=points(CONVEX[1]),
            owner="the curve",
        )


def test_x_breakpoints_that_repeat_are_not_supported():
    message = (
        "the curve has x breakpoints that are not strictly monotonic; give "
        "strictly increasing x breakpoints"
    )
    with pytest.raises(NotImplementedError, match=re.escape(message)):
        resolve_method(
            "auto",
            ">=",
            has_status=False,
            x_points=points({"a": [0, 50, 50, 100]}),
            y_points=points({"a": [0, 10, 20, 30]}),
            owner="the curve",
        )


def test_an_unknown_method_raises():
    message = (
        "method of the curve is one of ('auto', 'lp', 'incremental', 'sos2'); "
        "got 'spline'"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_method(
            "spline",
            ">=",
            has_status=False,
            x_points=points(CONVEX[0]),
            y_points=points(CONVEX[1]),
            owner="the curve",
        )


def listed(groups):
    return [
        (suffix, list(names), method, sign) for suffix, names, method, sign in groups
    ]


def test_options_form_groups_in_pypsas_order():
    names = pd.Index(["a", "b", "c", "d"], name="name")
    options = [
        PiecewiseOptions(
            "Generator", "marginal_cost", ">=", name="a", method="incremental"
        ),
        PiecewiseOptions(
            "Generator", "marginal_cost", "<=", name=("c", "x"), method="lp"
        ),
    ]
    # the named options sort by name in reverse: ("c", "x") before ("a",)
    assert listed(option_groups(names, options, ">=")) == [
        ("-option0", ["c"], "lp", "<="),
        ("-option1", ["a"], "incremental", ">="),
        ("", ["b", "d"], "auto", ">="),
    ]


def test_an_option_without_names_covers_every_remaining_name():
    names = pd.Index(["a", "b"], name="name")
    options = [PiecewiseOptions("Generator", "marginal_cost", "<=", method="lp")]
    assert listed(option_groups(names, options, ">=")) == [("", ["a", "b"], "lp", "<=")]


def test_a_breakpoint_parameter_has_a_value_at_each_valid_breakpoint():
    N = no.Set("Generator", np.array(["a", "b"]))
    B = no.Set("curve_breakpoint", np.arange(3))
    x = points({"a": [0, 50, 100], "b": [0, 100]})
    held = breakpoint_param("curve-x_points", (N, B), x, x.notnull())
    assert held.name == "curve_x_points"
    assert held.materialise().values().tolist() == [0.0, 50.0, 100.0, 0.0, 100.0]


# --- piecewise attributes: the same optimum as linopy ------------------------

LINOPY_METHOD = {"lp": "tangent", "incremental": "incremental", "sos2": "incremental"}
RESULT_FRAMES = ("p", "p0", "p1", "p2", "p_dispatch", "p_store")


def piecewise_results(n):
    """The dispatch, the capacities and every piecewise result PyPSA assigns.

    A piecewise result per snapshot is the curve divided by the dispatch, and
    is compared where the dispatch is not zero.
    """
    held = {}
    for c in n.components:
        if c.static.empty:
            continue
        for key, frame in c.dynamic.items():
            if frame.empty:
                continue
            if key in RESULT_FRAMES:
                held[f"{c.name}.{key}"] = frame.to_numpy(dtype=float)
            elif key.endswith("_piecewise_opt"):
                dispatch = c.dynamic["p_dispatch" if c.name == "StorageUnit" else "p"]
                running = dispatch.reindex_like(frame).abs() > 1e-6
                held[f"{c.name}.{key}"] = frame.where(running).to_numpy(dtype=float)
        for column in c.static.columns:
            if column.endswith(("_piecewise_opt", "_nom_opt")):
                held[f"{c.name}.{column}"] = c.static[column].to_numpy(dtype=float)
    return held


def assert_same_as_linopy(build):
    """Solve `build()` on both backends; the objective and the results agree."""
    a, kwargs = build()
    b, _ = build()
    a.optimize(solver_name=SOLVER, reformulate_sos=True, **kwargs)
    b.optimize(solver_name=SOLVER, backend="nimopt", **kwargs)
    assert b.objective == pytest.approx(a.objective, rel=1e-6)
    left, right = piecewise_results(a), piecewise_results(b)
    assert sorted(right) == sorted(left)
    for key, values in left.items():
        np.testing.assert_allclose(
            right[key], values, rtol=1e-6, atol=1e-6, err_msg=key
        )


def assert_same_method(build):
    """Each PyPSA call resolves to linopy's method, mapped to nimopt's."""
    a, kwargs = build()
    b, _ = build()
    linopy = a.optimize.create_model(**kwargs)
    nimopt = b.optimize.create_model(backend="nimopt", **kwargs).model
    stems = [name for name in linopy.variables if name.endswith("_piecewise")]
    assert stems
    for stem in stems:
        expected = {
            LINOPY_METHOD[formulation.method]
            for name, formulation in linopy._piecewise_formulations.items()
            if name == stem or name.startswith(f"{stem}_")
        }
        symbol = _symbol(stem)
        found = {
            declaration.method
            for name, declaration in nimopt.piecewise_declarations.items()
            if name == symbol or name.startswith(f"{symbol}_")
        }
        assert found == expected, stem


def cost_curve(curve, **generator):
    """A generator with a cost curve beside one at a fixed marginal cost."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
    n.add("Generator", "gen1", bus="bus0", p_nom=100, marginal_cost=curve, **generator)
    n.add("Load", "load", bus="bus0", p_set=80)
    return n, {}


def ragged_costs():
    """Two generators whose curves have three and two breakpoints."""
    nan = float("nan")
    segments = pd.DataFrame(
        [[0.0, 10.0, 0.0, 5.0], [0.5, 20.0, 1.0, 25.0], [1.0, 40.0, nan, nan]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("gen0", "p_pu"),
                ("gen0", "marginal_cost"),
                ("gen1", "p_pu"),
                ("gen1", "marginal_cost"),
            ],
            names=["name", "attribute"],
        ),
    )
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Generator", ["gen0", "gen1"], bus="bus0", p_nom=100, marginal_cost=segments)
    n.add("Load", "load", bus="bus0", p_set=80)
    return n, {}


def storage_costs(kind):
    """A storage unit or a store whose dispatch has a cost curve."""
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
    curve = {0.0: 0.0, 0.5: 3.0, 1.0: 10.0}
    if kind == "StorageUnit":
        n.add(
            "StorageUnit",
            "su0",
            bus="bus0",
            p_nom=100,
            max_hours=1,
            state_of_charge_initial=75,
            marginal_cost=curve,
        )
    else:
        n.add(
            "Store", "store0", bus="bus0", e_nom=100, e_initial=75, marginal_cost=curve
        )
    n.add("Load", "load", bus="bus0", p_set=50)
    return n, {}


def committed_costs(linearized=False):
    """A committable and a non-committable generator, each with a cost curve."""
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus0")
    n.add("Load", "load", bus="bus0", p_set=[80, 150])
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
    n.add(
        "Generator",
        "gen1",
        bus="bus0",
        p_nom=100,
        marginal_cost={0.0: 60.0, 0.1: 60.0, 0.5: 35.0, 1.0: 100.0},
    )
    n.add(
        "Generator",
        "gen-committable",
        bus="bus0",
        p_nom=100,
        marginal_cost={0.0: 60, 0.5: 75, 1.0: 100.0},
        committable=True,
        stand_by_cost=5,
    )
    return n, {"linearized_unit_commitment": True} if linearized else {}


def optioned_costs(named):
    """Cost curves under one option: without names, or naming one generator."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
    n.add(
        "Generator",
        "gen1",
        bus="bus0",
        p_nom=100,
        marginal_cost={0.0: 0.0, 0.5: 1.0, 1.0: 4.0},
    )
    if named:
        n.add(
            "Generator",
            "gen2",
            bus="bus0",
            p_nom=100,
            marginal_cost={0.0: 0.0, 0.5: 20.0, 1.0: 30.0},
        )
        option = PiecewiseOptions(
            "Generator", "marginal_cost", ">=", name="gen2", method="incremental"
        )
    else:
        option = PiecewiseOptions("Generator", "marginal_cost", ">=", method="lp")
    n.add("Load", "load", bus="bus0", p_set=150)
    return n, {"piecewise_options": [option]}


def modules_beside_a_curve():
    """A modular committable, with a status that counts modules, beside a curve."""
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus0")
    n.add("Load", "load", bus="bus0", p_set=[80, 150])
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50)
    n.add(
        "Generator",
        "gen-committable",
        bus="bus0",
        p_nom=100,
        marginal_cost={0.0: 60, 0.5: 75, 1.0: 100.0},
        committable=True,
        stand_by_cost=5,
    )
    n.add(
        "Generator",
        "modules",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_mod=10,
        p_nom_max=50,
        committable=True,
        capital_cost=1,
        marginal_cost=40,
    )
    return n, {}


COST_CURVES = {
    "convex": lambda: cost_curve({0.0: 0.0, 0.5: 1.0, 1.0: 4.0}),
    "not convex": lambda: cost_curve({0.0: 0.0, 0.1: 60.0, 0.5: 35.0, 1.0: 100.0}),
    "ragged": ragged_costs,
    "storage unit": lambda: storage_costs("StorageUnit"),
    "store": lambda: storage_costs("Store"),
    "committable and not": committed_costs,
    "committable, linearized": lambda: committed_costs(linearized=True),
    "an option without names": lambda: optioned_costs(named=False),
    "a named option": lambda: optioned_costs(named=True),
    "a modular committable beside a curve": modules_beside_a_curve,
}


@pytest.mark.parametrize("label", sorted(COST_CURVES))
def test_a_cost_curve_reaches_the_optimum_linopy_reaches(label):
    assert_same_as_linopy(COST_CURVES[label])


@pytest.mark.parametrize("label", sorted(COST_CURVES))
def test_a_cost_curve_resolves_the_method_linopy_resolves(label):
    assert_same_method(COST_CURVES[label])


def test_a_convex_cost_curve_prices_the_bus_as_linopy_does():
    a, _ = COST_CURVES["convex"]()
    b, _ = COST_CURVES["convex"]()
    a.optimize(solver_name=SOLVER)
    b.optimize(solver_name=SOLVER, backend="nimopt")
    np.testing.assert_allclose(
        b.c["Bus"].dynamic["marginal_price"].to_numpy(),
        a.c["Bus"].dynamic["marginal_price"].to_numpy(),
    )


def test_a_model_with_cost_curves_reads_back_from_its_file(tmp_path):
    n, kwargs = COST_CURVES["committable and not"]()
    model = n.optimize.create_model(backend="nimopt", **kwargs).model
    no.save(model, tmp_path / "curves.yaml")
    assert "where:" in (tmp_path / "curves.yaml").read_text()
    back = no.load(tmp_path / "curves.yaml")
    assert back.solve(SOLVER).objective == pytest.approx(model.solve(SOLVER).objective)


def test_a_curve_under_method_sos2_is_not_supported():
    n, _ = COST_CURVES["convex"]()
    option = PiecewiseOptions("Generator", "marginal_cost", ">=", method="sos2")
    message = (
        "method 'sos2' of piecewise 'marginal_cost' of Generator is not "
        "supported by the nimopt backend"
    )
    with pytest.raises(NotImplementedError, match=re.escape(message)):
        n.optimize.create_model(backend="nimopt", piecewise_options=[option])


def test_a_curve_with_a_repeated_x_breakpoint_is_not_supported():
    # linopy resolves sos2 for this curve
    n, _ = cost_curve(
        pd.DataFrame(
            {"p_pu": [0.0, 0.5, 0.5, 1.0], "marginal_cost": [0.0, 1.0, 2.0, 3.0]}
        )
    )
    message = (
        "piecewise 'marginal_cost' of Generator has x breakpoints that are not "
        "strictly monotonic"
    )
    with pytest.raises(NotImplementedError, match=re.escape(message)):
        n.optimize.create_model(backend="nimopt")


def test_a_curve_on_a_network_with_scenarios_is_not_supported(monkeypatch):
    # PyPSA rejects a curve and scenarios together; the backend checks as well
    n, _ = COST_CURVES["convex"]()
    monkeypatch.setattr(pypsa.Network, "has_scenarios", property(lambda self: True))
    with pytest.raises(NotImplementedError, match="piecewise Generator marginal_cost"):
        n.optimize.create_model(backend="nimopt", consistency_check=False)


def cost_over_periods(curve):
    """Cost curves of constant slope over two weighted periods, or their linear costs."""
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.investment_period_weightings["objective"] = [1.0, 0.7]
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom=100,
        marginal_cost=50,
        build_year=2020,
        lifetime=30,
    )
    for name, size, cost, built in (
        ("early", 40, 20.0, 2020),
        ("late", 60, 10.0, 2030),
    ):
        n.add(
            "Generator",
            name,
            bus="bus0",
            p_nom=size,
            marginal_cost={0.0: cost, 1.0: cost} if curve else cost,
            build_year=built,
            lifetime=30,
        )
    n.add("Load", "load", bus="bus0", p_set=[80, 90, 80, 90])
    return n


def test_a_cost_curve_over_investment_periods_prices_as_its_linear_cost():
    # linopy builds no operational curve over investment periods; a curve of
    # constant slope prices each period as the linear cost does: 7640
    linear, curved = cost_over_periods(curve=False), cost_over_periods(curve=True)
    linear.optimize(solver_name=SOLVER, multi_investment_periods=True)
    curved.optimize(solver_name=SOLVER, backend="nimopt", multi_investment_periods=True)
    assert curved.objective == pytest.approx(linear.objective, rel=1e-9)
    np.testing.assert_allclose(
        curved.c["Generator"].dynamic["p"].to_numpy(),
        linear.c["Generator"].dynamic["p"].to_numpy(),
        atol=1e-9,
    )


# --- piecewise attributes: capital cost --------------------------------------


def capital_costs(periods=False):
    """An extendable generator with a capital cost curve beside a linear one."""
    n = pypsa.Network(snapshots=range(2))
    if periods:
        n.investment_periods = [2020, 2030]
    n.add("Bus", "bus0")
    lived = {"build_year": 2020, "lifetime": 30} if periods else {}
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=1.8,
        marginal_cost=1,
        **lived,
    )
    n.add(
        "Generator",
        "gen1",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=pd.DataFrame(
            {"p_nom": [0.0, 10, 50, 100.0], "capital_cost": [0.0, 1, 1.5, 2.0]}
        ),
        marginal_cost=2,
        **lived,
    )
    n.add("Load", "load", bus="bus0", p_set=150 if periods else [150, 120])
    return n, {"multi_investment_periods": True} if periods else {}


CAPITAL_CURVES = {
    "one period": capital_costs,
    "two investment periods": lambda: capital_costs(periods=True),
}


@pytest.mark.parametrize("label", sorted(CAPITAL_CURVES))
def test_a_capital_cost_curve_reaches_the_optimum_linopy_reaches(label):
    assert_same_as_linopy(CAPITAL_CURVES[label])


@pytest.mark.parametrize("label", sorted(CAPITAL_CURVES))
def test_a_capital_cost_curve_resolves_the_method_linopy_resolves(label):
    assert_same_method(CAPITAL_CURVES[label])


def test_a_capital_cost_curve_with_an_overnight_cost_raises():
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen0",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost={0.0: 2.0, 100.0: 2.0},
        overnight_cost=1000,
        lifetime=20,
        discount_rate=0.05,
    )
    n.add("Load", "load", bus="bus0", p_set=50)
    message = (
        "Components ['gen0'] of type Generator define both a piecewise "
        "'capital_cost' curve and 'overnight_cost'."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        n.optimize.create_model(backend="nimopt", include_objective_constant=False)


# --- piecewise attributes: efficiency at a port ------------------------------


def port_curve(component, attr, curve, delayed=False):
    """A link or process whose output at bus1 follows an efficiency curve."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", ["bus0", "bus1"])
    n.add("Carrier", "gas")
    n.add("Generator", "gen0", carrier="gas", bus="bus0", p_nom=150, marginal_cost=1)
    n.add("Load", "load", bus="bus1", p_set=[20, 30, 40])
    common = {"carrier": "gas", "bus0": "bus0", "bus1": "bus1", "p_nom": 100}
    n.add(
        component,
        "curved",
        marginal_cost=20,
        **{attr: curve},
        **({"delay1": 1} if delayed else {}),
        **common,
    )
    if delayed:
        n.add(component, "plain", marginal_cost=30, **{attr: 0.5}, **common)
    return n, {}


def two_port_curve(component):
    """A link or process with a fixed rate at bus1 and a curve at bus2."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", ["bus0", "bus1", "bus2"])
    n.add("Carrier", "gas")
    n.add("Generator", "gen0", carrier="gas", bus="bus0", p_nom=100, marginal_cost=1)
    n.add("Load", "load1", bus="bus1", p_set=[25, 31.25, 37.5])
    n.add("Load", "load2", bus="bus2", p_set=[20, 30, 40])
    first, second = ("efficiency", "efficiency2")
    if component == "Process":
        first, second = ("rate1", "rate2")
    n.add(
        component,
        "curved",
        carrier="gas",
        bus0="bus0",
        bus1="bus1",
        bus2="bus2",
        p_nom=100,
        marginal_cost=20,
        **{first: 0.5, second: {0.0: 0.0, 0.1: 0.3, 0.5: 0.4, 1.0: 0.6}},
    )
    return n, {}


def committed_links():
    """A committable and a non-committable link on a curve linopy formulates by SOS2."""
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", ["bus0", "bus1"])
    n.add("Carrier", "gas")
    n.add("Generator", "gen0", carrier="gas", bus="bus0", p_nom=150)
    n.add("Load", "load", bus="bus1", p_set=[20, 30, 40])
    common = {
        "carrier": "gas",
        "bus0": "bus0",
        "bus1": "bus1",
        "p_nom": 100,
        "marginal_cost": 20,
    }
    curve = {0.0: 0.0, 0.1: 0.3, 0.5: 0.2, 1.0: 0.1}
    n.add("Link", "plain", efficiency=0.5, **common)
    n.add("Link", "free", efficiency=curve, p_min_pu=0.1, **common)
    n.add(
        "Link", "committed", efficiency=curve, committable=True, p_min_pu=0.1, **common
    )
    return n, {}


RISING = {0.0: 0.0, 0.5: 0.5, 1.0: 0.75}
PORT_CURVES = {
    "link efficiency": lambda: port_curve("Link", "efficiency", RISING),
    "process rate1": lambda: port_curve("Process", "rate1", RISING),
    "delayed process": lambda: port_curve("Process", "rate1", RISING, delayed=True),
    "link efficiency2": lambda: two_port_curve("Link"),
    "process rate2": lambda: two_port_curve("Process"),
    "committed links, sos2 in linopy": committed_links,
}


@pytest.mark.parametrize("label", sorted(PORT_CURVES))
def test_a_port_curve_reaches_the_optimum_linopy_reaches(label):
    assert_same_as_linopy(PORT_CURVES[label])


@pytest.mark.parametrize("label", sorted(PORT_CURVES))
def test_a_port_curve_resolves_the_method_linopy_resolves(label):
    assert_same_method(PORT_CURVES[label])


# --- piecewise attributes: primary energy -----------------------------------


def primary_curve(status=False):
    """Generators whose primary energy follows an efficiency curve, under a CO2 limit."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Carrier", "gas", co2_emissions=1.0)
    if status:
        curve = pd.DataFrame(
            {"p_pu": [0.0, 0.1, 0.5, 1.0], "efficiency": [0.0, 0.3, 0.3, 0.3]}
        )
        n.add(
            "Generator",
            "gen0",
            carrier="gas",
            bus="bus0",
            p_nom=80,
            marginal_cost=15,
            efficiency=0.6,
        )
        for name, committable in (("committed", True), ("free", False)):
            n.add(
                "Generator",
                name,
                carrier="gas",
                bus="bus0",
                p_nom=70,
                marginal_cost=20,
                p_min_pu=0.1,
                efficiency=curve,
                committable=committable,
            )
        n.add("Load", "load", bus="bus0", p_set=80)
    else:
        n.add(
            "Generator",
            "gen",
            carrier="gas",
            bus="bus0",
            p_nom=70,
            marginal_cost=20,
            efficiency={0.0: 0.0, 0.1: 0.2, 0.5: 0.4, 1.0: 0.6},
        )
        n.add("Generator", "backup", bus="bus0", p_nom=100, marginal_cost=100)
        n.add("Load", "load", bus="bus0", p_set=50)
    n.add(
        "GlobalConstraint",
        "co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=160,
    )
    return n, {}


PRIMARY_CURVES = {
    "one generator": primary_curve,
    "committable and not": lambda: primary_curve(status=True),
}


@pytest.mark.parametrize("label", sorted(PRIMARY_CURVES))
def test_a_primary_energy_curve_reaches_the_optimum_linopy_reaches(label):
    assert_same_as_linopy(PRIMARY_CURVES[label])


@pytest.mark.parametrize("label", sorted(PRIMARY_CURVES))
def test_a_primary_energy_curve_resolves_the_method_linopy_resolves(label):
    assert_same_method(PRIMARY_CURVES[label])
