# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Build and solve a network's optimisation problem with nimopt instead of linopy.

Selected by ``pypsa.options.params.optimize.backend = "nimopt"`` or by
``backend="nimopt"`` on ``n.optimize`` and ``n.optimize.create_model``. The
model stored at ``n.model`` exposes what PyPSA's solution and dual assignment
read from a linopy model: ``variables``, ``constraints``, ``objective.value``
and ``solve``.
"""

from pypsa.optimization.nimopt_backend.build import create_model
from pypsa.optimization.nimopt_backend.model import NimoptModel

__all__ = ["NimoptModel", "create_model"]
