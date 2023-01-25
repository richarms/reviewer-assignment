import pulp
import uuid
import numpy as np


class Model(object):
    """Reviewer Assignment formulated as a Linear Program."""

    def __init__(self, loads, coverages, weights, loads_lb=None):
        """Initialize the Basic matcher
        Args:
            loads - a list of integers specifying the maximum number of papers
                  for each reviewer.
            coverages - a list of integers specifying the number of reviews per
                 paper.
            weights - the affinity matrix (np.array) of papers to reviewers.
                   Rows correspond to reviewers and columns correspond to
                   papers.
            loads_lb - a list of integers specifying the min number of papers
                  for each reviewer (optional).
        Returns:
            initialized matcher.
        """
        self.weights = weights
        self.n_rev = np.size(weights, axis=0)
        self.n_pap = np.size(weights, axis=1)
        self.loads = loads
        self.loads_lb = loads_lb
        self.coverages = coverages

        assert np.sum(coverages) <= np.sum(loads)
        if loads_lb is not None:
            assert np.sum(coverages) >= np.sum(loads_lb)

        self.id = uuid.uuid1()
        self.opt_model = pulp.LpProblem(name=f"{self.id}:lp")

        self.coeff = list(self.weights.flatten())

        # Primal Vars.
        # self.lp_vars = self.m.addVars(self.n_rev, self.n_pap, vtype=GRB.BINARY, name="x", obj=coeff)
        self.lp_vars = LpVariable.dicts(
            "x", (range(self.n_rev), range(self.n_pap)), cat="Binary"
        )
        self.lp_vars.update()

        # Objective.
        self.opt_model.sense = pulp.LpMaximize
        # self.opt_model.update()

        # Constraints.
        opt_model.Constraints = self.opt_model.addConstraint(
            (self.lp_vars.sum(r, "*") <= l for r, l in enumerate(self.loads))
        )
