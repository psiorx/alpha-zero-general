import sys

try:
    import pulp
except ImportError:
    pulp = None


def solve_mdp_lp(mdp):
    if pulp is None:
        raise RuntimeError("pulp is not installed. Install with: pip install pulp")

    states = list(mdp.states())
    if not states:
        raise ValueError("MDP has no states")

    # For maximizing return, V* is the minimal solution of the Bellman optimality
    # inequalities V(s) >= max_a E[r + gamma V(s')], so we minimize the objective.
    vmin, vmax = mdp.value_bounds()
    has_min_states = any((not mdp.is_terminal(s)) and (not mdp.is_max_state(s)) for s in states)

    prob = pulp.LpProblem("mdp_lp", pulp.LpMaximize)
    state_ids = {s: i for i, s in enumerate(states)}
    V = {s: pulp.LpVariable(f"V_{state_ids[s]}", lowBound=vmin, upBound=vmax) for s in states}

    gamma = float(mdp.discount)

    for s in states:
        if mdp.is_terminal(s):
            prob += V[s] == 0.0
            continue
        actions = list(mdp.actions(s))
        if not actions:
            continue
        for a in actions:
            rhs = 0.0
            for p, s_next, r in mdp.transitions(s, a):
                rhs += p * (r + gamma * V[s_next])
            if has_min_states:
                if mdp.is_max_state(s):
                    prob += V[s] >= rhs
                else:
                    prob += V[s] <= rhs
            else:
                prob += V[s] >= rhs

        if has_min_states:
            M = vmax - vmin
            sid = state_ids[s]
            y = {a: pulp.LpVariable(f"y_{sid}_{a}", lowBound=0, upBound=1, cat="Binary")
                 for a in actions}
            prob += pulp.lpSum([y[a] for a in actions]) == 1
            for a in actions:
                rhs = 0.0
                for p, s_next, r in mdp.transitions(s, a):
                    rhs += p * (r + gamma * V[s_next])
                if mdp.is_max_state(s):
                    prob += V[s] <= rhs + M * (1 - y[a])
                else:
                    prob += V[s] >= rhs - M * (1 - y[a])

    weights = mdp.state_relevance(states)
    if has_min_states:
        if weights is None:
            prob += pulp.lpSum([V[s] for s in states if mdp.is_max_state(s)])
        elif isinstance(weights, dict):
            prob += pulp.lpSum([weights.get(s, 0.0) * V[s] for s in states])
        else:
            if len(weights) != len(states):
                raise ValueError("state_relevance length does not match number of states")
            prob += pulp.lpSum([w * V[s] for w, s in zip(weights, states)])
    else:
        # Standard MDP LP: minimize the upper envelope of Bellman optimality.
        prob.sense = pulp.LpMinimize
        if weights is None:
            prob += pulp.lpSum([V[s] for s in states])
        elif isinstance(weights, dict):
            prob += pulp.lpSum([weights.get(s, 0.0) * V[s] for s in states])
        else:
            if len(weights) != len(states):
                raise ValueError("state_relevance length does not match number of states")
            prob += pulp.lpSum([w * V[s] for w, s in zip(weights, states)])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError("LP solver did not find an optimal solution")

    values = {s: float(pulp.value(V[s])) for s in states}
    return values


def main():
    print("This module provides solve_mdp_lp(mdp).", file=sys.stderr)
    print("Create an MDP adapter and call solve_mdp_lp.")


if __name__ == '__main__':
    main()
