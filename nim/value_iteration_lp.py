import argparse
import sys
import time

import numpy as np

try:
    import pulp
except ImportError:
    pulp = None


def build_and_solve(max_stones, max_take):
    if pulp is None:
        raise RuntimeError("pulp is not installed. Install with: pip install pulp")

    prob = pulp.LpProblem("misere_nim", pulp.LpMaximize)

    V = {
        s: pulp.LpVariable(f"V_{s}", lowBound=-1.0, upBound=1.0)
        for s in range(0, max_stones + 1)
    }

    prob += V[0] == 1.0

    M = 2.0
    policy = {}
    for s in range(1, max_stones + 1):
        valid_takes = [take for take in range(1, max_take + 1) if take <= s]
        for take in valid_takes:
            policy[(s, take)] = pulp.LpVariable(f"y_{s}_{take}", lowBound=0, upBound=1, cat="Binary")
        prob += pulp.lpSum([policy[(s, take)] for take in valid_takes]) == 1
        for take in valid_takes:
            next_s = s - take
            if next_s == 0:
                q = -1.0
                prob += V[s] >= q
                prob += V[s] <= q + M * (1 - policy[(s, take)])
            else:
                prob += V[s] >= -V[next_s]
                prob += V[s] <= -V[next_s] + M * (1 - policy[(s, take)])

    prob += pulp.lpSum([V[s] for s in range(1, max_stones + 1)])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError("LP solver did not find an optimal solution")

    values = np.zeros(max_stones + 1, dtype=np.float64)
    for s in range(0, max_stones + 1):
        values[s] = float(pulp.value(V[s]))

    return values


def main():
    parser = argparse.ArgumentParser(description="Solve misere Nim via linear programming.")
    parser.add_argument('--max_stones', type=int, default=21, help='Number of stones')
    parser.add_argument('--max_take', type=int, default=3, help='Max stones to take')
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        values = build_and_solve(args.max_stones, args.max_take)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print("stones | value | optimal_take")
    print("-------+-------+-------------")
    for stones in range(1, args.max_stones + 1):
        best = -1.0
        best_take = None
        for take in range(1, args.max_take + 1):
            if take > stones:
                continue
            next_stones = stones - take
            if next_stones == 0:
                v = -1.0
            else:
                v = -values[next_stones]
            if v > best:
                best = v
                best_take = take
        print(f"{stones:>6} | {values[stones]:+.3f} | {best_take}")

    print(f"solve_ms: {elapsed_ms:.3f}")


if __name__ == '__main__':
    main()
