import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mdp.solve_lp import solve_mdp_lp
from nim.NimMDP import NimMDP


def main():
    parser = argparse.ArgumentParser(description="Solve Nim via generic MDP LP solver.")
    parser.add_argument('--max_stones', type=int, default=21)
    parser.add_argument('--max_take', type=int, default=3)
    args = parser.parse_args()

    mdp = NimMDP(max_stones=args.max_stones, max_take=args.max_take)
    values = solve_mdp_lp(mdp)

    print("stones | value | optimal_take")
    print("-------+-------+-------------")
    for stones in range(1, args.max_stones + 1):
        state = (stones, 1)
        v = values[state]
        best = -1.0
        best_take = None
        for take in mdp.actions(state):
            q = None
            for _p, next_state, reward in mdp.transitions(state, take):
                q = reward + values[next_state]
            if q is None:
                continue
            if q > best:
                best = q
                best_take = take
        print(f"{stones:>6} | {v:+.3f} | {best_take}")


if __name__ == '__main__':
    main()
