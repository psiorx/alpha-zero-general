import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from MCTS import MCTS
from nim.NimGame import NimGame
from utils import dotdict


class UniformNNet:
    def __init__(self, game):
        self.action_size = game.getActionSize()

    def predict(self, board):
        pi = np.ones(self.action_size, dtype=np.float64) / self.action_size
        v = np.array([0.0], dtype=np.float64)
        return pi, v


def optimal_winning_action(stones, max_take=3):
    # leave opponent with 1 mod 4 stones
    r = (stones - 1) % 4
    if r == 0:
        return None
    return r - 1  # convert take count to action index


def evaluate_sims(game, sims, temp):
    mcts_args = dotdict({'numMCTSSims': sims, 'cpuct': 1.0, 'debug': False})
    net = UniformNNet(game)
    details = []

    for stones in range(1, game.getMaxStones() + 1):
        board = np.array([[stones]], dtype=np.int8)
        mcts = MCTS(game, net, mcts_args)
        pi = mcts.getActionProb(board, temp=temp)
        s = game.stringRepresentation(board)
        q_vals = [float(mcts.Qsa.get((s, a), 0.0)) for a in range(game.getActionSize())]
        valids = game.getValidMoves(board, 1)

        # best action/value by MCTS
        best_action = int(np.argmax(pi))
        valid_qs = [q_vals[a] for a in range(game.getActionSize()) if valids[a]]
        best_q = float(np.max(valid_qs)) if valid_qs else 0.0

        # expected value by MCTS policy
        mcts_v = float(np.dot(np.array(pi, dtype=np.float64), np.array(q_vals, dtype=np.float64)))
        outcome = "win" if best_q > 0 else "lose"
        details.append((stones, best_action, best_q, mcts_v, outcome))

        # ground truth
        losing = (stones % 4 == 1)
        winning_action = optimal_winning_action(stones, game.max_take)

        # sign check: best Q should reflect win/loss
        if losing:
            if best_q >= 0:
                return False, stones, best_action, best_q, mcts_v
        else:
            if best_q <= 0:
                return False, stones, best_action, best_q, mcts_v
            if winning_action is not None and best_action != winning_action:
                return False, stones, best_action, best_q, mcts_v

    return True, details, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Find minimum MCTS sims to solve Nim (uniform priors).")
    parser.add_argument('--max_sims', type=int, default=2000, help='Maximum sims to try')
    parser.add_argument('--temp', type=float, default=0, help='Temperature for MCTS policy (default 0)')
    parser.add_argument('--step', type=int, default=1, help='Step size for sims')
    args = parser.parse_args()

    game = NimGame()

    for sims in range(1, args.max_sims + 1, args.step):
        ok, details, stones, action, q = evaluate_sims(game, sims, args.temp)
        if ok:
            print(f"min_sims={sims}")
            for stones, best_action, best_q, mcts_v, outcome in details:
                take = best_action + 1
                print(f"stones={stones:>2} take={take} best_q={best_q:+.3f} value={mcts_v:+.3f} outcome={outcome}")
            return
        if sims % 50 == 0:
            print(f"sims={sims} failed at stones={stones} action={action} best_q={q:.3f}")

    print("No solution found up to max_sims.")


if __name__ == '__main__':
    main()
