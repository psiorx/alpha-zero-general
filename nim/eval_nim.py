import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from MCTS import MCTS
from nim.NimGame import NimGame
from nim.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict


def format_pi(pi):
    return "[" + ", ".join(f"{p:.3f}" for p in pi) + "]"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nim policy/value across states with MCTS.")
    parser.add_argument('--sims', type=int, default=200, help='MCTS simulations per state')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path or filename (default: none)')
    parser.add_argument('--checkpoint_dir', type=str, default='./temp', help='Checkpoint directory (ignored if --checkpoint is a path)')
    parser.add_argument('--max_stones', type=int, default=21, help='Maximum stones to evaluate')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for MCTS policy')
    args = parser.parse_args()

    game = NimGame(max_stones=args.max_stones)
    nnet = NNet(game)

    if args.checkpoint:
        if os.path.isabs(args.checkpoint) or os.path.exists(args.checkpoint):
            checkpoint_dir = os.path.dirname(args.checkpoint) or '.'
            checkpoint_file = os.path.basename(args.checkpoint)
        else:
            checkpoint_dir = args.checkpoint_dir
            checkpoint_file = args.checkpoint
        nnet.load_checkpoint(checkpoint_dir, checkpoint_file)

    mcts_args = dotdict({
        'numMCTSSims': args.sims,
        'cpuct': 1.0,
        'debug': False,
    })
    mcts = MCTS(game, nnet, mcts_args)

    print(f"sims={args.sims} temp={args.temp} max_stones={args.max_stones}")
    print("stones | value (mcts) | policy (mcts)       | q (per action)")
    print("-------+--------------+--------------------+---------------------------")

    for stones in range(1, args.max_stones + 1):
        board = np.array([[stones]], dtype=np.int8)
        # fresh MCTS per state to avoid tree carryover
        mcts = MCTS(game, nnet, mcts_args)
        mcts_pi = mcts.getActionProb(board, temp=args.temp)
        s = game.stringRepresentation(board)
        q_vals = []
        for a in range(game.getActionSize()):
            q_raw = mcts.Qsa.get((s, a), 0.0)
            if isinstance(q_raw, np.ndarray):
                q_val = float(q_raw.reshape(-1)[0])
            else:
                q_val = float(q_raw)
            q_vals.append(q_val)
        mcts_v = float(np.dot(np.array(mcts_pi, dtype=np.float64), np.array(q_vals, dtype=np.float64)))
        print(f"{stones:>6} | {mcts_v:>12.4f} | {format_pi(mcts_pi)} | {format_pi(q_vals)}")


if __name__ == '__main__':
    main()
