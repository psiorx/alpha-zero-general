import argparse

import numpy as np


def bellman_backup(values, max_stones, max_take):
    new_values = values.copy()
    for stones in range(1, max_stones + 1):
        best = -1.0
        for take in range(1, max_take + 1):
            if take > stones:
                continue
            next_stones = stones - take
            if next_stones == 0:
                v = -1.0  # taking the last stone loses
            else:
                v = -values[next_stones]
            if v > best:
                best = v
        new_values[stones] = best
    return new_values


def main():
    parser = argparse.ArgumentParser(description="Solve misere Nim via value iteration.")
    parser.add_argument('--max_stones', type=int, default=21, help='Number of stones')
    parser.add_argument('--max_take', type=int, default=3, help='Max stones to take')
    parser.add_argument('--iters', type=int, default=50, help='Number of Bellman backups')
    args = parser.parse_args()

    values = np.zeros(args.max_stones + 1, dtype=np.float64)

    for _ in range(args.iters):
        values = bellman_backup(values, args.max_stones, args.max_take)

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
        print(f"{stones:>6} | {best:+.3f} | {best_take}")


if __name__ == '__main__':
    main()
