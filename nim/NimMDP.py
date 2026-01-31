from mdp.MDP import MDP


class NimMDP(MDP):
    def __init__(self, max_stones=21, max_take=3):
        self.max_stones = int(max_stones)
        self.max_take = int(max_take)

    def states(self):
        for stones in range(0, self.max_stones + 1):
            for player in (1, -1):
                yield (stones, player)

    def actions(self, state):
        stones, _player = state
        if stones <= 0:
            return []
        return [take for take in range(1, self.max_take + 1) if take <= stones]

    def transitions(self, state, action):
        stones, player = state
        next_stones = stones - action
        next_state = (next_stones, -player)
        if next_stones == 0:
            reward = -float(player)
        else:
            reward = 0.0
        return [(1.0, next_state, reward)]

    def is_terminal(self, state):
        stones, _player = state
        return stones == 0

    def is_max_state(self, state):
        _stones, player = state
        return player == 1
