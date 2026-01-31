class MDP:
    """
    Minimal finite MDP interface for LP solving.

    States and actions can be any hashable types.
    Rewards are from the perspective of the acting agent.
    """

    @property
    def discount(self):
        return 1.0

    def states(self):
        """Return an iterable of all states in the MDP."""
        raise NotImplementedError

    def actions(self, state):
        """Return an iterable of valid actions at state."""
        raise NotImplementedError

    def transitions(self, state, action):
        """
        Return a list of (probability, next_state, reward) tuples.
        Probabilities should sum to 1 for each (state, action).
        """
        raise NotImplementedError

    def is_terminal(self, state):
        """Return True if state is terminal (no future reward)."""
        raise NotImplementedError

    def is_max_state(self, state):
        """
        Return True if the agent chooses actions to maximize value at this state.
        For adversarial turn-based games, return False for the opponent's turn.
        """
        return True

    def state_relevance(self, states):
        """
        Optional: return a dict {state: weight} or list aligned to states.
        Used as the LP objective weights. Default is uniform.
        """
        return None

    def value_bounds(self):
        """
        Optional: return (min_value, max_value) bounds for V(s).
        Defaults to (-1, 1) for bounded episodic returns.
        """
        return (-1.0, 1.0)
