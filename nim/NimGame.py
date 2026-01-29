import sys
import numpy as np

sys.path.append('..')
from Game import Game


class NimGame(Game):
    """
    Nim with 21 stones, remove 1-3 each turn, taking the last stone loses.
    """

    def __init__(self, max_stones=21, max_take=3):
        Game.__init__(self)
        self.max_stones = int(max_stones)
        self.max_take = int(max_take)

    def getInitBoard(self):
        return np.array([[self.max_stones]], dtype=np.int8)

    def getBoardSize(self):
        return (1, 1)

    def getActionSize(self):
        return self.max_take

    def getNextState(self, board, player, action):
        stones = int(board[0, 0])
        sign = 1 if stones >= 0 else -1
        stones = abs(stones)
        take = int(action) + 1
        next_stones = stones - take
        next_board = np.array([[sign * next_stones]], dtype=np.int8)
        return next_board, -player

    def getValidMoves(self, board, player):
        stones = abs(int(board[0, 0]))
        valid = np.zeros(self.getActionSize(), dtype=np.int8)
        for i in range(self.max_take):
            if stones >= i + 1:
                valid[i] = 1
        return valid

    def getGameEnded(self, board, player):
        stones = abs(int(board[0, 0]))
        if stones <= 0:
            # No stones left to play: previous player took the last stone and lost.
            return 1
        return 0

    def getCanonicalForm(self, board, player):
        return board * player

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    def getMaxStones(self):
        return self.max_stones

    @staticmethod
    def display(board):
        stones = abs(int(board[0, 0]))
        print(f"Stones remaining: {stones}")
