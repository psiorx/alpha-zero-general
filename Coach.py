import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


def compute_entropy(policy):
    """Compute entropy of a probability distribution."""
    policy = np.clip(policy, 1e-10, 1.0)
    return -np.sum(policy * np.log(policy))


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        if hasattr(self.nnet, "args") and hasattr(self.args, "minimal_logging"):
            self.nnet.args.minimal_logging = self.args.minimal_logging
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        if hasattr(self.pnet, "args") and hasattr(self.args, "minimal_logging"):
            self.pnet.args.minimal_logging = self.args.minimal_logging
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        
        # TensorBoard logging
        if SummaryWriter is not None:
            run_name = f"azg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=os.path.join("runs", "connect4_ref", run_name))
        else:
            self.writer = None
        
        # Stats tracking
        self.total_games = 0
        self.total_moves = 0
        self.p1_wins = 0
        self.p2_wins = 0  # player -1 in this implementation
        self.draws_count = 0
        self.recent_entropies = deque(maxlen=100)
        self.recent_mcts_depths = deque(maxlen=100)
        self.recent_mcts_nodes = deque(maxlen=100)
        self.start_time = time.time()
        self.elo_new = 1500.0
        self.elo_old = 1500.0

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
            stats: dictionary with game statistics
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        policy_entropies = []
        mcts_depths = []
        mcts_nodes = []

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            
            # Track MCTS stats
            mcts_nodes.append(len(self.mcts.Ns))
            # Estimate depth from number of states visited
            mcts_depths.append(len(self.mcts.Ns))
            
            # Track policy entropy
            policy_entropies.append(compute_entropy(np.array(pi)))
            
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                # Determine winner
                is_draw = abs(r) < 0.5  # Draw returns small value like 1e-4
                if is_draw:
                    winner = 0
                else:
                    # r is from perspective of curPlayer (who would move next)
                    # If r > 0, curPlayer won (but they didn't make last move)
                    # If r < 0, curPlayer lost (opponent who just moved won)
                    winner = -self.curPlayer if r < 0 else self.curPlayer
                
                stats = {
                    "winner": winner,
                    "is_draw": is_draw,
                    "first_player_won": winner == 1,
                    "avg_entropy": float(np.mean(policy_entropies)) if policy_entropies else 0,
                    "avg_mcts_nodes": float(np.mean(mcts_nodes)) if mcts_nodes else 0,
                    "game_length": episodeStep,
                }
                
                examples = [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
                return examples, stats

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            if not self.args.minimal_logging:
                log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                iter_entropies = []
                iter_mcts_nodes = []
                iter_game_lengths = []

                use_tqdm = (not self.args.minimal_logging) and tqdm is not None
                loop = tqdm(range(self.args.numEps), desc="Self Play") if use_tqdm else range(self.args.numEps)
                if self.args.minimal_logging:
                    print(f'Iter {i} self-play start')
                for _ in loop:
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    examples, stats = self.executeEpisode()
                    iterationTrainExamples += examples
                    
                    # Update stats
                    self.total_games += 1
                    self.total_moves += stats["game_length"]
                    if stats["is_draw"]:
                        self.draws_count += 1
                    elif stats["winner"] == 1:
                        self.p1_wins += 1
                    else:
                        self.p2_wins += 1
                    
                    iter_entropies.append(stats["avg_entropy"])
                    iter_mcts_nodes.append(stats["avg_mcts_nodes"])
                    iter_game_lengths.append(stats["game_length"])
                    self.recent_entropies.append(stats["avg_entropy"])
                    self.recent_mcts_nodes.append(stats["avg_mcts_nodes"])
                
                # Log self-play stats to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("self_play/games_total", self.total_games, i)
                    self.writer.add_scalar("self_play/moves_total", self.total_moves, i)
                    self.writer.add_scalar("self_play/avg_game_length", float(np.mean(iter_game_lengths)), i)
                    self.writer.add_scalar("self_play/policy_entropy", float(np.mean(iter_entropies)), i)
                    self.writer.add_scalar("self_play/avg_policy_entropy", float(np.mean(self.recent_entropies)), i)
                    self.writer.add_scalar("self_play/mcts_nodes", float(np.mean(iter_mcts_nodes)), i)
                    self.writer.add_scalar("self_play/exp_pool_size", len(iterationTrainExamples), i)
                    
                    # First-player advantage
                    total_decisive = self.p1_wins + self.p2_wins
                    if total_decisive > 0:
                        self.writer.add_scalar("self_play/p1_win_rate", 100.0 * self.p1_wins / total_decisive, i)
                    total_games_tracked = self.p1_wins + self.p2_wins + self.draws_count
                    if total_games_tracked > 0:
                        self.writer.add_scalar("self_play/draw_rate", 100.0 * self.draws_count / total_games_tracked, i)
                    
                    elapsed = time.time() - self.start_time
                    self.writer.add_scalar("self_play/games_per_sec", self.total_games / max(1, elapsed), i)
                
                if self.args.minimal_logging:
                    print(f'Iter {i} self-play end: examples={len(iterationTrainExamples)}')

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                if not self.args.minimal_logging:
                    log.warning(
                        f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            losses = self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            if self.args.minimal_logging:
                if losses:
                    (start_pi, start_v), (end_pi, end_v) = losses
                    print(f'Iter {i} train start: loss_pi={start_pi:.6f} loss_v={start_v:.6f}')
                    print(f'Iter {i} train end: loss_pi={end_pi:.6f} loss_v={end_v:.6f}')
                else:
                    print(f'Iter {i} train start: loss_pi=nan loss_v=nan')
                    print(f'Iter {i} train end: loss_pi=nan loss_v=nan')
            elif not self.args.minimal_logging:
                log.info('PITTING AGAINST PREVIOUS VERSION')
            
            # Log training losses to TensorBoard
            if self.writer is not None and losses:
                (start_pi, start_v), (end_pi, end_v) = losses
                self.writer.add_scalar("loss/policy", end_pi, i)
                self.writer.add_scalar("loss/value", end_v, i)
                self.writer.add_scalar("loss/policy_start", start_pi, i)
                self.writer.add_scalar("loss/value_start", start_v, i)
                self.writer.add_scalar("loss/policy_end", end_pi, i)
                self.writer.add_scalar("loss/value_end", end_v, i)
                # Log gradient norm if available
                if hasattr(self.nnet, 'nnet') and hasattr(self.nnet, 'last_grad_norm'):
                    self.writer.add_scalar("training/grad_norm", self.nnet.last_grad_norm, i)
                self.writer.flush()
            
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, use_tqdm=not self.args.minimal_logging)

            if self.args.minimal_logging:
                decisive = max(1, pwins + nwins)
                win_rate = 100.0 * nwins / float(decisive)
                print(f'Iter {i} duel: new={nwins} prev={pwins} draws={draws} win_rate={win_rate:.1f}%')
            else:
                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            
            # Update Elo ratings
            decisive = pwins + nwins
            if decisive > 0:
                actual_score = nwins / decisive
                expected = 1.0 / (1.0 + 10 ** ((self.elo_old - self.elo_new) / 400.0))
                K = 32
                delta = K * (actual_score - expected)
                self.elo_new += delta
                self.elo_old -= delta
            
            # Log duel stats to TensorBoard
            if self.writer is not None:
                total_duel_games = pwins + nwins + draws
                if total_duel_games > 0:
                    self.writer.add_scalar("duel/win_rate", 100.0 * nwins / max(1, decisive), i)
                    self.writer.add_scalar("duel/draw_rate", 100.0 * draws / total_duel_games, i)
                    self.writer.add_scalar("duel/wins", nwins, i)
                    self.writer.add_scalar("duel/losses", pwins, i)
                    self.writer.add_scalar("duel/draws", draws, i)
                self.writer.add_scalar("elo/new", self.elo_new, i)
                self.writer.add_scalar("elo/old", self.elo_old, i)
                self.writer.flush()
            
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                if self.args.minimal_logging:
                    print(f'Iter {i} result: rejected')
                else:
                    log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                if self.args.minimal_logging:
                    print(f'Iter {i} result: accepted')
                else:
                    log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
