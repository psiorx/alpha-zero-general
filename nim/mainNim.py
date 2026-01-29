import logging
import os
import sys

import coloredlogs

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Coach import Coach
from nim.NimGame import NimGame as Game
from nim.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 100,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.4,
    'maxlenOfQueue': 1000,
    'numMCTSSims': 25,
    'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'minimal_logging': True,
    'debug': False,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process')
    c.learn()


if __name__ == "__main__":
    main()
