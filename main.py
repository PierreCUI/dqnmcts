import logging
import coloredlogs

import os

from Coach import Coach
from zodiac.game import Game
from zodiac.game_net import Net as nn

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = {
    'numIters': 1000,
    'tempThreshold': 60,

    'arenaCompare': 20,
    'updateThreshold': 0.6,

    'maxlenOfQueue': 10000,
    'maxlenOfPacks': 5,

    'cpuct': 1.0,
    'stepforce': 0.01,

    'checkpoint': './temp/',
    'loadFile': 'best.pth.tar',
    'skipFirst': False,

    'nJobs': -1,
    'nRuns': 10,
}


def main(): 
    if not os.path.exists(args["checkpoint"]):
        os.makedirs(args["checkpoint"])

    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading checkpoint "%s/%s"...', args["checkpoint"], args["loadFile"])
    nnet.load_checkpoint(args["checkpoint"], args["loadFile"])

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    log.info("Loading 'trainExamples' from file...")
    start = c.loadTrainExamples(False)

    log.info('Starting the learning process')
    c.learn(start + int(not args["skipFirst"]))


if __name__ == "__main__":
    main()
