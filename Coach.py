import copy
import logging

import os

import re
import numpy as np
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from joblib import Parallel, delayed

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.args = args
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamples = []
        self.curstep = float("inf")
        self.skipFirst = self.args["skipFirst"]

    def executeEpisode(self):
        canonical_board = self.game.getInitBoard()
        canonical_board_state = self.game.getState(canonical_board)
        s = self.game.stringRepresentation(canonical_board_state)

        cur_player = 1
        trainExamples = []
        considered_states = [s]
        episodes = [0]

        while True:
            episode_step = episodes[considered_states.index(s)] + 1
            r = self.game.getGameEnded(canonical_board)
            if r != 0:
                log.info(f'Game ended. r = {r * (-1) ** (cur_player != 1)}')
                return [(x[0], x[1], r * (-1) ** (x[2] != cur_player)) for x in trainExamples]

            temp = int(episode_step < self.args["tempThreshold"])
            log.info(f'Deduction: {episode_step}')
            pi = np.array(self.mcts.getActionProb(canonical_board, considered_states, temp, cur_player, episode_step))
            sym = copy.deepcopy(self.game.getSymmetries(canonical_board, pi))
            for b, p in sym:
                trainExamples.append([self.game.getState(b), p, cur_player])

            if np.sum(pi) > 0:
                log.info(f'Action: {episode_step}')
                action = np.random.choice(len(pi), p=pi)
                canonical_board = self.game.getNextState(canonical_board, action)
            else:
                episode_step -= 1
                log.info(f'No Action: {episode_step}')
                cur_player = 2 if cur_player == 1 else 1
                canonical_board = self.game.getCanonicalForm(canonical_board)

            canonical_board_state = self.game.getState(canonical_board)
            s = self.game.stringRepresentation(canonical_board_state)
            if s not in considered_states:
                considered_states.append(s)
                episodes.append(episode_step)

    def learn(self, start):
        maxlenOfData = self.args["maxlenOfQueue"] * self.args["maxlenOfPacks"]

        for i in range(start, self.args["numIters"] + 1):
            log.info(f'Starting Iter #{i} ...')

            iterationTrainExamples = deque([], maxlen=self.args["maxlenOfQueue"])
            if not self.skipFirst:
                while len(iterationTrainExamples) < self.args["maxlenOfQueue"]:
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    results = Parallel(n_jobs=self.args["nJobs"])(delayed(self.executeEpisode)() for _ in range(self.args["nRuns"]))
                    iterationTrainExamples += [item for sublist in results for item in sublist]
                    del results
                    log.info(f'len(iterationTrainExamples) = {len(iterationTrainExamples)}')
            else:
                log.info(f'Skipping Train Examples of Iter #{i} ...')

            self.mcts = None
            self.loadTrainExamples()
            self.trainExamples += list(iterationTrainExamples)
            shuffle(self.trainExamples)
            self.trainExamples = self.trainExamples[:maxlenOfData]
            del iterationTrainExamples
            log.info(f'len(trainExamples) = {len(self.trainExamples)}')
            self.saveTrainExamples(i - 1)

            self.nnet.save_checkpoint(folder=self.args["checkpoint"], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args["checkpoint"], filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            self.nnet.train(self.trainExamples)
            self.trainExamples.clear()
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.game, pmcts, nmcts, self.args["arenaCompare"])
            pwins, nwins, step = arena.playGames()
            del pmcts, nmcts, arena

            log.info('NEW/PREV WINS, STEPS : %d / %d / %d' % (nwins, pwins, step))
            if nwins / (pwins + nwins) > self.args["updateThreshold"] or nwins / (pwins + nwins) > 0.5 and step < self.curstep * 0.9:
                log.info('ACCEPTING NEW MODEL')
                self.curstep = step
                self.nnet.save_checkpoint(folder=self.args["checkpoint"], filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args["checkpoint"], filename='best.pth.tar')
                with open(self.args["checkpoint"] + "steps.dat", "w") as file:
                    file.write(str(self.curstep))
            else:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args["checkpoint"], filename='temp.pth.tar')
            del pwins, nwins, step

            self.skipFirst = False

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args["checkpoint"]
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamples)

    def extract_number(self, filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return 0

    def loadTrainExamples(self, load=True):
        file_names = sorted([file for file in os.listdir(self.args["checkpoint"]) if file.endswith(".examples")])
        file_names = sorted(file_names, key=self.extract_number)
        if not file_names:
            return 0
        load_file = file_names[-1]

        examples_file = os.path.join(self.args["checkpoint"], load_file)
        if load:
            if not os.path.isfile(examples_file):
                log.warning(f'File "{examples_file}" with trainExamples not found!')
                return 0
            else:
                log.info("File with trainExamples found. Loading it...")
                with open(examples_file, "rb") as f:
                    self.trainExamples = list(Unpickler(f).load())
                log.info('Loading done!')

        try:
            with open(self.args["checkpoint"] + "steps.dat", "r") as file:
                self.curstep = float(file.read())
        except Exception:
            self.curstep = float("inf")

        return self.extract_number(load_file) + 1
