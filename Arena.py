import logging

import copy
import numpy as np

from joblib import Parallel, delayed

log = logging.getLogger(__name__)

args = {
    'nJobs': -1,
    'nRuns': 10,
}


class Arena:
    def __init__(self, game, playerOld, playerNew, compareTimes):
        self.game = game
        self.playerOld = playerOld
        self.playerNew = playerNew
        self.compareTimes = compareTimes

    def playGame(self, player1, player2, board, considered_states=None, step=1, cur_player=1):
        if considered_states is None:
            considered_states = []

        r = self.game.getGameEnded(board)
        if r != 0:
            log.info(f'Game End Steps: {step}')
            return r * (-1) ** (cur_player != 1), step

        while True:
            r = self.game.getGameEnded(board)
            if r != 0:
                log.info(f'Game End Steps: {step}')
                return r * (-1) ** (cur_player != 1), step

            board_state = self.game.getState(board)
            s = self.game.stringRepresentation(board_state)
            if s not in considered_states:
                considered_states.append(s)

            log.info(f'Arena Deduction: {step}')
            pi = player1.getActionProb(board, considered_states, 0, cur_player, step)
            if np.sum(pi) > 0:
                log.info(f'Arena Action: {step}')
                action = np.random.choice(len(pi), p=pi)
                board = self.game.getNextState(board, action)
                step += 1
            else:
                log.info(f'Arena No Action: {step}')
                cur_player = 2 if cur_player == 1 else 1
                board, curActedState = self.game.getCanonicalForm(board)
                break

        return self.playGame(player2, player1, board, considered_states, step, cur_player)

    def playGames(self):
        log.info(f'Arena Play')

        oldPlayerWin = 0
        newPlayerWin = 0
        step = 0

        results = []
        while len(results) < self.compareTimes // 2:
            board = self.game.getInitBoard()
            result = Parallel(n_jobs=args["nJobs"])(delayed(self.playGame)(copy.deepcopy(self.playerOld), copy.deepcopy(self.playerNew), copy.deepcopy(board)) for _ in range(args["nRuns"]))

            for r, s in result:
                results.append(r)
                log.info(f'Part I Game Number: {len(results)}, Game End: {r}')
                if r == 1:
                    oldPlayerWin += 1
                elif r == -1:
                    newPlayerWin += 1
                step += s

        results = []
        while len(results) < self.compareTimes // 2:
            board = self.game.getInitBoard()
            result = Parallel(n_jobs=args["nJobs"])(delayed(self.playGame)(copy.deepcopy(self.playerNew), copy.deepcopy(self.playerOld), copy.deepcopy(board)) for _ in range(args["nRuns"]))

            for r, s in result:
                results.append(r)
                log.info(f'Part II Game Number: {len(results)}, Game End: {r}')
                if r == 1:
                    newPlayerWin += 1
                elif r == -1:
                    oldPlayerWin += 1
                step += s

        return oldPlayerWin, newPlayerWin, step
