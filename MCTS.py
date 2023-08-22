import copy
import logging

import math
import numpy as np

import sys

EPS = 1e-8
sys.setrecursionlimit(1000000)
log = logging.getLogger(__name__)


class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.step = 0

    def getActionProb(self, canonical_board, considered_states, temp, cur_player, episode_step):
        numMCTS = self.game.getActiveAction(canonical_board).reshape(-1).tolist().count(0)
        for i in range(numMCTS):
            log.info(f'Simulation: {i}')
            self.step = episode_step
            self.search(copy.deepcopy(canonical_board), copy.deepcopy(considered_states), cur_player)

        canonical_board_state = self.game.getState(canonical_board)
        s = self.game.stringRepresentation(canonical_board_state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize() * np.max(self.game.getInitUnitNumber()))]

        if np.max(counts) == 0:
            return counts

        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        if not temp:
            probs = [x ** 2 for x in probs]
            probs_sum = float(sum(probs))
            probs = [x / probs_sum for x in probs]
        return probs

    def updateSA(self, s, a, v):
        if a > 0:
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v * (1 - math.tanh(self.step * self.args["stepforce"]))) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v * (1 - math.tanh(self.step * self.args["stepforce"]))
                self.Nsa[(s, a)] = 1
            self.Ns[s] += 1

    def search(self, canonical_board, considered_states, cur_player):
        canonical_board_state = self.game.getState(canonical_board)
        s = self.game.stringRepresentation(canonical_board_state)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonical_board)
        if self.Es[s] != 0:
            return self.Es[s], cur_player

        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonical_board_state)
            self.Ps[s], v = self.Ps[s][0] + EPS, v[0]
            valids = self.game.getValidActions(canonical_board)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] = self.Ps[s] / sum_Ps_s if sum_Ps_s > 0 else self.Ps[s]
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v, cur_player

        best_act = -1
        next_player = cur_player
        if np.sum(self.Ps[s]) > 0:
            valids = self.Vs[s]
            cur_best = -float('inf')
            for a in range(len(valids)):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        u = self.Qsa[(s, a)] + self.args["cpuct"] * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) / (1 + self.Nsa[(s, a)])
                    else:
                        u = self.args["cpuct"] * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    if u > cur_best:
                        cur_best = u
                        best_act = a
            canonical_board = self.game.getNextState(canonical_board, best_act)
            self.step += 1
        else:
            self.game.passTurn()
            next_player = 2 if cur_player == 1 else 1
            canonical_board = self.game.getCanonicalForm(canonical_board)

        canonical_board_state = self.game.getState(canonical_board)
        s_next = self.game.stringRepresentation(canonical_board_state)
        if s_next in considered_states:
            self.updateSA(s, best_act, 0)
            return 0, cur_player
        considered_states.append(s)

        v, end_player = self.search(canonical_board, considered_states, next_player)

        self.updateSA(s, best_act, v * (-1) ** (end_player != cur_player))
        return v, end_player
