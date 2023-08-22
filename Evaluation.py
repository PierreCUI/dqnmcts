import coloredlogs
import logging

import numpy as np

from MCTS import MCTS
from zodiac.game import Game
from zodiac.game_net import Net as nn

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = {
    'cpuct': 1.0,
    'stepforce': 0.01,

    'checkpoint': './temp/',
    'loadFile': ('best.pth.tar', -1),
}


class Player:
    Human = 1
    Computer = 2


def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def multi_to_single_index(position, shape):
    index = 0
    multiplier = 1
    for i in range(len(shape)-1, -1, -1):
        index += position[i] * multiplier
        multiplier *= shape[i]
    return index


def playGame(game, net, board, considered_states, step, cur_player):
    r = game.getGameEnded(board)
    if r != 0:
        log.info(f'Game End Steps: {step}')
        return r * (-1) ** (cur_player != Player.Human), step

    while True:
        r = game.getGameEnded(board)
        if r != 0:
            log.info(f'Game End Steps: {step}')
            return r * (-1) ** (cur_player != Player.Human), step

        board_state = game.getState(board)
        s = game.stringRepresentation(board_state)
        if s not in considered_states:
            considered_states.append(s)

        if cur_player == Player.Computer:
            log.info(f'Evaluation Deduction: {step}')
            pi = net.getActionProb(board, considered_states, 0, cur_player, step)
            if np.sum(pi) > 0:
                log.info(f'Evaluation Action: {step}')
                action = np.random.choice(len(pi), p=pi)
                board = game.getNextState(board, action)
                step += 1
            else:
                log.info(f'Evaluation No Action: {step}')
                break
        else:
            valids = game.getValidActions(board)
            if np.sum(valids) > 0:
                while True:
                    action = input("Please enter player, skill, row and col numbers separated by spaces: ")
                    action = action.split()
                    try:
                        action = [int(x) for x in action]
                        action[0] -= 1
                    except ValueError:
                        log.error(f'Invalid Input')
                        continue

                    action = multi_to_single_index(action, (np.max(game.getInitUnitNumber()), len(flatten_list(game.skill_actions)), game.map.size, game.map.size))
                    if valids[action] > 0:
                        board = game.getNextState(board, action)
                        step += 1
                        break
                    else:
                        log.error(f'Invalid Action')
                        continue
            else:
                break

    board = game.getCanonicalForm(board)
    cur_player = Player.Human if cur_player == Player.Computer else Player.Computer
    return playGame(game, net, board, considered_states, step, cur_player)


def main():
    log.info('Loading %s...', Game.__name__)
    game = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(game)

    log.info('Loading checkpoint "%s/%s"...', args["checkpoint"], args["loadFile"][0])
    if not nnet.load_checkpoint(args["checkpoint"], args["loadFile"][0]):
        exit(-1)
    computerPlayer = MCTS(game, nnet, args)

    board = game.getInitBoard()
    r, s = playGame(game, computerPlayer, board, [], 0, Player.Human)

    if r == 1:
        log.info(f'Game Result: Human Wins, Game Steps: {s}')
    else:
        log.info(f'Game Result: Computer Wins, Game Steps: {s}')


if __name__ == "__main__":
    main()
