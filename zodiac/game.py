import copy
from enum import Enum
import numpy as np

import logging
import sys

log = logging.getLogger(__name__)
sys.setrecursionlimit(1000000)


def contains(target_list, list_of_lists):
    for lst in list_of_lists:
        if np.array_equal(target_list, lst):
            return True
    return False


def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def onehot(enum_values):
    enum_class = type(enum_values[0])
    num_values = len(enum_class)
    onehot_list = [0] * num_values
    for enum_value in enum_values:
        onehot_list[enum_value.value] = 1
    return onehot_list


def onehots(enum_values_dict):
    onehotlist = []
    for enum_value in enum_values_dict.values():
        if isinstance(enum_value, Enum):
            onehotlist.append(onehot([enum_value]))
        elif isinstance(enum_value, list) and len(enum_value) and isinstance(enum_value[0], Enum):
            onehotlist.append(onehot(enum_value))
        else:
            onehotlist.append(enum_value)
    return onehotlist


class UnitType(Enum):
    VOID = 0
    ACTIVE = 1
    PASSIVE = 2
    EMPTY = -1
    WALL = -2
    UNKNOWN = -3
    LEVER = -4


class ActionType(Enum):
    MOVE = [0]
    ATTACK = [1]
    VITAL = [2]
    FIRE = [3]
    TELE = [4, 5]
    LEVER = [6]
    OPEN = [7]


class Id(Enum):
    VOID = 0
    USELESS = -1
    WAITING = -2


class SkillGroup(Enum):
    VOID = 0
    MOVE = 1
    ATTACK = 2
    BATTLE = 3
    SCENE = 4


class SkillType(Enum):
    VOID = 0
    IMMEDIATE = 1
    CONTINUE = 2


class BuffType(Enum):
    VOID = 0
    AFTERCALL = 1
    CONTINUE = 2


class Map:
    def __init__(self, size, empty_unit):
        self.size = size
        self.grid = np.zeros((self.size, self.size)).astype(int).tolist()
        for row in range(self.size):
            for col in range(self.size):
                empty = copy.deepcopy(empty_unit)
                empty.info["row"] = row
                empty.info["col"] = col
                self.grid[row][col] = empty
        self.grid = np.array(self.grid)

    def add_unit(self, unit):
        self.grid[unit.info["row"]][unit.info["col"]] = copy.deepcopy(unit)
        return unit

    def add_units(self, rows, cols, unit, id=1):
        for row, col in zip(rows, cols):
            u = copy.deepcopy(unit)
            u.info["row"] = row
            u.info["col"] = col
            if u.info["id"] == Id.WAITING:
                u.info["id"] = id
                id += 1
            self.grid[row][col] = u
        return id


class Unit:
    def __init__(self, info, basic, skill, buff):
        self.info = info
        self.basic = basic
        self.skill = skill
        self.buff = buff

    def get_all_positions(self, board, types, max_range):
        positions = []
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row][col].basic["type"] in types and abs(row - self.info["row"]) + abs(col - self.info["col"]) <= max_range:
                    positions.append([row, col])
        return np.array(positions)

    def move(self, target_position, board, empty_unit):
        empty = copy.deepcopy(empty_unit)
        empty.info["row"] = target_position[0]
        empty.info["col"] = target_position[1]
        board[self.info["row"]][self.info["col"]] = copy.deepcopy(empty)

        self.info["row"], self.info["col"] = target_position[0], target_position[1]
        board[target_position[0]][target_position[1]] = copy.deepcopy(self)
        return board

    def attack(self, target_position, board, empty_unit, attack_point=None):
        if attack_point is None:
            attack_point = self.skill["attack"]["skill_point"]

        target = board[target_position[0]][target_position[1]]
        target.basic["hp"] -= max(attack_point - target.basic["defense"], 1)

        if target.basic["hp"] <= 0:
            empty = copy.deepcopy(empty_unit)
            empty.info["row"] = target_position[0]
            empty.info["col"] = target_position[1]
            board[target_position[0]][target_position[1]] = copy.deepcopy(empty)
        return board

    def vital(self, target_position, board, empty_unit):
        target = board[target_position[0]][target_position[1]]
        target.basic["hp"] -= max(self.skill["vital"]["skill_point"] - target.basic["defense"], 1)

        if target.basic["hp"] <= 0:
            empty = copy.deepcopy(empty_unit)
            empty.info["row"] = target_position[0]
            empty.info["col"] = target_position[1]
            board[target_position[0]][target_position[1]] = copy.deepcopy(empty)
        return board

    def fire(self, target_position, board, empty_unit):
        target = board[target_position[0]][target_position[1]]
        target.buff["fire"] = {"period": 0, "max_period": self.skill["fire"]["skill_point"], "buff_type": BuffType.AFTERCALL, "target": self, "skill_point": self.skill["fire"]["skill_point"]}
        return self.attack(target_position, board, empty_unit)

    def tele(self, target_position, board, empty_unit, period):
        if period == 0:
            target = board[target_position[0]][target_position[1]]
            self.buff["tele"] = {"period": 1, "max_period": 2, "buff_type": BuffType.CONTINUE, "target": target, "skill_point": self.skill["tele"]["skill_point"]}
        elif period == 1:
            target = self.buff["tele"]["target"]
            board = target.move(target_position, board, empty_unit)
            board = target.attack(target_position, board, empty_unit, self.buff["tele"]["skill_point"])
            self.buff.pop("tele")
        return board

    def open(self, open_position, board, empty_unit):
        empty = copy.deepcopy(empty_unit)
        empty.info["row"] = open_position[0]
        empty.info["col"] = open_position[1]
        board[open_position[0]][open_position[1]] = copy.deepcopy(empty)
        return board

    def act(self, board, action, empty_unit, wall_unit):
        action_type = action // (board.shape[0] * board.shape[1])
        action = action % (board.shape[0] * board.shape[1])
        target_position = [action // board.shape[1], action % board.shape[1]]

        # log.info(f"name: {self.info['name']}, player: {self.info['id']}, action_type: {action_type}, target_position: {target_position}")

        if action_type in ActionType.MOVE.value:
            self.skill["move"]["cd"] = self.skill["move"]["max_cd"]
            return self.move(target_position, board, empty_unit)
        elif action_type in ActionType.ATTACK.value:
            self.skill["attack"]["cd"] = self.skill["attack"]["max_cd"]
            return self.attack(target_position, board, empty_unit)
        elif action_type in ActionType.VITAL.value:
            self.skill["vital"]["cd"] = self.skill["vital"]["max_cd"]
            return self.vital(target_position, board, empty_unit)
        elif action_type in ActionType.FIRE.value:
            self.skill["fire"]["cd"] = self.skill["fire"]["max_cd"]
            return self.fire(target_position, board, empty_unit)
        elif action_type in ActionType.TELE.value:
            self.skill["tele"]["cd"] = self.skill["tele"]["max_cd"]
            return self.tele(target_position, board, empty_unit, ActionType.TELE.value.index(action_type))
        elif action_type in ActionType.LEVER.value:
            self.skill["lever"]["cd"] = self.skill["lever"]["max_cd"]
            open_position = board[target_position[0]][target_position[1]].skill["open"]["open_position"]
            board[target_position[0]][target_position[1]] = copy.deepcopy(wall_unit)
            return self.open(open_position, board, empty_unit)

        return board


class Game:
    def __init__(self):
        self.empty_skill  = {"cd": -1, "max_cd": -1, "skill_point": -1, "skill_range": -1, "open_position": [-1, -1], "target": UnitType.VOID, "skill_group": SkillGroup.VOID, "skill_type": SkillType.VOID}
        self.skills       = ["move", "attack", "vital", "fire", "tele", "lever", "open"]
        self.skill_actions = [item.value for item in ActionType]

        self.unknown    = Unit({"row": -2, "col": -2, "id": Id.USELESS, "name": "unknown"},
                               {"hp": -1, "defense": -1, "sight": -1, "type": UnitType.UNKNOWN}, {}, {})
        self.empty_unit = Unit({"row": -2, "col": -2, "id": Id.USELESS, "name": "empty"},
                               {"hp": -1, "defense": -1, "sight": -1, "type": UnitType.EMPTY},   {}, {})
        self.wall       = Unit({"row": -2, "col": -2, "id": Id.USELESS, "name": "wall"},
                               {"hp": -1, "defense": -1, "sight": -1, "type": UnitType.WALL},    {}, {})

        self.map = Map(16, self.empty_unit)
        self.map.add_units([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                           [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15], self.wall)

        L1 = Unit({"row": 11, "col": 10, "id": 1, "name": "L1"},
                  {"hp": 0, "defense": 0, "sight": 0, "type": UnitType.LEVER},
                  {"open": {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 0, "open_position": [10, 7], "target": UnitType.WALL, "skill_group": SkillGroup.SCENE, "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(L1)

        A = Unit({"row": 12, "col": 6, "id": 1, "name": "A"},
                 {"hp": 10, "defense": 4, "sight": 6, "type": UnitType.ACTIVE},
                 {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 2, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                  "attack": {"cd": 0, "max_cd": 1, "skill_point": 2, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE},
                  "lever" : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.LEVER,   "skill_group": SkillGroup.SCENE,  "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(A)
        B = Unit({"row": 12, "col": 7, "id": 2, "name": "B"},
                 {"hp": 8, "defense": 3, "sight": 6, "type": UnitType.ACTIVE},
                 {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 2, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                  "attack": {"cd": 0, "max_cd": 1, "skill_point": 3, "skill_range": 2, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE},
                  "vital" : {"cd": 0, "max_cd": 3, "skill_point": 5, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.IMMEDIATE},
                  "lever" : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.LEVER,   "skill_group": SkillGroup.SCENE,  "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(B)
        C = Unit({"row": 12, "col": 8, "id": 3, "name": "C"},
                 {"hp": 10, "defense": 4, "sight": 6, "type": UnitType.ACTIVE},
                 {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 2, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                  "attack": {"cd": 0, "max_cd": 1, "skill_point": 2, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE},
                  "tele"  : {"cd": 0, "max_cd": 2, "skill_point": 0, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.CONTINUE},
                  "lever" : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.LEVER,   "skill_group": SkillGroup.SCENE,  "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(C)
        X = Unit({"row": 14, "col": 6, "id": 4, "name": "X"},
                 {"hp": 6, "defense": 2, "sight": 6, "type": UnitType.ACTIVE},
                 {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                  "attack": {"cd": 0, "max_cd": 1, "skill_point": 6, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE},
                  "fire"  : {"cd": 0, "max_cd": 4, "skill_point": 3, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.IMMEDIATE},
                  "lever" : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.LEVER,   "skill_group": SkillGroup.SCENE,  "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(X)
        Y = Unit({"row": 14, "col": 7, "id": 5, "name": "Y"},
                 {"hp": 4, "defense": 2, "sight": 6, "type": UnitType.ACTIVE},
                 {"move"  : {"cd": 0, "max_cd": 1, "skill_point":  0, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                  "attack": {"cd": 0, "max_cd": 1, "skill_point":  8, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE},
                  "vital" : {"cd": 0, "max_cd": 2, "skill_point": 12, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.IMMEDIATE},
                  "fire"  : {"cd": 0, "max_cd": 4, "skill_point":  3, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.IMMEDIATE},
                  "lever" : {"cd": 0, "max_cd": 1, "skill_point":  0, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.LEVER,   "skill_group": SkillGroup.SCENE,  "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(Y)
        Z = Unit({"row": 14, "col": 8, "id": 6, "name": "Z"},
                 {"hp": 6, "defense": 2, "sight": 6, "type": UnitType.ACTIVE},
                 {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                  "attack": {"cd": 0, "max_cd": 1, "skill_point": 4, "skill_range": 8, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE},
                  "tele"  : {"cd": 0, "max_cd": 2, "skill_point": 0, "skill_range": 8, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.CONTINUE},
                  "fire"  : {"cd": 0, "max_cd": 4, "skill_point": 2, "skill_range": 8, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.IMMEDIATE},
                  "vital" : {"cd": 0, "max_cd": 2, "skill_point": 6, "skill_range": 8, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.BATTLE, "skill_type": SkillType.IMMEDIATE},
                  "lever" : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.LEVER,   "skill_group": SkillGroup.SCENE,  "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_unit(Z)

        E1 = Unit({"row": -2, "col": -2, "id": Id.WAITING, "name": "E1"},
                  {"hp": 20, "defense": 4, "sight": 6, "type": UnitType.PASSIVE},
                  {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 2, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                   "attack": {"cd": 0, "max_cd": 1, "skill_point": 1, "skill_range": 1, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE}}, {})
        id_E1 = self.map.add_units([7, 7, 7], [6, 7, 8], E1)
        E2 = Unit({"row": -2, "col": -2, "id": Id.WAITING, "name": "E2"},
                  {"hp": 10, "defense": 2, "sight": 6, "type": UnitType.PASSIVE},
                  {"move"  : {"cd": 0, "max_cd": 1, "skill_point": 0, "skill_range": 6, "open_position": [-1, -1], "target": UnitType.EMPTY,   "skill_group": SkillGroup.MOVE,   "skill_type": SkillType.IMMEDIATE},
                   "attack": {"cd": 0, "max_cd": 1, "skill_point": 3, "skill_range": 3, "open_position": [-1, -1], "target": UnitType.PASSIVE, "skill_group": SkillGroup.ATTACK, "skill_type": SkillType.IMMEDIATE}}, {})
        self.map.add_units([5, 5, 5], [6, 7, 8], E2, id_E1)

        self.init_map = copy.deepcopy(self.map)

        self.empty_buff   = {"period": -1, "max_period": -1, "skill_point": -1, "target": [0] * np.max(self.getInitUnitNumber()), "buff_type": BuffType.VOID}
        self.buffs        = ["fire", "tele"]

    def stringRepresentation(self, board):
        return board.tostring()

    def getBoardSize(self):
        return self.map.size, self.map.size

    def getActionSize(self):
        return self.map.size * self.map.size * len(flatten_list(self.skill_actions))

    def getInitUnitNumber(self):
        init_units_active = self.getUnitsOfType(self.init_map.grid, UnitType.ACTIVE)
        init_units_passive = self.getUnitsOfType(self.init_map.grid, UnitType.PASSIVE)
        return len(init_units_active), len(init_units_passive)

    def getInitBoard(self):
        self.map = copy.deepcopy(self.init_map)
        return self.map.grid

    def getUnitsOfType(self, board, type):
        units = []
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col].basic["type"] == type:
                    units.append(board[row][col])
        units = sorted(units, key=lambda x: x.info["id"])
        return np.array(units)

    def getUnitOfIdType(self, board, id, type=UnitType.ACTIVE):
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row][col].info["id"] == id and board[row][col].basic["type"] == type:
                    return board[row][col]
        return self.empty_unit

    def getStateDimension(self):
        board_dimension = len(flatten_list(onehots(self.empty_unit.basic)))
        skill_dimension = len(flatten_list(onehots(self.empty_skill))) * len(self.skills)
        buff_dimension = len(flatten_list(onehots(self.empty_buff))) * len(self.buffs)
        return board_dimension + skill_dimension + buff_dimension

    def getUnitSight(self, board, r, c, sight, sight_blocks):
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        if sight > 0:
            for d in directions:
                if 0 <= r + d[0] < board.shape[0] and 0 <= c + d[1] < board.shape[1]:
                    if not contains([r + d[0], c + d[1]], sight_blocks):
                        sight_blocks.append([r + d[0], c + d[1]])
                        if board[r + d[0]][c + d[1]].basic["type"] != UnitType.WALL:
                            self.getUnitSight(board, r + d[0], c + d[1], sight - 1, sight_blocks)
        return sight_blocks

    def getBlocksInSight(self, board):
        player1_units = self.getUnitsOfType(board, UnitType.ACTIVE)
        sights = []
        for unit in player1_units:
            unit_sight = self.getUnitSight(board, unit.info["row"], unit.info["col"], unit.basic["sight"], [[unit.info["row"], unit.info["col"]]])
            for us in unit_sight:
                if us not in sights:
                    sights.append(us)
        return np.array(sights)

    def targetToOnehot(self, target):
        onehot = [0] * np.max(self.getInitUnitNumber())
        if isinstance(target, Id):
            target = target.value
        if target <= 0:
            return onehot
        onehot[target - 1] = 1
        return onehot

    def getState(self, board):
        state = np.zeros_like(board).tolist()
        sight = self.getBlocksInSight(board)

        for row in range(len(state)):
            for col in range(len(state[row])):
                if contains([row, col], sight):
                    target = board[row][col]
                else:
                    target = copy.deepcopy(self.unknown)
                    target.info["row"] = row
                    target.info["col"] = col

                state_list = flatten_list(onehots(target.basic))
                for skill in self.skills:
                    if skill in target.skill:
                        state_list += flatten_list(onehots(target.skill[skill]))
                    else:
                        state_list += flatten_list(onehots(self.empty_skill))

                for buff in self.buffs:
                    if buff in target.buff:
                        buff_dict = copy.deepcopy(target.buff[buff])
                        for bl in buff_dict:
                            if isinstance(buff_dict[bl], Unit):
                                buff_dict[bl] = self.targetToOnehot(buff_dict[bl].info["id"])
                        state_list += flatten_list(onehots(buff_dict))
                    else:
                        state_list += flatten_list(onehots(self.empty_buff))

                state[row][col] = state_list

        return np.array(state)

    def getActiveAction(self, board):
        acted = [1 for _ in range(np.max(self.getInitUnitNumber()) * len(self.skill_actions))]
        acted = np.array(acted).reshape(-1, len(self.skill_actions))

        player1_units = self.getUnitsOfType(board, UnitType.ACTIVE)
        for player in player1_units:
            for skill in range(len(self.skill_actions)):
                if self.skills[skill] in player.skill and not player.skill[self.skills[skill]]["cd"]:
                    acted[player.info["id"] - 1][skill] = 0

        return acted

    def getTurnEndResults(self, board):
        player2_units = self.getUnitsOfType(board, UnitType.PASSIVE)
        for player in player2_units:
            for buff in player.buff:
                if player.buff[buff]["buff_type"] == BuffType.AFTERCALL:
                    target = player.buff[buff]["target"]
                    player.buff[buff]["period"] += 1

                    if buff == "fire":
                        target.basic["hp"] -= player.buff[buff]["skill_point"]
                        player.buff[buff]["skill_point"] -= 1
                        if target.basic["hp"] <= 0:
                            empty = copy.deepcopy(self.empty_unit)
                            empty.info["row"] = target.info["row"]
                            empty.info["col"] = target.info["col"]
                            board[target.info["row"]][target.info["col"]] = empty

            for buff in list(player.buff.keys()):
                if player.buff[buff]["buff_type"] == BuffType.AFTERCALL and player.buff[buff]["period"] == player.buff[buff]["max_period"]:
                    player.buff.pop(buff)

        player1_units = self.getUnitsOfType(board, UnitType.ACTIVE)
        for player in player1_units:
            for skill in player.skill:
                if player.skill[skill]["cd"] > 0:
                    player.skill[skill]["cd"] -= 1

        return board

    def getCanonicalForm(self, board):
        self.getTurnEndResults(board)
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row][col].basic["type"] == UnitType.ACTIVE:
                    board[row][col].basic["type"] = UnitType.PASSIVE
                elif board[row][col].basic["type"] == UnitType.PASSIVE:
                    board[row][col].basic["type"] = UnitType.ACTIVE

        return board

    def getGameEnded(self, board):
        player_alive = len(self.getUnitsOfType(board, UnitType.ACTIVE))
        enemy_alive = len(self.getUnitsOfType(board, UnitType.PASSIVE))

        if player_alive and enemy_alive:
            return 0
        elif player_alive:
            return 1
        elif enemy_alive:
            return -1
        else:
            return 0

    def gaussian2d(self, board, centers, sigmas=None):
        if sigmas is None:
            sigmas = [1 for _ in range(len(centers))]

        x = np.arange(board.shape[0])
        y = np.arange(board.shape[1])
        X, Y = np.meshgrid(x, y)

        gaussian_board = np.zeros_like(board).astype(float)
        for center, sigma in zip(centers, sigmas):
            dx = X - center[0]
            dy = Y - center[1]
            exponent = -(dx ** 2 + dy ** 2) / (2 * sigma ** 2)
            gaussian = np.exp(exponent).transpose()
            gaussian_board += gaussian
        gaussian_board /= np.sum(gaussian_board)

        return gaussian_board

    def getPeriodSkillValidActions(self, board):
        valid_actions = []
        player1_units = self.getUnitsOfType(board, UnitType.ACTIVE)
        player1_ids = [player.info["id"] for player in player1_units]
        for index in range(np.max(self.getInitUnitNumber())):
            valid_actions.append(np.array([0 for _ in range(board.shape[0] * board.shape[1] * len(flatten_list(self.skill_actions)))]).reshape((len(flatten_list(self.skill_actions)), board.shape[0], board.shape[1])))
            if index + 1 in player1_ids:
                player = player1_units[player1_ids.index(index + 1)]
                for buff in player.buff:
                    if player.buff[buff]["buff_type"] == BuffType.CONTINUE:
                        if buff == "tele":
                            target = player.buff[buff]["target"]
                            valid = target.get_all_positions(board, [UnitType.EMPTY], 1)
                            for r, c in valid:
                                valid_actions[-1][self.skill_actions[self.skills.index("tele")][player.buff[buff]["period"]]][r][c] = 1
        return np.array(valid_actions).reshape(-1)

    def getSightBoard(self, board, sights):
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if not contains([r, c], sights):
                    unknown = copy.deepcopy(self.unknown)
                    unknown.info["row"] = r
                    unknown.info["col"] = c
                    board[r][c] = unknown
        return board

    def getValidActions(self, board):
        sight_blocks = self.getBlocksInSight(board)
        sight_board = self.getSightBoard(copy.deepcopy(board), sight_blocks)

        valid_actions = self.getPeriodSkillValidActions(sight_board)
        if 1 in valid_actions:
            return np.array(valid_actions)

        acted = self.getActiveAction(board)
        player1_units = self.getUnitsOfType(board, UnitType.ACTIVE)
        player1_ids = [player.info["id"] for player in player1_units]
        levers = list(self.getUnitsOfType(sight_board, UnitType.LEVER))
        player2_units = list(self.getUnitsOfType(sight_board, UnitType.PASSIVE))

        centers = [[unit.info["row"], unit.info["col"]] for unit in player2_units + levers]
        if not centers:
            player1_positions = [[unit.info["row"], unit.info["col"]] for unit in player1_units]
            center = np.mean(player1_positions, axis=0).astype(int)
            centers.append([board.shape[0] - center[0], board.shape[1] - center[1]])
        gaussian = self.gaussian2d(board, centers).reshape(-1)

        valid_actions = []
        for index in range(np.max(self.getInitUnitNumber())):
            if index + 1 in player1_ids:
                unit = player1_units[player1_ids.index(index + 1)]
                for skill in self.skills:
                    for member in range(len(self.skill_actions[self.skills.index(skill)])):
                        valid_skill = np.array([0 for _ in range(board.shape[0] * board.shape[1])]).reshape(board.shape[0], board.shape[1]).tolist()
                        if member == 0:
                            if not acted[index][self.skills.index(skill)]:
                                target_positions = unit.get_all_positions(sight_board, flatten_list([unit.skill[skill]["target"]]), unit.skill[skill]["skill_range"])
                                if unit.skill[skill]["skill_type"] == SkillType.IMMEDIATE:
                                    for r, c in target_positions:
                                        valid_skill[r][c] = 1
                                elif unit.skill[skill]["skill_type"] == SkillType.CONTINUE:
                                    if skill == "tele":
                                        for r, c in target_positions:
                                            if len(board[r][c].get_all_positions(sight_board, [UnitType.EMPTY], 1)):
                                                valid_skill[r][c] = 1
                        valid_actions.append(valid_skill)
            else:
                for skill in self.skills:
                    for _ in range(len(self.skill_actions[self.skills.index(skill)])):
                        valid_actions.append(np.array([0 for _ in range(board.shape[0] * board.shape[1])]).reshape(board.shape[0], board.shape[1]).tolist())

        valid_actions = np.array(valid_actions).reshape(-1, board.shape[0] * board.shape[1])
        valid_actions = np.array([va * gaussian for va in valid_actions]).reshape(-1)

        return valid_actions

    def getNextState(self, board, action):
        player = action // self.getActionSize()
        skill = action % self.getActionSize()

        player1_units = self.getUnitsOfType(board, UnitType.ACTIVE)
        player1_ids = [player.info["id"] for player in player1_units]
        player = player1_units[player1_ids.index(player + 1)]
        board = player.act(board, skill, self.empty_unit, self.wall)

        skill_group = 0
        skill_index = skill // (board.shape[0] * board.shape[1])
        for action_index in self.skill_actions:
            if skill_index in action_index:
                skill = self.skills[self.skill_actions.index(action_index)]
                skill_group = player.skill[skill]["skill_group"]
                break

        for skill in player.skill:
            if player.skill[skill]["skill_group"] == skill_group:
                player.skill[skill]["cd"] = player.skill[skill]["max_cd"]

        return board

    def passTurn(self):
        log.info("Turn passed")

    def getSymmetries(self, board, pi):
        pi_board = pi.reshape(-1, board.shape[0] * board.shape[1])
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i)
                if j:
                    new_b = np.fliplr(new_b)
                temp_pi = []
                for pb in pi_board:
                    tpi = np.rot90(pb.reshape(board.shape), i)
                    if j:
                        tpi = np.fliplr(tpi)
                    temp_pi.append(tpi.reshape(-1).tolist())
                l += [(new_b, np.array(temp_pi).reshape(-1))]

        return l
