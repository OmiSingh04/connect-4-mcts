import numpy as np
from collections import defaultdict
import h5py
from game.game import Player, Game, call_has_won, call_preferred_opponent_move
import random
import json


#state is simply a numpy array of shape (6, 7)

class MDP:
    def get_actions(self, state):
        return np.where(state[:1,] == 0)[1]

    def is_terminal(self, state):
        result = self.check_winner(state)
        if result in [1, 2]:
            return True
        if np.any(state==0):
            return False
        return True

    def get_reward(self, state, action, next_state, i, j):
        winner = self.check_winner(next_state)
        return +1 if winner == 1 else -1 if winner == 2 else 0


    def check_winner(self, board):
        for player in [1, 2]:
            # Horizontal check
            for row in range(6):
                for col in range(4):
                    if (board[row, col] == player and
                        board[row, col + 1] == player and
                        board[row, col + 2] == player and
                        board[row, col + 3] == player):
                        return player

            # Vertical check
            for col in range(7):
                for row in range(3):
                    if (board[row, col] == player and
                        board[row + 1, col] == player and
                        board[row + 2, col] == player and
                        board[row + 3, col] == player):
                        return player

            # Diagonal checks
            for row in range(3):
                for col in range(4):
                    # Positive diagonal
                    if (board[row, col] == player and
                        board[row + 1, col + 1] == player and
                        board[row + 2, col + 2] == player and
                        board[row + 3, col + 3] == player):
                        return player
                    # Negative diagonal
                    if (board[row + 3, col] == player and
                        board[row + 2, col + 1] == player and
                        board[row + 1, col + 2] == player and
                        board[row, col + 3] == player):
                        return player

        return 0

    def get_discount_factor(self):
        return 0.9

    def get_initial_state(self):
        return np.zeros((6, 7))

    def get_goal_states(self):
        pass

    #we do this one today
    #return (next_State, reward, done = Boolean!!)
    def execute(self, state, action, player: Player):
        #we assume action is legal

        #(new_state, reward, done)
        new_state = state.copy()
        target_i, target_j = 0, 0

        for i in range(5, -1, -1):
            if new_state[i, action] == 0:
                new_state[i, action] = player.value
                target_i, target_j = i, action
                break


        #def get_reward(self, state, action, next_state, i, j):
        reward = self.get_reward(state, action, new_state, target_i, target_j)

        #if the reward is non-zero, we have a terminal state.
        if reward in [1, -1] or np.all(new_state != 0):
            return (new_state, reward, self.is_terminal(new_state))

        #we can assume that the new state is not terminal
        selected_action = call_preferred_opponent_move(state=new_state)

        for i in range(5, -1, -1):
            if new_state[i, selected_action] == 0:
                new_state[i, selected_action] = Player.OPPONENT.value
                break
        
        return (new_state, reward, self.is_terminal(new_state))


    def get_transitions(self, state, action):
        board = np.copy(state)
        #this game has no stochastic nature. a specific move in a specific state has only 1 outcome
        probability = 1
        for i in range(5, -1, -1):
            if board[i, action] == 0:
                board[i, action] = 1
                break

        if self.is_terminal(board):
            return (board, probability)

        best_move = call_preferred_opponent_move(board)

        for i in range(5, -1, -1):
            if board[i, best_move] == 0:
                board[i, best_move] = 2
                break

        return [(board, probability)]

class QFunction:
    def __init__(self, filename=None):
        self.filename = filename
        if filename:
            self.load_policy(filename)
        else:
            self.qtable = {}

    def save_policy(self, file_path):
        with h5py.File(file_path, 'w') as hdf5_file:
            for key, inner_dict in self.qtable.items():
                str_key = json.dumps(key)  # Serialize tuple key as string
                group = hdf5_file.create_group(str_key)
                for inner_key, value in inner_dict.items():
                    group.create_dataset(str(inner_key), data=value)

    def load_policy(self, file_path):
        data_dict = {}
        with h5py.File(file_path, 'r') as hdf5_file:
            for str_key in hdf5_file.keys():
                tuple_key = tuple(json.loads(str_key))  # Deserialize string back to tuple
                inner_dict = {int(k): float(v[()]) for k, v in hdf5_file[str_key].items()}
                data_dict[tuple_key] = inner_dict

        self.qtable = data_dict


    def update(self, state, action, delta):
        if tuple(state.flatten()) not in self.qtable:
            self.qtable[tuple(state.flatten())] = {}
        if action not in self.qtable[tuple(state.flatten())]:
            self.qtable[tuple(state.flatten())][action] = 0.0
        self.qtable[tuple(state.flatten())][action] += delta


    def get_q_value(self, state, action):
        if tuple(state.flatten()) in self.qtable and action in self.qtable[tuple(state.flatten())]:
            return self.qtable[tuple(state.flatten())][action]
        return 0.0


    def get_argmax_q(self, state, actions):
        (argmax_q, max_q) = self.get_max_pair(state, actions)
        return argmax_q

    def get_max_q(self, state, actions):
        (argmax_q, max_q) = self.get_max_pair(state, actions)
        return max_q

    def get_max_pair(self, state, actions):
        max_q = float("-inf")
        max_actions = []
        for action in actions:
            value = self.get_q_value(state, action)
            if value > max_q:
                max_actions = [action]
                max_q = value
            elif value == max_q:
                max_actions += [action]

        arg_max_q = random.choice(max_actions)
        return (arg_max_q, max_q)


#exploration vs exploitation action selecion - UCB1 selection
class Bandit:
    def __init__(self, filename=None):
        self.total = 0
        self.n_s_a = {} #N(s) = sum(N, a) for all a
        if filename:
            self.n_s_a = self.load_bandit(filename)
        

    def save_bandit(self, filename):
        with h5py.File(filename, 'w') as f:
            for state, actions in self.n_s_a.items():
                state_group = f.create_group(str(state))
                for action, n_value in actions.items():
                    state_group.create_dataset(str(action), data=n_value)

    def load_bandit(self, filename):
        self.n_s_a = {}
        with h5py.File(filename, 'r') as f:
            for state in f:
                self.n_s_a[state] = {}
                state_group = f[state]
                for action in state_group:
                    self.n_s_a[state][int(action)] = state_group[action][()]

    def select(self, state, actions, qfunction):
        if tuple(state.flatten()) not in self.n_s_a.keys():
            self.n_s_a[tuple(state.flatten())] = {}

        for action in actions:
            if action not in self.n_s_a[tuple(state.flatten())].keys():
                self.n_s_a[tuple(state.flatten())][action] = 1
                return action

        max_actions = []
        max_value = float('-inf')

        for action in actions:
            value = qfunction.get_q_value(state, action) + (np.sqrt((2 * np.log(sum(x for x in self.n_s_a[tuple(state.flatten())].values()))) / (self.n_s_a[tuple(state.flatten())][action])))
            if value > max_value:
                max_value = value
                max_actions = [action]

            elif value == max_value:
                max_actions.append(action)

        result = random.choice(max_actions)
        
        self.n_s_a[tuple(state.flatten())][result] += 1

        return result
