from game.game import Game, call_opponent_move, call_has_won, Player, get_user_move
from mcts.mdp import MDP, QFunction, Bandit
from mcts.node import MCTS, Node
import numpy as np
import requests

mdp = MDP()
qfunction = QFunction()
bandit = Bandit()

load_n_s_a = False
load_qfunction = False

n_s_a_file_path = "outputs/nsa.hdf5"
qfunction_file_path = "outputs/qfunction.hdf5"

if load_n_s_a:
    bandit.load_bandit(n_s_a_file_path)
if load_qfunction:
    qfunction.load_policy(qfunction_file_path)

mcts = MCTS(mdp, qfunction, bandit)

#*************************************
#THE MAIN TRAINING
root_node = mcts.mcts(timeout=300)
#*************************************

print(root_node.node_id)
print("next node id - ", Node.next_node_id)

print(qfunction.qtable)

input("enter to leave")
#this q function is essentially what we need
qfunction.save_policy(qfunction_file_path)
bandit.save_bandit(n_s_a_file_path)
