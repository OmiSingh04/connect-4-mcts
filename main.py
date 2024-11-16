from mcts.mdp import MDP, QFunction, Bandit
from mcts.node import MCTS, Node
import numpy as np
import requests

qfunction = QFunction()

load_qfunction = True

qfunction_file_path = "outputs/qfunction.hdf5"

if load_qfunction:
    qfunction.load_policy(qfunction_file_path)

for (i, j) in qfunction.qtable.items():
    print(i, j)



#game = Game()
#game.print()
#
#done = False
#current_player = Player.AGENT
#
#while not done:
#    (i, j) = call_best_opponent_move(game=game) if current_player == Player.OPPONENT else get_user_move(game=game)
#    game.print()
#    done = call_has_won(game, i, j, current_player)
#    if done in [-1 ,1, 2]:
#        #print(f'{'Agent wins' if done == 1 else 'Opponent wins' if done == 2 else 'Draw'}')
#        print(f"{'Agent wins' if done == 1 else 'Opponent wins' if done == 2 else 'Draw'}")
#        break
#     
#    current_player = Player.OPPONENT if current_player == Player.AGENT else Player.AGENT
