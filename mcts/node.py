import time
from collections import defaultdict
from game.game import Player
import numpy as np
import random

class Node:

    #CLASS VARIABLES. NOT OBJECT VARIABLES.
    next_node_id = 0
    #a dict which maps states to number of visits 
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        #assign a unique id
        self.node_id = Node.next_node_id
        Node.next_node_id += 1

        self.mdp = mdp
        self.parent = parent

        self.state = state

        self.qfunction = qfunction
        self.bandit = bandit

        self.reward = reward

        #action generated at this node ???
        self.action = action
        self.children = {}

    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)
        if len(valid_actions) == len(self.children):
            return True
        return False

    def select(self):
        #if this current node itself it not fully expanded, return this node.
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.state):
            #this condition is False if the node is fully expanded and the node is not terminal
            return self

        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.state, actions, self.qfunction)
            return self.get_outcome_child(action).select()



    def expand(self):
        if not self.is_fully_expanded():
            actions = list(set(self.mdp.get_actions(self.state)) - set(self.children.keys()))
            selected_action = random.choice(actions)
            self.children[selected_action] = []
            return self.get_outcome_child(selected_action)
            
        return self
        
    def backpropogate(self, reward, child):
        action = child.action
        Node.visits[tuple(self.state.flatten())] += 1
        Node.visits[(tuple(self.state.flatten()), action)] += 1
        
        q_value = self.qfunction.get_q_value(self.state, action)
        delta = (1 / (Node.visits[(tuple(self.state.flatten()), action)])) * (reward - self.qfunction.get_q_value(self.state, action))
        
        self.qfunction.update(self.state, action, delta)
        if self.parent != None:
            self.parent.backpropogate(self.reward + reward, self)


    def get_outcome_child(self, action):
        #choose some outcome based on the probabilities
        (next_state, reward, done) = self.mdp.execute(self.state, action, player=Player.AGENT)

        
        for (child, _) in self.children[action]:
            if np.all(next_state == child.state):
                return child

        new_child = Node(self.mdp, self, next_state, self.qfunction, self.bandit, reward, action)

        print(len(self.mdp.get_transitions(self.state, action)))
        for (outcome, probability) in self.mdp.get_transitions(self.state, action):
            if np.all(outcome == next_state):
                self.children[action] += [(new_child, probability)]
                return new_child


    #V(s)
    def get_value(self):
        max_q_value = self.qfunction.get_max_q(
            self.state, self.mdp.get_actions(self.state))
        return max_q_value

    def get_visits(self):
        return Node.visits[tuple(self.state.flatten())]


class MCTS:
    def __init__(self, mdp, qfunction, bandit):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit

    #runs the algorithm
    def mcts(self, timeout=1, root_node=None):
        #if there is no root node, create one
        if root_node is None:
            root_node = self.create_root_node()
        
        start_time = time.time()
        current_time = time.time()


        # train until a certain timeout is reached.
        while current_time < start_time + timeout:
            #select a node from the root node that 
            #is not fully expanded yet.

            #THE CORE OF MCTS ALGORITHM
            selected_node = root_node.select()
            if not self.mdp.is_terminal(selected_node.state):
                child = selected_node.expand()
                print(child)
                reward = self.simulate(child)
                selected_node.backpropogate(reward, child)

            current_time = time.time()

        return root_node

    def create_root_node(self):
        return Node(self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit, 0.0, None)

    def choose(self, state):
        return random.choice(self.mdp.get_actions(state))

    def simulate(self, node):
        #simulate the game and get the cumulative reward
        state = node.state
        cumulative_reward = 0.0
        depth = 0
        #during the simulation phase of mcts from the new node, 
            #we literally just take random actions
        while not self.mdp.is_terminal(state):
            action = self.choose(state)
            (next_state, reward, done) = self.mdp.execute(state, action, Player.AGENT)
            #future rewards get discounted, all the way till the terminal state.
            cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * reward
            depth += 1
            state = next_state

        return cumulative_reward
