import random
import requests
import numpy as np 
from enum import Enum

class Player(Enum):
    AGENT = 1
    OPPONENT = 2

class Game:
    def __init__(self):   
        self.state = np.zeros((6, 7), dtype=np.int8)
        self.steps = 0
        self.done = False
        self.next_player = Player.AGENT

    def print(self):
        print(self.state, end="\n\n")

    def reset(self):
        self.__init__()

    def is_action_illegal(self, k) -> bool:
        return self.state[:1,k].item() != 0

    def is_game_done(self):
    # Check horizontal, vertical, and diagonal wins
        board = self.state
        rows, cols = board.shape

        # Check horizontal and vertical
        for r in range(rows):
            for c in range(cols):
                # Horizontal check (right direction)
                if c + 3 < cols and board[r, c] == board[r, c + 1] == board[r, c + 2] == board[r, c + 3] != 0:
                    return board[r, c]
                # Vertical check (down direction)
                if r + 3 < rows and board[r, c] == board[r + 1, c] == board[r + 2, c] == board[r + 3, c] != 0:
                    return board[r, c]
                # Diagonal (down-right direction)
                if r + 3 < rows and c + 3 < cols and board[r, c] == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3] != 0:
                    return board[r, c]
                # Diagonal (down-left direction)
                if r + 3 < rows and c - 3 >= 0 and board[r, c] == board[r + 1, c - 1] == board[r + 2, c - 2] == board[r + 3, c - 3] != 0:
                    return board[r, c]

        # Check for tie (board is full)
        if np.all(board != 0):
            return -1  # Tie

        # If no winner and space is available
        return 0  # Game ongoing
            

    def step(self, k, player: Player):
        #we will assume that k is a valid move.
        self.steps += 1
        for i in range(5, -1, -1):
            if self.state[i, k] == 0:
                self.state[i, k] = 1 if player == Player.AGENT else 2
                return (i, k)

        #what we can do is make calling the bot from here itself.

def render_state(game: Game):
	pass


#1 if agent wins, 2 if opponent wins, -1 is draw, and 0 if the game is still ongoing
def call_has_won(game: Game, i, j, player):
    board_str = ''.join(game.state.astype(np.int32).astype(str).flatten())
    player_value = 1 if player == Player.AGENT else 2
    url = f"http://localhost:8009/index.php/hasWon?board_data={board_str}&player={player_value}&i={i}&j={j}"
    response = requests.get(url)
    data = response.json()

    if data:
        return 1 if player == Player.AGENT else 2

    if not data:
        if np.any(game.state == 0):
            return 0
    
    return -1


def call_preferred_opponent_move(state=None, game: Game=None):
    return call_best_opponent_move(state=state, game=game)
    #return call_opponent_move(state=state, game=game)


def call_best_opponent_move(state=None, game: Game=None):
    if game:
        board_str = ''.join(game.state.astype(np.int32).astype(str).flatten())
    else:
        board_str = ''.join(state.astype(np.int32).astype(str).flatten())

    url = f"http://localhost:8009/index.php/getMoves?board_data={board_str}&player=2"
    response = requests.get(url)
    data = response.json()
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)
    best_move_key = int(sorted_data[0][0])
    if not game:
        return best_move_key
    return game.step(best_move_key, player=Player.OPPONENT)


#a mix of 50% second best vs 50% best move. the second best move is usually just really bad very often it seems
def call_opponent_move(state=None, game: Game=None):
    if game:
        board_str = ''.join(game.state.astype(np.int32).astype(str).flatten())
    else:
        board_str = ''.join(state.astype(np.int32).astype(str).flatten())

    url = f"http://localhost:8009/index.php/getMoves?board_data={board_str}&player=2"
    response = requests.get(url)
    data = response.json()

    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

    if len(set(data.values())) == 1:
        # If all values are the same, randomly sample
        selected_key = int(random.choice(list(data.keys())))
        if not game:
            return selected_key
        return game.step(selected_key, player=Player.OPPONENT)
    else:
        # Step 3: Get the highest and second-highest values
        highest_key, highest_value = sorted_data[0]
        second_highest_key, second_highest_value = sorted_data[1] if len(sorted_data) > 1 else (None, None)

        # Step 4: Select based on the 90/10 chance
        if random.random() < 0.5:
            selected_key = highest_key
        else:
            selected_key = second_highest_key

        selected_key = int(selected_key)
        if not game:
            return selected_key
        return game.step(selected_key, player=Player.OPPONENT)


def get_user_move(game: Game):
    print("Enter valid j")
    j = int(input())
    return game.step(j, player=Player.AGENT)
