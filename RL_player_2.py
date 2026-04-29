#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Constants
CORNERS = [(0,0), (0, 7), (7, 0), (7, 7)]


MAX,MIN = float('inf'),float('-inf')

class TimeoutException(Exception):
    pass

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        print(turn)
        print(board)



        # Minimax - Replace with your algorithm
        """NOTES:
            * 1 = white
            * -1 = black
        """      
        start_time = time.time()
        TIME_LIMIT = 5

        def check_time():
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutException 
            
        def find_available_moves(current_board: np.ndarray, turn: int) -> list:
            """Return a list of legal moves"""
            temp_board = reversi()
            temp_board.board = current_board
            
            # Find possible legal moves
            moves = []
            for a in range(8):
                for b in range(8):
                    if temp_board.step(a, b, turn, False) > 0:
                        moves.append((a, b)) # Append each possible move as tuple
            return moves
            
        def use_turn(current_board: np.ndarray, move: tuple, turn: int) -> np.ndarray:
            """Return new board copy after making a move"""
            game = reversi() # temp game instance
            
            game.board = np.copy(current_board) # Copy board
            
            # Make Move
            x, y = move
            game.step(x, y, turn, True)
            
            return game.board
            
        def board_score(current_board: np.ndarray, player: int) -> int:
            """Give a score to a board after a move is done"""
            # Score for corners
            corner_score = 0
            for x, y in CORNERS:
                if current_board[x, y] == player:
                    corner_score += 1
                elif current_board[x, y] == (-player):
                    corner_score -= 1
                    
            # Pieces Score
            player_pieces = np.sum(current_board == player)
            opponent_pieces = np.sum(current_board == (-player))
            pieces_score = player_pieces - opponent_pieces # Total pieces score
            
            # Mobility Score (Amount of moves available) for each player
            player_moves = len(find_available_moves(current_board, player))
            opponent_moves = len(find_available_moves(current_board, (-player)))
            mobility_score = player_moves - opponent_moves # Total mobility score
            
            # Assign multiplier values
            cs_mult = 25 # Corner score multiplier
            ps_mult = 1 # Pieces score multiplier
            ms_mult = 5 # Mobility score multiplier
            # FIXME: Add edges?
            
            total_player_score = (corner_score * cs_mult) + (pieces_score * ps_mult) + (mobility_score * ms_mult)
            return int(total_player_score)

        class PolicyNetwork(nn.Module):
            def __init__(self):
                super(PolicyNetwork, self).__init__()
                self.layer1 = nn.Linear(64, 128)
                self.layer2 = nn.Linear(128, 64)

            def forward(self, x):
                x = torch.tanh(self.layer1(x))
                return F.softmax(self.layer2(x), dim=1)

        def state_tensor(current_board: np.ndarray, player: int):
            state = (current_board * player).astype(np.float32).reshape(1, 64)
            return torch.tensor(state, dtype=torch.float32)

        def legal_action_probs(policy, current_board: np.ndarray, player: int, moves: list):
            probs = policy(state_tensor(current_board, player)).squeeze(0)
            mask = torch.zeros(64)
            for move_x, move_y in moves:
                mask[move_x * 8 + move_y] = 1.0

            probs = probs * mask
            if torch.sum(probs).item() == 0:
                probs = mask / torch.sum(mask)
            else:
                probs = probs / torch.sum(probs)
            return probs

        def greedy_move(current_board: np.ndarray, player: int, moves: list):
            best_move = moves[0]
            best_score = float('-inf')
            for move in moves:
                new_board = use_turn(current_board, move, player)
                score = board_score(new_board, player)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move

        def game_reward(current_board: np.ndarray, player: int):
            player_pieces = np.sum(current_board == player)
            opponent_pieces = np.sum(current_board == -player)
            if player_pieces > opponent_pieces:
                return 1.0
            if player_pieces < opponent_pieces:
                return -1.0
            return 0.0

        def train_policy():
            torch.manual_seed(1)
            np.random.seed(1)
            policy = PolicyNetwork()
            optimizer = optim.Adam(policy.parameters(), lr=0.001)

            for episode in range(30):
                temp_game = reversi()
                current_turn = 1
                learner = 1 if episode % 2 == 0 else -1
                passes = 0
                log_probs = []

                while passes < 2:
                    legal_moves = find_available_moves(temp_game.board, current_turn)
                    if len(legal_moves) == 0:
                        passes += 1
                        current_turn = -current_turn
                        continue

                    passes = 0
                    if current_turn == learner:
                        probs = legal_action_probs(policy, temp_game.board, current_turn, legal_moves)
                        action = torch.distributions.Categorical(probs).sample()
                        log_probs.append(torch.log(probs[action] + 1e-8))
                        move = (int(action.item()) // 8, int(action.item()) % 8)
                    else:
                        move = greedy_move(temp_game.board, current_turn, legal_moves)

                    temp_game.board = use_turn(temp_game.board, move, current_turn)
                    current_turn = -current_turn

                reward = game_reward(temp_game.board, learner)
                if len(log_probs) > 0:
                    loss = -torch.stack(log_probs).sum() * reward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            return policy

        if "reversi_policy" not in globals():
            globals()["reversi_policy"] = train_policy()
        reversi_policy = globals()["reversi_policy"]

        moves = find_available_moves(board, turn)
        x, y = (-1, -1)
        if len(moves) > 0:
            with torch.no_grad():
                probs = legal_action_probs(reversi_policy, board, turn, moves)

            best_move = moves[0]
            best_score = float('-inf')
            for move in moves:
                action_index = move[0] * 8 + move[1]
                new_board = use_turn(board, move, turn)
                score = board_score(new_board, turn) + (float(probs[action_index]) * 25)
                if score > best_score:
                    best_score = score
                    best_move = move
            x, y = best_move


        game_socket.send(pickle.dumps([x,y]))


if __name__ == '__main__':
    main()
