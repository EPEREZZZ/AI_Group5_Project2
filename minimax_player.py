#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi
import time

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

        def MM_Algorithm(board: np.ndarray, turn: int, depth: int, curr_player: int, alpha: int, beta: int) -> int:
            """Recursive MM_Algorithm algorithm"""
            legal_moves = find_available_moves(board, turn)
            opponent = -turn
            
            # Depth limit reached
            if depth == 0:
                score = board_score(board, curr_player)
                return score
            
            # No moves
            if len(legal_moves) == 0:
                opponent_moves = find_available_moves(board, opponent)
                
                # Game over (Undo recursion)
                if len(opponent_moves) == 0:
                    score = board_score(board, curr_player)
                    return score
                
                # Skip curr_player's turn
                else:
                    return MM_Algorithm(board, opponent, depth-1, curr_player,alpha,beta)
            
            
            # Maximizing player
            if turn == curr_player:
                highest_score = float('-inf')
                
                # Test all moves
                for move in legal_moves:
                    new_board = use_turn(board, move, turn)
                    score = MM_Algorithm(new_board, opponent, depth - 1, curr_player, alpha, beta)
                    # Find highest score
                    highest_score = max(highest_score, score)
                    alpha = max(alpha,highest_score)
                    if beta <= alpha:
                        break
                return highest_score
            
            # Minimizing player
            else:
                lowest_score = float('inf')
                
                for move in legal_moves:
                    new_board = use_turn(board, move, turn)
                    score = MM_Algorithm(new_board, opponent, depth-1, curr_player, alpha, beta)
                    
                    # Find lowest score
                    lowest_score = min(lowest_score, score)
                    beta = min(beta,lowest_score)
                    if beta <= alpha:
                        break
                return lowest_score
        
        
        x = -1
        y = -1
        best_root_score = float('-inf')

        moves = find_available_moves(board, turn)
        p = len(moves)
        if p <= 3:
            depth=7
        elif p <= 6:
            depth=6
        elif p <=7:
            depth=5
        elif p <=8:
            depth=4
        else:
            depth=3

        print(depth)

        # Default move so we always return something
        x, y = (-1, -1)
        if len(moves) > 0:
            x, y = moves[0]

        try:

            for move in moves:

                check_time()

                new_board = use_turn(board, move, turn)
                score = MM_Algorithm(new_board, -turn, depth-1, turn, MIN, MAX)

                if score > best_root_score:
                    best_root_score = score
                    x, y = move

        except TimeoutException:
            print(f"Search stopped due to time limit : p = {p}")
            game_socket.send(pickle.dumps([x,y]))


        game_socket.send(pickle.dumps([x,y]))


if __name__ == '__main__':
    main()