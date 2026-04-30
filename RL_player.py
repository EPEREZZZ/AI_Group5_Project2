# Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi
import torch
import torch.nn as nn
import torch.optim as optim
import os


# Trains once, save weights, then load in the future if they exist
TRAINING_EPISODES = 10000
LEARNING_RATE = 0.001
INSPECT_TRAINING = 20
FINAL_REWARD_MULTIPLIER = 100
POLICY_BONUS_MULTIPLIER = 1
HIDDEN_NEURONS = 128
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reversi_policy.pt")

# Constants
CORNERS = [(0,0), (0, 7), (7, 0), (7, 7)]

class PolicyNetwork(nn.Module):
    """Small multilayer policy network."""

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Create network layers
        self.layer1 = nn.Linear(64, HIDDEN_NEURONS)
        self.layer2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.layer3 = nn.Linear(HIDDEN_NEURONS, 64)

    def forward(self, x):
        # Forward pass through hidden layers
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class ReversiHelper:
    def find_available_moves(self, current_board: np.ndarray, turn: int) -> list:
        """Return a list of legal moves"""
        temp_game = reversi()
        temp_game.board = current_board

        # Find possible legal moves
        moves = []
        for i in range(8):
            for j in range(8):
                if temp_game.step(i, j, turn, False) > 0:
                    moves.append((i, j)) # Append each possible move as tuple
        return moves

    def use_turn(self, current_board: np.ndarray, move: tuple, turn: int) -> np.ndarray:
        """Return new board copy after making a move"""
        temp_game = reversi() # temp game instance
        temp_game.board = np.copy(current_board) # Copy board
        
        # Make Move
        x, y = move
        temp_game.step(x, y, turn, True)
        return temp_game.board

    def greedy_move(self, current_board: np.ndarray, turn: int, moves: list):
        """Greedy algorithm for training"""
        temp_game = reversi()
        temp_game.board = current_board

        # Find move that flips most pieces
        best_move = moves[0]
        best_flips = 0
        for move in moves:
            flips = temp_game.step(move[0], move[1], turn, False)
            if flips > best_flips:
                best_flips = flips
                best_move = move
        return best_move

    def board_score(self, current_board: np.ndarray, player: int) -> int:
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
        player_moves = len(self.find_available_moves(current_board, player))
        opponent_moves = len(self.find_available_moves(current_board, (-player)))
        mobility_score = player_moves - opponent_moves # Total mobility score

        # Assign multiplier values
        cs_mult = 25 # Corner score multiplier
        ps_mult = 1 # Pieces score multiplier
        ms_mult = 5 # Mobility score multiplier
        # FIXME: Add edges?

        total_player_score = (corner_score * cs_mult) + (pieces_score * ps_mult) + (mobility_score * ms_mult)
        return int(total_player_score)

    def game_reward(self, current_board: np.ndarray, player: int) -> float:
        """Final win/loss reward to influence the learner"""
        # Count pieces
        player_pieces = np.sum(current_board == player)
        opponent_pieces = np.sum(current_board == -player)

        # Return win/loss reward
        if player_pieces > opponent_pieces:
            return 1.0
        if player_pieces < opponent_pieces:
            return -1.0
        return 0.0


class PolicyAgent:
    def __init__(self):
        # Set up agent
        self.device = self.choose_device()
        self.helper = ReversiHelper()
        self.policy = PolicyNetwork().to(self.device)
        self.load_or_train_policy()

    def choose_device(self):
        """Prefer GPU when training"""
        # MPS (Apple Silicon GPU)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Training using mpc (Apple Silicon)")
            return torch.device("mps")
        # Cuda (NVIDIA GPU)
        if torch.cuda.is_available():
            print("Training using cuda")
            return torch.device("cuda")
        # CPU
        else:
            print("Training using CPU")
            return torch.device("cpu")

    def load_or_train_policy(self):
        """Load saved weights if they exist; otherwise train and save them."""
        # Load existing weights
        if os.path.exists(MODEL_PATH):
            try:
                self.policy.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.policy.eval()
                print(f"Loaded policy weights from {MODEL_PATH}")
                return
            except RuntimeError:
                print("Failed to load policy weights. Training new weights...")

        # Train then save new weights
        self.train_policy()
        torch.save(self.policy.state_dict(), MODEL_PATH)
        print(f"Saved policy weights to {MODEL_PATH}")

    def board_tensor(self, current_board: np.ndarray, player: int):
        """Convert the board to a 1x64 tensor where this player's pieces are 1 and the opponent's are -1"""
        state = (current_board * player).astype(np.float32).reshape(1, 64)
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def legal_move_scores(self, current_board: np.ndarray, player: int, moves: list):
        """Return move scores where illegal moves are blocked"""
        # Evaluate scores for all 64 pieces
        move_scores = self.policy(self.board_tensor(current_board, player)).squeeze(0)
        # Block all moves by default
        blocked_moves = torch.full((64,), -1.0e9, dtype=torch.float32, device=self.device)
        # Allow legal moves
        for move_x, move_y in moves:
            blocked_moves[move_x * 8 + move_y] = 0.0

        return move_scores + blocked_moves

    def sample_policy_move(self, current_board: np.ndarray, player: int, moves: list):
        """Sample legal move for training and return log prob"""
        # Create prob distribution from sores of legal moves
        move_scores = self.legal_move_scores(current_board, player, moves)
        distribution = torch.distributions.Categorical(logits=move_scores)

        # Sample move
        action = distribution.sample()
        move = (int(action.item()) // 8, int(action.item()) % 8)
        return move, distribution.log_prob(action)

    def choose_policy_move(self, current_board: np.ndarray, player: int, moves: list):
        """Choose the best move, with small policy bonus"""
        # Get NN move scores
        with torch.no_grad():
            move_scores = self.legal_move_scores(current_board, player, moves)

        # Compare every legal move
        best_move = moves[0]
        best_score = float("-inf")
        for move in moves:
            action_index = move[0] * 8 + move[1]
            new_board = self.helper.use_turn(current_board, move, player)
            board_score = self.helper.board_score(new_board, player)
            policy_bonus = float(move_scores[action_index].detach().cpu()) * POLICY_BONUS_MULTIPLIER
            score = board_score + policy_bonus

            # Save and return best move
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def train_policy(self):
        """Train 🏋️‍♀️"""
        # Random seed
        torch.manual_seed(1)
        np.random.seed(1)

        optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        print(f"Training policy network on {self.device} for {TRAINING_EPISODES} episodes against greedy")

        # Play against greedy for training
        for episode in range(1, TRAINING_EPISODES + 1):
            temp_game = reversi()
            learner = 1 if episode % 2 == 0 else -1
            current_turn = 1
            passes = 0
            move_feedback = []

            while passes < 2:
                legal_moves = self.helper.find_available_moves(temp_game.board, current_turn)
                if len(legal_moves) == 0:
                    passes += 1
                    current_turn = -current_turn
                    continue

                passes = 0

                # RL learner move
                if current_turn == learner:
                    before_score = self.helper.board_score(temp_game.board, learner)
                    move, log_prob = self.sample_policy_move(temp_game.board, current_turn, legal_moves)
                    new_board = self.helper.use_turn(temp_game.board, move, current_turn)
                    after_score = self.helper.board_score(new_board, learner)
                    score_change = after_score - before_score
                    move_feedback.append((log_prob, float(score_change)))

                # Greedy move
                else:
                    move = self.helper.greedy_move(temp_game.board, current_turn, legal_moves)
                    new_board = self.helper.use_turn(temp_game.board, move, current_turn)

                # Update board and turn
                temp_game.board = new_board
                current_turn = -current_turn

            # Final game reward
            final_reward = self.helper.game_reward(temp_game.board, learner) * FINAL_REWARD_MULTIPLIER
            if len(move_feedback) > 0:
                losses = []

                # Build loss from move rewards
                for log_prob, score_change in move_feedback:
                    reward = score_change + final_reward
                    if reward != 0.0:
                        losses.append(-log_prob * reward)

                if len(losses) == 0:
                    continue

                # Update policy weights
                loss = torch.stack(losses).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % INSPECT_TRAINING == 0:
                print(f"trained episodes: {episode}/{TRAINING_EPISODES}")

        self.policy.eval()

    def choose_move(self, current_board: np.ndarray, turn: int):
        """Choose a legal move from a given board"""
        # Find all legal moves
        moves = self.helper.find_available_moves(current_board, turn)
        if len(moves) == 0:
            return -1, -1
        # Choose best move
        return self.choose_policy_move(current_board, turn, moves)


def main():
    agent = PolicyAgent()

    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))

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

        #Local RL_Player
        x, y = agent.choose_move(board, turn)
        game_socket.send(pickle.dumps([x, y]))


if __name__ == "__main__":
    main()
