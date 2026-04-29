#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Constants
CORNERS = [(0,0), (0, 7), (7, 0), (7, 7)]

MAX,MIN = float('inf'),float('-inf')

HIDDEN_NEURONS = 256
TRAINING_EPISODES = 100000
EVAL_EVERY = 1000
EVAL_GAMES = 30
CONTINUE_TRAINING = True
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reversi_policy.pt")

class TimeoutException(Exception):
    pass

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(64, HIDDEN_NEURONS)
        self.layer2 = nn.Linear(HIDDEN_NEURONS, 64)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        return F.softmax(self.layer2(x), dim=1)

class ReversiHelper:
    def find_available_moves(self, current_board: np.ndarray, turn: int) -> list:
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

    def use_turn(self, current_board: np.ndarray, move: tuple, turn: int) -> np.ndarray:
        """Return new board copy after making a move"""
        game = reversi() # temp game instance

        game.board = np.copy(current_board) # Copy board

        # Make Move
        x, y = move
        game.step(x, y, turn, True)

        return game.board

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

    def greedy_train(self, current_board: np.ndarray, player: int, moves: list):
        """Use greedy algorithm as initial trainer"""
        best_move = moves[0]
        best_score = float('-inf')
        for move in moves:
            new_board = self.use_turn(current_board, move, player)
            score = self.board_score(new_board, player)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def game_reward(self, current_board: np.ndarray, player: int):
        """Determine final game reward"""
        player_pieces = np.sum(current_board == player)
        opponent_pieces = np.sum(current_board == -player)
        if player_pieces > opponent_pieces:
            return 1.0
        if player_pieces < opponent_pieces:
            return -1.0
        return 0.0

class PolicyAgent:
    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.helper = ReversiHelper()
        self.policy = self.load_or_train_policy()

    def board_tensor(self, current_board: np.ndarray, player: int):
        """Return board state as a tensor"""
        state = (current_board * player).astype(np.float32).reshape(1, 64)
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def legal_action_probs(self, policy, current_board: np.ndarray, player: int, moves: list):
        """Return probabilities for legal moves"""
        probs = policy(self.board_tensor(current_board, player)).squeeze(0)
        mask = torch.zeros(64, device=self.device)
        for move_x, move_y in moves:
            mask[move_x * 8 + move_y] = 1.0

        probs = probs * mask
        if torch.sum(probs).item() == 0:
            probs = mask / torch.sum(mask)
        else:
            probs = probs / torch.sum(probs)
        return probs

    def choose_policy_move(self, policy, current_board: np.ndarray, player: int, moves: list):
        """Choose best legal policy move"""
        with torch.no_grad():
            probs = self.legal_action_probs(policy, current_board, player, moves)

        best_move = moves[0]
        best_score = float('-inf')
        for move in moves:
            action_index = move[0] * 8 + move[1]
            new_board = self.helper.use_turn(current_board, move, player)
            score = self.helper.board_score(new_board, player) + (float(probs[action_index].detach().cpu()) * 25)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def save_policy(self, policy, episode: int, best_score: float):
        """Save policy that was trained"""
        checkpoint = {
            "policy_state_dict": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
            "hidden_neurons": int(HIDDEN_NEURONS),
            "episodes": int(episode),
            "best_score": float(best_score),
            "device_used": str(self.device),
        }
        torch.save(checkpoint, MODEL_PATH)

    def load_checkpoint(self):
        """Load saved checkpoint data"""
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        except pickle.UnpicklingError:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        if checkpoint.get("hidden_neurons") != HIDDEN_NEURONS:
            raise ValueError("Checkpoint was trained with a different hidden layer size.")
        return checkpoint

    def load_policy_from_checkpoint(self, training: bool = False):
        """Load policy from checkpoint"""
        policy = PolicyNetwork().to(self.device)
        checkpoint = self.load_checkpoint()
        policy.load_state_dict(checkpoint["policy_state_dict"])
        if training:
            policy.train()
        else:
            policy.eval()
        return policy, checkpoint

    def load_policy(self):
        """Load policy (if exists)"""
        policy, checkpoint = self.load_policy_from_checkpoint(False)
        self.save_policy(policy, int(checkpoint.get("episodes", 0)), float(checkpoint.get("best_score", 0.0)))
        print(f"Loaded policy checkpoint from {MODEL_PATH}")
        return policy

    def evaluate_policy(self, policy, games: int):
        """Return policy evaluation score"""
        policy.eval()
        total_score = 0.0
        for game_number in range(games):
            temp_game = reversi()
            learner = 1 if game_number % 2 == 0 else -1
            current_turn = 1
            passes = 0

            while passes < 2:
                legal_moves = self.helper.find_available_moves(temp_game.board, current_turn)
                if len(legal_moves) == 0:
                    passes += 1
                    current_turn = -current_turn
                    continue

                passes = 0
                if current_turn == learner:
                    move = self.choose_policy_move(policy, temp_game.board, current_turn, legal_moves)
                else:
                    move = self.helper.greedy_train(temp_game.board, current_turn, legal_moves)

                temp_game.board = self.helper.use_turn(temp_game.board, move, current_turn)
                current_turn = -current_turn

            total_score += self.helper.game_reward(temp_game.board, learner) * 100
            total_score += np.sum(temp_game.board == learner) - np.sum(temp_game.board == -learner)

        policy.train()
        return float(total_score / games)

    def train_policy(self):
        """Train the policy network"""
        torch.manual_seed(1)
        np.random.seed(1)
        opponent_policy = None
        best_state = None
        if os.path.exists(MODEL_PATH):
            try:
                policy, checkpoint = self.load_policy_from_checkpoint(True)
                opponent_policy, _ = self.load_policy_from_checkpoint(False)
                best_score = float(checkpoint.get("best_score", float('-inf')))
                best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
                print(f"Training from checkpoint against previous policy: {MODEL_PATH}")
            except (RuntimeError, ValueError, KeyError, pickle.UnpicklingError) as error:
                print(f"Could not use checkpoint for self-play: {error}")
                print("Training from a new policy against greedy.")
                policy = PolicyNetwork().to(self.device)
                best_score = float('-inf')
        else:
            policy = PolicyNetwork().to(self.device)
            best_score = float('-inf')
        optimizer = optim.Adam(policy.parameters(), lr=0.001)

        print(f"Training policy on {self.device} for {TRAINING_EPISODES} episodes...")
        for episode in range(1, TRAINING_EPISODES + 1):
            policy.train()
            if episode % 1000 == 0:
                print(f"trained episodes: {episode}/{TRAINING_EPISODES}")
            temp_game = reversi()
            current_turn = 1
            learner = 1 if episode % 2 == 0 else -1
            passes = 0
            log_probs = []

            while passes < 2:
                legal_moves = self.helper.find_available_moves(temp_game.board, current_turn)
                if len(legal_moves) == 0:
                    passes += 1
                    current_turn = -current_turn
                    continue

                passes = 0
                if current_turn == learner:
                    probs = self.legal_action_probs(policy, temp_game.board, current_turn, legal_moves)
                    action = torch.distributions.Categorical(probs).sample()
                    log_probs.append(torch.log(probs[action] + 1e-8))
                    move = (int(action.item()) // 8, int(action.item()) % 8)
                else:
                    if opponent_policy is not None:
                        move = self.choose_policy_move(opponent_policy, temp_game.board, current_turn, legal_moves)
                    else:
                        move = self.helper.greedy_train(temp_game.board, current_turn, legal_moves)

                temp_game.board = self.helper.use_turn(temp_game.board, move, current_turn)
                current_turn = -current_turn

            reward = self.helper.game_reward(temp_game.board, learner)
            if len(log_probs) > 0:
                loss = -torch.stack(log_probs).sum() * reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % EVAL_EVERY == 0:
                score = self.evaluate_policy(policy, EVAL_GAMES)
                print(f"episode={episode} eval_score={score:.2f} best_score={best_score:.2f}")
                if score > best_score:
                    best_score = score
                    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
                    self.save_policy(policy, episode, best_score)

        if best_state is not None:
            policy.load_state_dict(best_state)
        elif not os.path.exists(MODEL_PATH):
            best_score = self.evaluate_policy(policy, EVAL_GAMES)
            self.save_policy(policy, TRAINING_EPISODES, best_score)

        policy.eval()
        return policy

    def load_or_train_policy(self):
        """Return loaded or trained policy"""
        if CONTINUE_TRAINING or not os.path.exists(MODEL_PATH):
            return self.train_policy()
        try: # Attempt to use saved policy if exists
            return self.load_policy()
        except (RuntimeError, ValueError, KeyError) as error:
            print(f"Could not load checkpoint: {error}")
            print("Training a new policy checkpoint.")
            return self.train_policy()

    def choose_move(self, current_board: np.ndarray, turn: int):
        """Choose move for the current board"""
        moves = self.helper.find_available_moves(current_board, turn)
        if len(moves) == 0:
            return -1, -1
        return self.choose_policy_move(self.policy, current_board, turn, moves)

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    agent = PolicyAgent()

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
        x, y = agent.choose_move(board, turn)


        game_socket.send(pickle.dumps([x,y]))


if __name__ == '__main__':
    main()
