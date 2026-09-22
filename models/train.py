import torch
import pyspiel
#from models.model import Model
from models.trm import TinyRecurrentModel as Model
from models.utils import state_to_tensor
import torch.nn.functional as F

import time

class Trainer:
    def __init__(self):
        super().__init__()


        self.model = Model()

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device {self.device}")

        self.model = self.model.to(self.device)
        #self.model = torch.compile(self.model)

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-4
        )

    def chose_move(self, state):

        observation = state_to_tensor(state, self.device)

        policy_logits, value = self.model(observation)

        legal_moves = state.legal_actions()

        mask = torch.full((362,), float("-inf")).to(self.device)

        for action in legal_moves:
            mask[action] = 0

        masked_logits = policy_logits + mask

        probabilities = torch.softmax(masked_logits, dim=-1)

        action = torch.multinomial(probabilities, 1).item()

        action_log_prob = torch.log(probabilities[0, action] + 1e-8)

        return (observation, action, action_log_prob, mask)




    def play_game(self):

        board = pyspiel.load_game("go(board_size=19)")

        state = board.new_initial_state()


        game_data = []

        while not state.is_terminal():

            observation, action, action_log_prob, mask = self.chose_move(state)

            game_data.append({
                "observation": observation.detach(),
                "action": action,
                "log_prob": action_log_prob,
                "player": state.current_player(),
                "mask": mask
            })

            state.apply_action(action)

        returns = state.returns()

        return game_data, returns


    def train_game(self, game_data, returns):

        observations = torch.cat([
            sample["observation"]
            for sample in game_data
        ], dim=0)

        actions = torch.tensor(
            [sample["action"] for sample in game_data],
            dtype=torch.long,
            device=self.device
        )

        players = torch.tensor(
            [sample["player"] for sample in game_data],
            dtype=torch.long,
            device=self.device
        )

        targets = torch.tensor([
            [returns[player]] for player in players],
            dtype=torch.float32,
            device=self.device
        ).squeeze()

        mask = torch.tensor(
            [sample["mask"] for sample in game_data],
            dtype=torch.long,
            device=self.device
        )



        policy_logits, values = self.model(observations)


        masked_logits = policy_logits + mask



        values = values.squeeze(-1)

        value_loss = F.mse_loss(
            values,
            targets
        )

        log_probs = F.log_softmax(
            policy_logits,
            dim=-1
        )

        chosen_log_probs = log_probs[
            torch.arange(
                len(actions),
                device=self.device
            ),
            actions
        ]

        policy_loss = -(
            chosen_log_probs * targets
        ).mean()

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()


    def train(self, num_games=1_000):

        print("Training has started...")

        for game_number in range(num_games):


            game_data, returns = self.play_game()

            loss, policy_loss, value_loss = self.train_game(game_data, returns)

            print(
                f"Game {game_number + 1}/{num_games} | "
                f"Moves: {len(game_data)} | "
                f"Result: {returns} | "
                f"Loss: {loss:.4f} | "
                f"Policy: {policy_loss:.4f} | "
                f"Value: {value_loss:.4f} | "
            )

        print("Training has ended...")


