import torch
import pyspiel
#from models.model import Model
from models.trm import TinyRecurrentModel as Model
from models.utils import state_to_tensor
import torch.nn.functional as F

from models.batch_wrapper import GoEnv

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

    @torch.no_grad()
    def chose_move(self, envs):


        observations = []
        legal_actions = []
        masks = []

        for env in envs:
            state = env.state

            observation = state_to_tensor(state, "cpu")

            mask = torch.full((362,), float("-inf"))

            legal_moves = state.legal_actions()

            for action in legal_moves:
                mask[action] = 0

            observations.append(observation)
            legal_actions.append(action)
            masks.append(mask)

        observation = torch.cat(observations, dim=0)

        observation_gpu = observation.to(self.device, non_blocking=True)

        mask_gpu = torch.stack(masks).to(self.device, non_blocking=True)

        with torch.no_grad():   

            policy_logits, values = self.model(observation_gpu)

        masked_logits = policy_logits + mask_gpu

        probabilities = torch.softmax(masked_logits, dim=-1)

        action = torch.multinomial(probabilities, num_samples=1).squeeze(1)

        return (observation, action.cpu(), mask)


    def collect_data(self, num_envs=16, num_games=20):

        envs = [GoEnv() for _ in range(num_envs)]

        trajectories = [[] for _ in range(num_envs)]

        experiences = []

        completed_games = 0

        while completed_games < num_games:
            observations, actions, masks = self.chose_move(envs)

            for env_idx, env in enumerate(envs):

                players = env.state.current_player()

                trajectories[env_idx].append({
                    "observation": observations[env_idx],
                    "action": actions[env_idx],
                    "player": players,
                    "mask": masks[env_idx]
                })

                env.step(actions[env_idx].item())

                if env.state.is_terminal():
                    returns = env.state.returns()

                    for step in trajectories[env_idx]:
                        step["returns"] = returns[step["player"]]

                    experiences.extend(trajectories[env_idx])

                    trajectories[env_idx] = []

                    completed_games += 1

                trajectories[env_idx] = []

                if num_games >= completed_games:
                    env.reset()


        return experiences

    def train_batch(self, samples):

        observations = torch.cat([
            sample["observation"]
            for sample in samples
        ], dim=0).to(self.device, non_blocking=True)

        actions = torch.tensor(
            [sample["action"] for sample in samples],
            dtype=torch.long,
            device=self.device
        ).to(self.device, non_blocking=True)

        players = torch.tensor(
            [sample["player"] for sample in samples],
            dtype=torch.long,
            device=self.device
        ).to(self.device, non_blocking=True)

        targets = torch.tensor([
            [sample["returns"]] for sample in samples],
            dtype=torch.float32,
            device=self.device
        ).squeeze().to(self.device, non_blocking=True)

        masks = torch.stack(
            [sample["mask"] for sample in samples]
        ).to(self.device, non_blocking=True)

        policy_logits, values = self.model(observations)

        masked_logits = policy_logits + masks

        values = values.squeeze(-1)

        value_loss = F.mse_loss(
            values,
            targets
        )

        log_probs = F.log_softmax(
            masked_logits,
            dim=-1
        )

        chosen_log_probs = log_probs[
            torch.arange(
                len(actions),
                device=self.device
            ),
            actions
        ]

        advantages = targets - values.detach()

        policy_loss = -(
            chosen_log_probs * advantages
        ).mean()

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def train(self, num_games=1_000, num_envs=16):

        print("Training has started...")

        for game_number in range(num_games):

            game_data, returns = self.collect_data(num_envs=num_envs)

            loss, policy_loss, value_loss = self.train_batch(game_data, returns)

            print(
                f"Game {game_number + 1}/{num_games} | "
                f"Moves: {len(game_data)} | "
                f"Result: {returns} | "
                f"Loss: {loss:.4f} | "
                f"Policy: {policy_loss:.4f} | "
                f"Value: {value_loss:.4f} | "
            )

        print("Training has ended...")


