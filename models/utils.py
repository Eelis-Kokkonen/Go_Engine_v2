import torch

def state_to_tensor(state, device):
    tensor = torch.tensor(state.observation_tensor(), dtype=torch.float32).reshape(4, 19, 19).to(device)

    black = tensor[0]
    white = tensor[1]

    if state.current_player() == 0:
        current = black
        opponent = white
    else:
        current = white
        opponent = black

    return torch.stack([current, opponent]).unsqueeze(0)
