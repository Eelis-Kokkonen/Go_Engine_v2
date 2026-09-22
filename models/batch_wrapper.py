import pyspiel


class GoEnv:

    def __init__(self, board_size=19):

        self.game = pyspiel.load_game(
            f"go(board_size={board_size})"
        )

        self.reset()

    def reset(self):

        self.state = (
            self.game.new_initial_state()
        )

        return self.state

    def step(self, action):

        self.state.apply_action(action)

        done = self.state.is_terminal()

        if done:
            returns = self.state.returns()
        else:
            returns = None

        return done, returns

    def legal_actions(self):

        return self.state.legal_actions()

    def current_player(self):

        return self.state.current_player()

    def is_terminal(self):

        return self.state.is_terminal()