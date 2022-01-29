import gym.spaces as spaces
from gym import ActionWrapper


class FlattenAction(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, action):
        return spaces.unflatten(self.env.action_space, action)
