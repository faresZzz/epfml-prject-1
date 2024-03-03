from dataclasses import dataclass


@dataclass
class ExponentialLR:
    update_gamma_ratio: float
    initial_gamma: float

    def __post_init__(self):
        self.__i = 0

    def step(self):
        self.__i += 1

    @property
    def gamma(self):
        return self.initial_gamma * (self.update_gamma_ratio ** self.__i)

    def reset(self):
        self.__i = 0
