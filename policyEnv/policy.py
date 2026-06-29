# -------- Policies Class -------------------
# Ensuring all policy code has the same base.
# -------------------------------------------

# import libraries
from abc import ABC, abstractmethod
class Policy(ABC):
    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError