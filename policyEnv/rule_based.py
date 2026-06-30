# -------- Rule-Based Policy Class -------------------------------------
# Creating the vanilla model, first comparison with the final RL model.
# ----------------------------------------------------------------------

# import libraries
from policyEnv.policy import Policy

class RuleBasedPolicy(Policy):
    def choose_action(self, state):
        """
        Chooses action based on a rule-based system depending on the number
        of organs, expired organs, and urgent patients.

        Parameters:
            state (dict): describes current state of available organs, expired organs,
                and patients in urgent status which informs policy decision
        """
        organs = state["organs"]
        expired = state["expired"]
        urgent = state["urgent"]

        # if organs are at risk of loss, prioritize using then
        if expired > 0 or organs > 30:
            return 3
        
        # if there are many urgent patients
        if urgent > 6:
            return 0

        # if it's balanced
        return 2