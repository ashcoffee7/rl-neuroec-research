# -------- Environment Testing Files  -------------------------------------
# Testing file to evaluate policy environments
# -------------------------------------------------------------------------

# import libraries
from policyEnv.environment import OrganAllocationEnv
from policyEnv.rule_based import RuleBasedPolicy
from evaluation.evaluate import evaluate

# initialize envs
env = OrganAllocationEnv()

models = {

"Rule":
RuleBasedPolicy()

# "RL":
# "to be added",

# "SCRL":
# "to be added",

# "Neuro":
# "to be added"

}

# run reward calculation
for name, model in models.items():

    result = evaluate(
        env,
        model
    )

    print(name,result)