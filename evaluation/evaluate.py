# -------- Rule-Based Policy Class -------------------------------------
# Evaluating the RL models based on reward
# NOTE: AM ADDING OTHER EVALUATION PROTOCOLS LATER ON (e.g. fairness etc)
# ----------------------------------------------------------------------

def evaluate(env, policy):
    """
    Given environment, evaluates reward to discover the model that best minimizes
    waste or organs.

    Parameters:
        env (str): name of the current policy environment
        policy (Policy): one of the policy environments established in policyEnv
    """
    # initialization
    state = env.reset()
    total_reward = 0

    # until done, run loop
    while True:
        action = policy.choose_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done: 
            break

    return {"reward": total_reward, **info}

