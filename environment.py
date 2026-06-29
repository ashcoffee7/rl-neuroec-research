# import libraries
import numpy as np

class OrganAllocationEnv:
    def __init__(self, demand, supply, expiration):
        self.reset()

    def reset(self, organs=20, waitlist = 300, urgent=50, wait_time = 100):
        self.organs = organs
        self.waitlist = waitlist
        self.urgent = urgent
        self.wait_time = wait_time
        self.expired = 0
        self.transplanted = 0
        return self.state()
    
    def state(self):
        return np.array([
            self.organs,
            self.waitlist,
            self.urgent,
            self.wait_time,
            self.expired
        ])
    
    def step(self, action):
        """
        Actions
        0 = urgency priority
        1 = wait-time priority
        2 = balanced
        """
        # 1. allocate organs
        if action == 0:
            # then every patient should be served
            served = min(self.organs, self.urgent)
        
        elif action == 1:
            # then serve patients depending on waitlist
            served = min(self.organs, self.waitlist)
    
        else:
            priority_pool = int(0.7 * self.urgent + 0.3 * self.waitlist)
            served = min(self.organs, priority_pool)
    
        # update counts
        self.organs -= served
        self.waitlist -= served
        self.urgent -= min(served, self.urgent)

        # update reward
        reward = (served - 3 * self.expired)

        # simultae next period using Poisson distribution
        self.organs += np.random.poisson(5)
        self.waitlist += np.random.poisson(20)
        self.urgent += np.random.poisson(5)

        # set expired
        self.expired = np.random.binomial(self.organs, 0.05)
        self.organs -= self.expired
        
        # return
        done = False
        return (self.state(), reward, done, {})
