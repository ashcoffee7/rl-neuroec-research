# import libraries
import numpy as np

class OrganAllocationEnv:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, organs=20, waitlist = 300, urgent=50, wait_time = 100):
        self.organs = organs
        self.waitlist = waitlist
        self.urgent = urgent
        self.wait_time = wait_time
        self.expired = 0
        self.transplanted = 0
        self.day = 0
        return self.state()
    
    def state(self):
        return {
            "organs": self.organs,
            "waitlist": self.waitlist,
            "urgent": self.urgent,
            "waitlist": self.wait_time,
            "expired": self.expired,
            "day": self.day
        }
    
    def step(self, action):
        """
        Actions:
        0 = urgency focused
        1 = wait-time focused
        2 = balanced
        3 = waste avoidance
        4 = fairness focused
        """
        # organs before count
        organs_before = self.organs
        served = 0

        # 1. allocation organs
        if action == 0:
            # prioritize urgency
            served = min(self.organs, self.urgent)
        elif action == 1:
            # prioritize by waitlist
            nonurgent = max(self.waitlist - self.urgent, 0)
            served = min(self.organs, nonurgent)
        elif action == 2:
            # balanced
            priority_pool = int(0.7 * self.urgent + 0.3 * self.waitlist)
            served = min(self.organs, priority_pool)
        elif action == 3:
            # avoid waste, use everything before expiration
            served = self.organs
        elif action == 4:
            # fairness, slightly slower allocation
            served = int(0.8 * self.organs)
        
        # update states
        self.organs -= served
        self.waitlist -= served
        self.urgent = max(self.urgent - served, 0)
        self.transplanted += served

        # update reward
        reward = (served - 3 * self.expired)

        # new arrivals
        new_organs = self.rng.poisson(5)
        new_patients = self.rng.poisson(20)
        new_urgent = self.rng.poisson(5)
        self.organs += new_organs
        self.waitlist += new_patients
        self.urgent += new_urgent
        
        # expiration setting
        self.expired = self.rng.binomial(self.organs, 0.05)
        self.organs -= self.expired 
        self.day += 1
        done = self.day >= 365
        info = {

            "transplanted":
                self.transplanted,

            "expired":
                self.expired,

            "waiting":
                self.waitlist
        }


        return (
            self.state(),
            reward,
            done,
            info
        )

