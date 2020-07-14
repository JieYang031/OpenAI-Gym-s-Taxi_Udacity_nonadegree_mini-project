import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.1, gamma=1, epsilon=0.005):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1- self.epsilon + self.epsilon /self.nA
        
        return np.random.choice(np.arange(self.nA), p = policy_s)
   
    def return_policy(self, state):
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1- self.epsilon + self.epsilon /self.nA
        return policy_s
        
        

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward     + self.gamma * np.dot(self.Q[next_state], self.return_policy(next_state)) - self.Q[state][action] )
        