# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        updated_values = util.Counter()
        for iteration in range(self.iterations):
          for state in self.mdp.getStates():
            new_values = []
            if self.mdp.isTerminal(state):
              new_values.append(0)
            for action in self.mdp.getPossibleActions(state):
              new_values.append(self.computeQValueFromValues(state, action))
            updated_values[state] = max(new_values)
          self.values = updated_values.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        tran_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        values = []
        for tran_prob in tran_probs:
          reward = self.mdp.getReward(state, action, tran_prob[0])
          value = tran_prob[1] * (reward + self.discount * self.values[tran_prob[0]])
          values.append(value)
        return sum(values)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
          return None
        max_q_value = -float("inf")
        next_action = None
        for action in actions:
          q_value = self.computeQValueFromValues(state, action)
          if q_value > max_q_value:
            max_q_value = q_value
            next_action = action
        return next_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        num_states = len(states)
        for iteration in range(self.iterations):
          i = iteration % num_states
          state = states[i]
          if self.mdp.isTerminal(state):
            continue
          actions = self.mdp.getPossibleActions(state)
          if not actions:
            self.values[state] = self.discount * self.values[state]
          else:
            new_values = []
            for action in actions:
              new_values.append(self.computeQValueFromValues(state, action))
            self.values[state] = max(new_values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = dict()
        states = self.mdp.getStates()
        for state in states:
          predecessor = set()
          for prev in states:
            actions = self.mdp.getPossibleActions(prev)
            if not actions:
              continue
            for action in actions:
              tran_probs = self.mdp.getTransitionStatesAndProbs(prev, action)
              for tran_prob in tran_probs:
                if tran_prob[0] == state and tran_prob[1] > 0:
                  predecessor.add(prev)
          predecessors[state] = predecessor
        p_q = util.PriorityQueue()
        for state in states:
          if self.mdp.isTerminal(state):
            continue
          value = self.values[state]
          acns = self.mdp.getPossibleActions(state)
          if not acns:
            new_value = self.discount * value
          else:
            new_values = []
            for acn in acns:
              new_values.append(self.computeQValueFromValues(state, acn))
            new_value = max(new_values)
          diff = abs(value - new_value)
          p_q.push(state, -1 * diff)
        for iteration in range(self.iterations):
          if p_q.isEmpty():
            return
          s = p_q.pop()
          v = self.values[s]
          acts = self.mdp.getPossibleActions(s)
          if not acts:
            new_v = self.discount * v
          else:
            new_vs = []
            for act in acts:
              new_vs.append(self.computeQValueFromValues(s, act))
            new_v = max(new_vs)
          self.values[s] = new_v
          for p in predecessors[s]:
            p_value = self.values[p]
            p_actions = self.mdp.getPossibleActions(p)
            if not p_actions:
              new_p_value = self.discount * p_value
            else:
              new_p_values = []
              for p_action in p_actions:
                new_p_values.append(self.computeQValueFromValues(p, p_action))
              new_p_value = max(new_p_values)
            p_diff = abs(p_value - new_p_value)
            if p_diff > self.theta:
              p_q.update(p, -1*p_diff)
