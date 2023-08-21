# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodDistance = []
        foods = currentGameState.getFood().asList()
        if action == 'STOP':
            return -float("inf")
        for newGhostState in newGhostStates:
            newGhostPosition = newGhostState.getPosition()
            if newGhostPosition == newPos and newGhostState.scaredTimer == 0:
                return -float("inf")
        for food in foods:
            foodDistance.append(-abs(food[0]-newPos[0])-abs(food[1]-newPos[1]))
        foodPart = max(foodDistance)
        return foodPart + min(newScaredTimes)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        agentIndex = 0
        currDepth = 0
        return (self.getValue(gameState, agentIndex, currDepth))[0]

    def getValue(self, gameState, agentIndex, currDepth):
        numAgents = gameState.getNumAgents()
        if agentIndex >= numAgents:
            agentIndex = 0
            currDepth += 1
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, currDepth)
        else:
            return self.minVal(gameState, agentIndex, currDepth)

    def minVal(self, gameState,agentIndex, currDepth):
        value = ("action", float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            if action == "STOP":
                continue
            newVal = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, currDepth)
            if type(newVal) == tuple:
                newVal = newVal[1]
            if newVal < value[1]:
                value = (action, newVal)
        return value

    def maxVal(self, gameState, agentIndex, currDepth):
        value = ("action", -float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            if action == "STOP":
                continue
            newVal = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, currDepth)
            if type(newVal) == tuple:
                newVal = newVal[1]
            if newVal > value[1]:
                value = (action, newVal)
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentIndex = 0
        currDepth = 0
        alpha = -float("inf")
        beta = float("inf")
        return (self.getValue(gameState, agentIndex, currDepth, alpha, beta))[0]

    def getValue(self, gameState, agentIndex, currDepth, alpha, beta):
        numAgents = gameState.getNumAgents()
        if agentIndex >= numAgents:
            agentIndex = 0
            currDepth += 1
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, currDepth, alpha, beta)
        else:
            return self.minVal(gameState, agentIndex, currDepth, alpha, beta)

    def minVal(self, gameState,agentIndex, currDepth, alpha, beta):
        value = ("action", float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            if action == "STOP":
                continue
            newVal = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, currDepth, alpha, beta)
            if type(newVal) == tuple:
                newVal = newVal[1]
            if newVal < value[1]:
                value = (action, newVal)
            if value[1] < alpha:
                return value
            beta = min(beta, value[1])
        return value

    def maxVal(self, gameState, agentIndex, currDepth, alpha, beta):
        value = ("action", -float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            if action == "STOP":
                continue
            newVal = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, currDepth, alpha, beta)
            if type(newVal) == tuple:
                newVal = newVal[1]
            if newVal > value[1]:
                value = (action, newVal)
            if value[1] > beta:
                return value
            alpha = max(alpha, value[1])
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        agentIndex = 0
        currDepth = 0
        return (self.getValue(gameState, agentIndex, currDepth))[0]

    def getValue(self, gameState, agentIndex, currDepth):
        numAgents = gameState.getNumAgents()
        if agentIndex >= numAgents:
            agentIndex = 0
            currDepth += 1
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, currDepth)
        else:
            return self.expVal(gameState, agentIndex, currDepth)

    def expVal(self, gameState,agentIndex, currDepth):
        value = ("action", float("inf"))
        values = []
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            if action == "STOP":
                continue
            newVal = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, currDepth)
            if type(newVal) == tuple:
                newVal = newVal[1]
            values.append(newVal)
            value = (action, sum(values)/len(values))
        return value

    def maxVal(self, gameState, agentIndex, currDepth):
        value = ("action", -float("inf"))
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            if action == "STOP":
                continue
            newVal = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, currDepth)
            if type(newVal) == tuple:
                newVal = newVal[1]
            if newVal > value[1]:
                value = (action, newVal)
        return value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    My evaluation function considers the following factors:
    1. Distance to nearest food
    2. Distance to nearest ghost
    3. Current score
    4. Number of scared ghosts
    """
    foodDistance = []
    ghostDistance = []
    numScared = 0

    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacPosition = currentGameState.getPacmanPosition()

    for food in foods:
        foodDistance.append(-abs(food[0]-pacPosition[0])-abs(food[1]-pacPosition[1]))
    if len(foodDistance) == 0:
        foodDistance.append(0)
    for ghost in ghosts:
        if ghost.scaredTimer > 0:
            numScared += 1
            ghostDistance.append(0)
            continue
        ghostPosition = ghost.getPosition()
        x = abs(ghostPosition[0]-pacPosition[0])
        y = abs(ghostPosition[1]-pacPosition[1])
        if x + y == 0:
            ghostDistance.append(0)
        else:
            ghostDistance.append(-1/(x+y))
    return max(foodDistance) + min(ghostDistance) + currentGameState.getScore() + numScared

# Abbreviation
better = betterEvaluationFunction
