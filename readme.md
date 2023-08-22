Turn-based game neural networks based on DQN and MCTS
--------
What should be done for you game:
--------
stringRepresentation: A string representing the state of your game at the current time.

getBoardSize: Get the board size.

getActionSize: Get the action size.

getInitUnitNumber: Get the Init Number of players of both size.

getInitBoard: Get a start board.

getUnitsOfType: Get all the units of a certain type.

getUnitOfIdType: Get the player of certain type of a certain id.

getStateDimension: Get the dimension for the input of network.

getUnitSight: Get the sight of an unit. This is according to the game.

getBlocksInSight: Get all the blocks in sight. This is according to the game.

targetToOnehot: Change target id to onehot.

getState: Turn the board into an array that can be put into the network.

getActiveAction: Get the init active array of each turn.

getCanonicalForm: Change the side of active and passive.

getTurnEndResults: Get the end-of-round settlements. This is according to the game.

getGameEnded: Judge if the active part wins or not.

gaussian2d: Get a gaussian function on the head of the enemies.

getPeriodSkillValidActions: Get the validation for skills with multiple steps.

getSightBoard: Get the board in sight.

getValidActions: Get all the valid actions in this step.

getNextState: Do one step and get the board of next state.

getSymmetries: Get the symmetries of a board for information enhancement.

Attention: Some fonction may be according to the game while others according to the network.

Attention: We must always keep the dimension of all the arrays used as an input.

Infos for params:
--------
numIters : Epoch to run.

numMCTSSims: Number of consideration before making each decision of each game.

tempThreshold: Random intensity, the first tempThreshold steps of a game will be highly random, the later steps less random.

arenaCompare: Number of games played by Arena.

updateThreshold: Threshold for Arena to accept new models.

maxlenOfQueue: Number of steps gathered by the Coach each epoch.

numItersForTrainExamplesHistory: Maximum queues gathered by the Coach, each queue is of length maxlenOfQueue.

cpuct: The ratio of intensity to update the Action's Value for the current Q-value.

stepforce: The intensity to punish the network according to the steps. We always want lower steps when finishing the game.

load_model: Load the previous model or not.

checkpoint: The root for all the temp files.

load_file: Name of the model and the examples to be loaded. -1 means the last examples.

skipFirst: Whether the examples loaded are trained and compared or not. If yes, skip the first gathering step.

Components of Network
--------
main.py: Used to initialise the Game, the network (nn) and the trainer (Coach), read in the results of previous train, and start Coach.

Coach.py: The main code of the trainer, which is responsible for calling MCTS and selecting an action based on the probabilities it returns. When the game is over, it collects all the examples under a certain maximum length and sends them to the network for training. It uses symmetries for information enhancement. Finally, it calls the Arena to determine whether to keep the model or not.

MCTS.py: The main code for tree search, which uses the network to make a judgement about the current node and continue down the tree if it has children, otherwise a new child is created. Finally, it updates the weights (Q-values) of all the process before based on the results of the game. Draw is traited as 0 for the update. After completing the search, it returns the probabilities based on the number of times each action has been retrieved in the current state.

Arena.py: The main code of the comparator, which compares the two networks before and after training by operating a few actual confrontations, and decides whether to keep the new network or not.

Attention: This is a self-game model, where we always set the action side to 1 and the passive side to 2!

Game Example:
--------
The game.py contains an example of the game, supporting all basic actions such as move and attack, as well as skills and buffs. The game_net.py contains the network for the game.

Attention: This algorithme actually supports a very large number of games. By changing the game and the actual structure of the game_net, the algorithme can be easily moved to other games.

Attention: The package zodiac is not the game, it is just a small test for the network. :)

Attention: If you have the code of the previous version, it contains with serious bugs and has very limited fonctions which are all covered in the new version. Please cover the file directly.

