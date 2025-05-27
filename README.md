## Chess AI with Minimax and LSTM

This project is a chess-playing software that implements two types of AI:
- Minimax with alpha-beta pruning: a classic algorithm for decision making in games.
- LSTM (Long Short-term Memory): a neural network model that learns to predict the next best move based on the board state.

The goal is to compare the effectiveness of rule-based search versus learned behavior, and to explore future improvements through reinforcement learning.

## Members
- Dang Quoc Vinh - 20235247
- Nguyen Cong Minh - 20235161
- Le Hoang Phuc - 20235190

## Installation
1. Clone the repository: git clone https://github.com/vinhdangquoc729/Chess_V3.git
2. Install dependencies: pip install -r requirements.txt
3. Run the program: python main.py

## How to use

When starting the program, game menu will appear with some options for you to choose:

- 1-player (default): Let you play against the bot. You can choose the algorithm/model for the bot.
![image](https://github.com/user-attachments/assets/dc34483d-01c6-466e-8a43-6f77c2e896f3)
  * Minimax 1: Minimax algorithm with depth 1
  * Minimax 2: Minimax algorithm with depth 2
  * Minimax 3: Minimax algorithm with depth 4
- Self-play: Let 2 bots play against each other. You can choose the algorithm/model for each bot.
  ![image](https://github.com/user-attachments/assets/896af75d-a4c1-44bd-8c26-c4f3c995fbe1)
- 2-player: Let you play with your friend.

After choosing all the options you want, click **Start Game** to start playing.

You can undo the moves, even restart the game whenever you want.
![image](https://github.com/user-attachments/assets/62690764-353e-45fa-bf06-65fb5519a2b7)

*Enjoy your game!*
