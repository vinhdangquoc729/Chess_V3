import chess
import chess.engine
import AI  # Your AI.py
import game  # Your game.py

def play_vs_stockfish(stockfish_path, stockfish_elo=1350, bot_white=True, max_moves=200):
    # Setup
    board = chess.Board()
    game_instance = game.Game()
    game_instance.board = board
    game_instance.ai_white = "m3" 
    game_instance.ai_black = "m3"

    # Start Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        print(board)
        print("FEN:", board.fen())
        if (board.turn == chess.WHITE and bot_white) or (board.turn == chess.BLACK and not bot_white):
            # Your bot's move
            if board.turn == chess.WHITE:
                move = AI.AI_move_white(game_instance)
            else:
                move = AI.AI_move_black(game_instance)
            print("Bot move:", board.san(move))
        else:
            # Stockfish move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            print("Stockfish move:", board.san(move))
        board.push(move)
        move_count += 1

    print("Game over:", board.result())
    engine.quit()

if __name__ == "__main__":
    # Example usage
    play_vs_stockfish(
        stockfish_path = "stockfish-windows-x86-64-avx2.exe", 
        stockfish_elo = 1500,
        bot_white = True  
    )