import chess
import model
import minimax

def handle_click(game, pos):
    if not game.game_started:
        if game.button['play'].rect.collidepoint(pos):
            game.game_started = True
        elif game.button['mode_list']['0P'].rect.collidepoint(pos):
            game.mode = "0P"
        elif game.button['mode_list']['1P'].rect.collidepoint(pos):
            game.mode = "1P"
        elif game.button['mode_list']['2P'].rect.collidepoint(pos):
            game.mode = "2P"
        
        if (game.mode == "0P"):
            for model in game.button['model-algo-1']:
                if game.button['model-algo-1'][model].rect.collidepoint(pos):
                    game.ai_white = model
            for model in game.button['model-algo-2']:
                if game.button['model-algo-2'][model].rect.collidepoint(pos):
                    game.ai_black = model
        
        elif (game.mode == "1P"):
            for model in game.button['model-algo-2']:
                if game.button['model-algo-2'][model].rect.collidepoint(pos):
                    game.ai_black = model
    else:
        if game.promoting:
            for promote, (x, y) in game.promoting_square.items():
                if x <= pos[0] <= x + game.SQUARE_SIZE // 2 and y <= pos[1] <= y + game.SQUARE_SIZE // 2:
                    move = chess.Move(game.selected_square, chess.square(game.promoting_draw_pos[0], game.promoting_draw_pos[1]), promotion=promote)
                    if move in game.board.legal_moves:
                        game.highlight_last_move.clear()
                        game.highlight_last_move.append(chess.square(game.promoting_draw_pos[0], game.promoting_draw_pos[1]))
                        game.highlight_last_move.append(game.selected_square)
                        game.board.push(move)
                        game.promoting_draw_pos = [-1, -1]
                        game.promoting = False
                        game.selected_square = None
                    return
        if game.button['undo'].rect.collidepoint(pos):
            if len(game.board.move_stack) > 0:
                game.board.pop()
                if (game.mode == "1P"): game.board.pop()
                game.selected_square = None
                game.promoting_draw_pos = [-1, -1]
                game.highlight_last_move.clear()
                game.promoting = False
            return        
        
        if game.button['retry'].rect.collidepoint(pos):
            game.selected_square = None
            game.promoting_draw_pos = [-1, -1]
            game.promoting = False
            game.highlight_last_move.clear()
            game.board = chess.Board()
            return
        
        if game.button['menu'].rect.collidepoint(pos):
            game.selected_square = None
            game.promoting_draw_pos = [-1, -1]
            game.promoting = False
            game.highlight_last_move.clear()
            game.board = chess.Board()
            game.game_started = False
            return

        col = (pos[0] - game.padding) // game.SQUARE_SIZE
        row = 7 - (pos[1] - game.padding) // game.SQUARE_SIZE
        if col < 0 or col >= game.WIDTH or row < 0 or row >= game.HEIGHT:
            return
        square = chess.square(col, row)
        #print(square)
        if game.selected_square is not None:
            available_moves = []
            move = chess.Move(game.selected_square, square)
            if move in game.board.legal_moves:
                available_moves.append(move)
            for promoting in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(game.selected_square, square, promotion=promoting)
                #print(move.uci())
                if move in game.board.legal_moves:
                    available_moves.append(move)

            #print(available_moves)
            if len(available_moves) == 0:
                game.selected_square = None
                game.promoting_draw_pos = [-1, -1]
                game.promoting = False
            elif len(available_moves) == 1:
                game.board.push(available_moves[0])
                game.promoting_draw_pos = [-1, -1]
                game.highlight_last_move.clear()
                game.highlight_last_move.append(square)
                game.highlight_last_move.append(game.selected_square)
                game.promoting = False
                game.selected_square = None
            else:
                game.promoting = True
                if (row == 0):
                    game.promoting_draw_pos = [col, row]
                elif (row == 7):
                    game.promoting_draw_pos = [col, row]
        else:
            piece = game.board.piece_at(square)
            if piece and piece.color == game.board.turn:
                game.selected_square = square

def available_moves_from_square(game, square):
    moves = []
    for move in game.board.legal_moves:
        if move.from_square == square:
            moves.append(move.to_square)
    return moves

def minimax_move(game, depth = 3, maximizing_player=False):
    best_move, eval = minimax.minimax(game.board, depth, float('-inf'), float('inf'), maximizing_player)
    if best_move:
        return best_move
    return None

def minimax_iterative(game, depth = 3, maximizing_player = False, time_limit = 5):
    best_move = minimax.iterative_deepening(game.board, depth, maximizing_player, time_limit)
    if best_move:
        return best_move
    return None

def rnn_move(game):
    board_sequence = model.extract_board_sequence(game.board)
    predicted_move = model.predict_move(game.RNNModel, board_sequence, game.board)
    if predicted_move:
        return predicted_move
    return None

def lstm_move_256(game):
    board_sequence = model.extract_board_sequence(game.board)
    predicted_move = model.predict_move(game.LSTMModel256, board_sequence, game.board)
    if predicted_move:
        return predicted_move
    return None

def lstm_move_640(game):
    board_sequence = model.extract_board_sequence(game.board)
    predicted_move = model.predict_move(game.LSTMModel640, board_sequence, game.board)
    if predicted_move:
        return predicted_move
    return None

def cnn_lstm_move(game):
    board_sequence = model.extract_board_sequence(game.board)
    predicted_move = model.predict_move(game.CNN_LSTMModel, board_sequence, game.board)
    if predicted_move:
        return predicted_move
    return None

def AI_move_black(game):
    move = None
    if game.ai_black == "m1":
        move = minimax_move(game, 1, False)
    if game.ai_black == "m2":
        move = minimax_move(game, 2, False)
    if game.ai_black == "m3":
        move = minimax_move(game, 4, False)
    if game.ai_black == "lstm":
        move = lstm_move_256(game)
    if game.ai_black == "cnn-lstm":
        move = lstm_move_640(game)
    return move

def AI_move_white(game):
    move = None
    if game.ai_white == "m1":
        move = minimax_move(game, 1, True)
    if game.ai_white == "m2":
        move = minimax_move(game, 2, True)
    if game.ai_white == "m3":
        move = minimax_move(game, 4, True)
    if game.ai_white == "lstm":
        move = lstm_move_256(game)
    if game.ai_white == "cnn-lstm":
        move = lstm_move_640(game)
    return move
