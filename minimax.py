import chess
import random
import time

def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 10000
    }
    
    value = 0
    for piece in piece_values:
        value += len(board.pieces(piece, chess.WHITE)) * piece_values[piece]
        value -= len(board.pieces(piece, chess.BLACK)) * piece_values[piece]
    
    return value

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}

PST = {
    chess.PAWN: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10, -20, -20,  10,  10,   5,
         5,  -5, -10,   0,   0, -10,  -5,   5,
         8,   0,   0,  20,  20,   0,   0,   8,
         8,   5,  10,  25,  25,  10,   5,   8,
        10,  10,  20,  30,  30,  20,  10,  10,
        50,  50,  50,  50,  50,  50,  50,  50,
         0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10,  10,  10,  10,  10,   5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   5,   5,   0,   0,   0,
    ],
    chess.QUEEN: [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   5,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20,
    ],
}

WEIGHTS = {
    "material":   1.0,
    "pst":        0.1,
    "mobility":   0.05,
    "king_safety":0.2,
    "pawn_struct":0.1,
    "bishop_pair":0.05,
    "center":     0.02,
}

CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]

def is_endgame(board):
    # Simple: only kings and pawns, or very low material
    material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values())
    return material <= 1300  # Tune this threshold

def king_activity(board):
    # Encourage king to center in endgame
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_score = max(0, 4 - min(abs(chess.square_file(white_king) - 3.5), abs(chess.square_rank(white_king) - 3.5)))
    black_score = max(0, 4 - min(abs(chess.square_file(black_king) - 3.5), abs(chess.square_rank(black_king) - 3.5)))
    return white_score - black_score

def evaluate_board_v2(board):
    #Material + PST
    mat = 0
    pst_score = 0
    endgame = is_endgame(board)
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        mat += val if piece.color == chess.WHITE else -val
        idx = sq if piece.color == chess.WHITE else (63 - sq)
        pst_score += (PST[piece.piece_type][idx]
                      if piece.color == chess.WHITE
                      else -(PST[piece.piece_type][idx]) - 1)

    #Mobility
    board_turn = board.turn
    board.turn = chess.WHITE
    w_moves = board.legal_moves.count()
    board.turn = chess.BLACK
    b_moves = board.legal_moves.count()
    board.turn = board_turn  # restore
    mobility = w_moves - b_moves

    #King Safety
    def pawn_shield(color):
        ksq = board.king(color)
        r = chess.square_rank(ksq)
        f = chess.square_file(ksq)
        shield_rank = r + (1 if color == chess.WHITE else -1)
        cnt = 0
        for df in (-1, 0, +1):
            ff = f + df
            if 0 <= ff < 8 and 0 <= shield_rank < 8:
                sq2 = chess.square(ff, shield_rank)
                p = board.piece_at(sq2)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    cnt += 1
        return cnt

    w_attack = len(board.attackers(chess.BLACK, board.king(chess.WHITE)))
    b_attack = len(board.attackers(chess.WHITE, board.king(chess.BLACK)))
    king_safety = (pawn_shield(chess.WHITE) - w_attack) - (pawn_shield(chess.BLACK) - b_attack)

    #Pawn Structure (isolated, doubled, passed)
    def pawn_structure(color):
        pawns = list(board.pieces(chess.PAWN, color))
        file_counts = {}
        for sq in pawns:
            f = chess.square_file(sq)
            file_counts[f] = file_counts.get(f, 0) + 1
        iso = 0; dbl = 0; passed = 0
        for sq in pawns:
            f = chess.square_file(sq); r = chess.square_rank(sq)
            # doubled
            if file_counts[f] > 1: dbl += 1
            # isolated
            if file_counts.get(f-1,0) + file_counts.get(f+1,0) == 0:
                iso += 1
            # passed
            enemy = board.pieces(chess.PAWN, not color)
            ahead = range(r+1,8) if color==chess.WHITE else range(r-1,-1,-1)
            blocked = False
            for e in enemy:
                ef, er = chess.square_file(e), chess.square_rank(e)
                if ef in (f-1,f,f+1) and er in ahead:
                    blocked = True
                    break
            if not blocked: passed += 1
        return -15*iso -15*dbl + (25 if not endgame else 50)*passed

    pawn_struct = pawn_structure(chess.WHITE) - pawn_structure(chess.BLACK)

    #Bishop Pair
    bp = (2 <= len(board.pieces(chess.BISHOP, chess.WHITE))) - (2 <= len(board.pieces(chess.BISHOP, chess.BLACK)))
    bishop_pair = 50 * bp

    #Center Control
    center = sum(len(board.attackers(chess.WHITE, sq)) for sq in CENTER_SQUARES) \
           - sum(len(board.attackers(chess.BLACK, sq)) for sq in CENTER_SQUARES)
    
    king_act = 0
    if endgame:
        king_act = 20 * king_activity(board)

    score = (
        WEIGHTS["material"]    * mat
      + WEIGHTS["pst"]         * pst_score
      + WEIGHTS["mobility"]    * mobility
      + WEIGHTS["king_safety"] * king_safety
      + WEIGHTS["pawn_struct"] * pawn_struct
      + WEIGHTS["bishop_pair"] * bishop_pair
      + WEIGHTS["center"]      * center
      + king_act
    )
    return score

def minimax(board, depth, alpha, beta, maximizing_player, trans_table=None):
    if trans_table is None:
        trans_table = {}

    board_fen = board.fen()
    tt_key = (board_fen, depth, maximizing_player)
    if tt_key in trans_table:
        return trans_table[tt_key]

    if len(list(board.legal_moves)) == 0:
        if board.is_checkmate():
            result = (None, float('-inf') if maximizing_player else float('inf'))
        else:
            result = (None, 0)
        trans_table[tt_key] = result
        return result

    best_move = random.choice(list(board.legal_moves))
    if depth == 0 or board.is_game_over():
        result = (best_move, evaluate_board_v2(board))
        trans_table[tt_key] = result
        return result

    if maximizing_player:
        max_eval = float('-inf')
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, trans_table)[1]
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        result = (best_move, max_eval)
        trans_table[tt_key] = result
        return result
    else:
        min_eval = float('inf')
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, trans_table)[1]
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        result = (best_move, min_eval)
        trans_table[tt_key] = result
        return result

def iterative_deepening(board, max_depth, maximizing_player, time_limit=None):
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth + 1):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        move, _ = minimax(board, depth, float('-inf'), float('inf'), maximizing_player)
        if move is not None:
            best_move = move
        if time_limit and (time.time() - start_time) > time_limit:
            break
    return best_move