import pygame
import chess

def draw(game):
    game.screen.fill((255, 255, 255))
    game.screen.blit(game.bg, (0, 0))
    if not game.game_started:
        game.button['play'].draw(game.screen)
        for mode in game.button['mode_list']:
            game.button['mode_list'][mode].draw(game.screen)
        if game.mode == "0P":
            for model1 in game.button['model-algo-1']:
                if model1 == game.ai_white:
                    game.button['model-algo-1'][model1].selecting = True
                else:
                    game.button['model-algo-1'][model1].selecting = False
                game.button['model-algo-1'][model1].draw(game.screen)
            for model2 in game.button['model-algo-2']:
                if model2 == game.ai_black:
                    game.button['model-algo-2'][model2].selecting = True
                else:
                    game.button['model-algo-2'][model2].selecting = False
                game.button['model-algo-2'][model2].draw(game.screen)
        if game.mode == "1P":
            for model2 in game.button['model-algo-2']:
                if model2 == game.ai_black:
                    game.button['model-algo-2'][model2].selecting = True
                else:
                    game.button['model-algo-2'][model2].selecting = False
                game.button['model-algo-2'][model2].draw(game.screen)
    else:
        draw_frame(game)
        draw_board(game)
        draw_pieces(game)
        draw_highlight(game)
        draw_promotion(game)
        draw_letter(game)
        draw_lost_pieces(game)
        game.button['undo'].draw(game.screen)
        game.button['retry'].draw(game.screen)
        game.button['menu'].draw(game.screen)
        if (game.board.is_game_over()) :
            if (game.board.result() == "1-0"):
                game.screen.blit(game.text['white'][0], game.text['white'][1])
            elif (game.board.result() == "0-1"):
                game.screen.blit(game.text['black'][0], game.text['black'][1])
            elif (game.board.is_stalemate()):
                game.screen.blit(game.text['stale'][0], game.text['stale'][1])
            else:
                game.screen.blit(game.text['none'][0], game.text['none'][1])

def draw_frame(game):
    frame_border = game.SQUARE_SIZE // 2
    rect = pygame.Rect(game.padding - frame_border, game.padding - frame_border, 
                        game.SQUARE_SIZE * game.WIDTH + 4 * frame_border, 
                        game.SQUARE_SIZE * game.HEIGHT + 4 * frame_border)
    game.frame = pygame.transform.scale(game.frame, (game.SQUARE_SIZE * game.WIDTH + 2 * frame_border, 
                                                        game.SQUARE_SIZE * game.HEIGHT + 2 * frame_border))
    game.screen.blit(game.frame, rect)

def draw_board(game):
    colors = [pygame.Color(255, 206, 158), pygame.Color(209, 139, 71)]
    selected_color = [pygame.Color(204, 164, 126), pygame.Color(132, 88, 45)]
    for row in range(8):
        for col in range(8):
            if game.selected_square is not None and (row, col) == (7 - game.selected_square // 8, game.selected_square % 8):
                color = selected_color[(row + col) % 2]
            else:
                color = colors[(row + col) % 2]
            pygame.draw.rect(game.screen, color, (col * game.SQUARE_SIZE + game.padding, row * game.SQUARE_SIZE + game.padding, 
                                                game.SQUARE_SIZE, game.SQUARE_SIZE))
            if ((7 - row) * 8 + col) in game.highlight_last_move:
                pygame.draw.rect(game.screen, (127, 127, 0), (col * game.SQUARE_SIZE + game.padding, row * game.SQUARE_SIZE + game.padding, 
                                                            game.SQUARE_SIZE, game.SQUARE_SIZE), 3)

def draw_highlight(game):
    if game.selected_square is not None:
        moves = game.available_moves_from_square(game.selected_square)
        for move in moves:
            row, col = divmod(move, 8)
            if (game.board.piece_at(move)):
                pygame.draw.circle(game.screen, (150, 105, 75), ((col * game.SQUARE_SIZE + game.padding + game.SQUARE_SIZE // 2), 
                                                    (7 - row) * game.SQUARE_SIZE + game.padding + game.SQUARE_SIZE // 2), 
                                                    game.SQUARE_SIZE // 2 - 4, 6)
            else:
                pygame.draw.circle(game.screen, (125, 73, 13), ((col * game.SQUARE_SIZE + game.padding + game.SQUARE_SIZE // 2), 
                                                    (7 - row) * game.SQUARE_SIZE + game.padding + game.SQUARE_SIZE // 2), 
                                                    game.SQUARE_SIZE // 6)

def draw_pieces(game):
    piece_images = {
        'P': 'img/wp.png',
        'N': 'img/wn.png',
        'B': 'img/wb.png',
        'R': 'img/wr.png',
        'Q': 'img/wq.png',
        'K': 'img/wk.png',
        'p': 'img/bp.png',
        'n': 'img/bn.png',
        'b': 'img/bb.png',
        'r': 'img/br.png',
        'q': 'img/bq.png',
        'k': 'img/bk.png'
    }
    for square in chess.SQUARES:
        piece = game.board.piece_at(square)
        if piece:
            image = pygame.image.load(piece_images[piece.symbol()])
            x = (square % 8) * game.SQUARE_SIZE + game.padding
            y = (7 - square // 8) * game.SQUARE_SIZE + game.padding
            image = pygame.transform.scale(image, (game.SQUARE_SIZE, game.SQUARE_SIZE))
            game.screen.blit(image, (x, y))

def draw_promotion(game):
    if (game.promoting_draw_pos[0] == -1): return
    if (game.promoting_draw_pos[1] == -1): return
    col, row = game.promoting_draw_pos
    coor_x = game.padding + game.SQUARE_SIZE * col - game.SQUARE_SIZE // 2
    coor_y = game.padding + game.SQUARE_SIZE * (7 - row) + ((-game.SQUARE_SIZE // 2) if row == 7 else (game.SQUARE_SIZE))
    promotion_images = {
        chess.QUEEN: 'img/wq.png',
        chess.ROOK: 'img/wr.png',
        chess.BISHOP: 'img/wb.png',
        chess.KNIGHT: 'img/wn.png'
    } if game.board.turn == chess.WHITE else {
        chess.QUEEN: 'img/bq.png',
        chess.ROOK: 'img/br.png',
        chess.BISHOP: 'img/bb.png',
        chess.KNIGHT: 'img/bn.png'
    }
    for i, (promote, image_path) in enumerate(promotion_images.items()):
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (game.SQUARE_SIZE // 2, game.SQUARE_SIZE // 2))
        x = coor_x + (i * game.SQUARE_SIZE // 2) 
        y = coor_y
        pygame.draw.rect(game.screen, (255, 206, 158), (x, y, game.SQUARE_SIZE // 2, game.SQUARE_SIZE // 2))
        pygame.draw.rect(game.screen, (204, 164, 126), (x, y, game.SQUARE_SIZE // 2, game.SQUARE_SIZE // 2), 3)
        game.screen.blit(image, (x, y))
        game.promoting_square.update({promote: (x, y)})

def draw_letter(game):
    for letter in game.letters:
        game.screen.blit(letter[0], letter[1])

def draw_lost_pieces(game):
    """
    Draws the lost pieces for both sides around (700, game.padding + 400) and below.
    """
    lost = count_lost_pieces(game.board)
    piece_images = {
        'P': 'img/wp.png',
        'N': 'img/wn.png',
        'B': 'img/wb.png',
        'R': 'img/wr.png',
        'Q': 'img/wq.png',
        'p': 'img/bp.png',
        'n': 'img/bn.png',
        'b': 'img/bb.png',
        'r': 'img/br.png',
        'q': 'img/bq.png'
    }
    x_base = 700
    y_base = game.padding + 470
    x, y = x_base, y_base
    for i, piece in enumerate(['Q', 'R', 'B', 'N', 'P']):
        count = lost[piece]
        if count < 0: continue
        for j in range(count):
            img = pygame.image.load(piece_images[piece])
            img = pygame.transform.scale(img, (game.SQUARE_SIZE // 2, game.SQUARE_SIZE // 2))
            game.screen.blit(img, (x + (j * (game.SQUARE_SIZE // 4 + 2)), y))
        y += game.SQUARE_SIZE // 4 + 10

    y = (game.SQUARE_SIZE // 4 + 10) * 3 + (game.padding)
    for i, piece in enumerate(['q', 'r', 'b', 'n', 'p']):
        count = lost[piece]
        if count < 0: continue
        for j in range(count):
            img = pygame.image.load(piece_images[piece])
            img = pygame.transform.scale(img, (game.SQUARE_SIZE // 2, game.SQUARE_SIZE // 2))
            game.screen.blit(img, (x + (j * (game.SQUARE_SIZE // 4 + 2)), y))
        y -= (game.SQUARE_SIZE // 4 + 10)

def count_lost_pieces(board):
    starting_counts = {
        'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1,
        'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1
    }
    current_counts = {k: 0 for k in starting_counts}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            if symbol in current_counts:
                current_counts[symbol] += 1
    lost = {k: starting_counts[k] - current_counts[k] for k in starting_counts}
    return lost