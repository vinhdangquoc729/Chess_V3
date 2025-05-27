import chess
import pygame
import torch
import model
import gui
import AI

class Button:
    def __init__(self, text, x, y, width, height, color=(0, 255, 0), selecting_color = (255, 0, 0), hover_color = (0, 255, 0), font_size = 28):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.font = pygame.font.Font('font/CrimsonPro.ttf', font_size)
        self.hover_color = hover_color
        self.selecting = False
        self.selecting_color = selecting_color

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        if self.selecting:
            current_color = self.selecting_color
        else: 
            current_color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(screen, current_color, self.rect, border_radius = 15)
        if not self.selecting:
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 3, border_radius = 15)
        else:
            pygame.draw.rect(screen, (255, 255, 255), self.rect, 5, border_radius = 15)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess Game")
        self.SQUARE_SIZE = 75
        self.WIDTH = 8
        self.HEIGHT = 8
        self.SCREEN_WIDTH = 900
        self.SCREEN_HEIGHT = 700
        self.font = pygame.font.Font(None, 36)
        self.letter_font = pygame.font.Font(None, 24)
        self.mode = "1P" #or "2P"
        self.padding = (self.SCREEN_HEIGHT - (self.HEIGHT * self.SQUARE_SIZE)) // 2
        self.board = chess.Board()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.above_rect = pygame.Rect(self.padding, 10, self.SQUARE_SIZE * self.WIDTH, self.padding)
        self.button = {
            "play": Button("Start Game", 300, 220, 300, 102, color=(220, 150, 100), hover_color=(230, 160, 110), selecting_color=(200, 140, 90), font_size=48),
            "undo": Button("Undo", 700, self.padding + 350, 160, 80, color=(180, 120, 80), hover_color=(210, 160, 100), selecting_color=(160, 110, 70)),
            "retry": Button("Retry", 700, self.padding + 250, 160, 80, color=(180, 120, 80), hover_color=(210, 160, 100), selecting_color=(150, 100, 60)),
            "menu": Button("Menu", 700, self.padding + 150, 160, 80, color=(180, 120, 80), hover_color=(210, 160, 100), selecting_color=(150, 100, 60)),
            "mode_list": {
                "0P": Button("Self-play", 100, 360, 200, 80, color=(200, 148, 98), hover_color=(210, 158, 108), selecting_color=(180, 138, 88)),
                "1P": Button("1-player", 350, 360, 200, 80, color=(180, 128, 78), hover_color=(190, 138, 88), selecting_color=(160, 118, 68)),
                "2P": Button("2-player", 600, 360, 200, 80, color=(160, 108, 58), hover_color=(170, 118, 68), selecting_color=(140, 98, 48)),
            },
            "model-algo-1": {
                "m1": Button("Minimax 1", 15, 590, 150, 80, color=(205, 133, 63), hover_color=(215, 143, 73), selecting_color=(185, 123, 53)),
                "m2": Button("Minimax 2", 195, 590, 150, 80, color=(165, 103, 33), hover_color=(175, 113, 43), selecting_color=(145, 93, 23)),
                "m3": Button("Minimax 3", 375, 590, 150, 80, color=(125, 73, 13), hover_color=(135, 83, 23), selecting_color=(105, 63, 3)),
                "lstm": Button("LSTM-256", 555, 590, 150, 80, color=(150, 105, 75), hover_color=(160, 115, 85), selecting_color=(130, 95, 65)),
                "cnn-lstm": Button("LSTM-640", 735, 590, 150, 80, color=(150, 105, 75), hover_color=(160, 115, 85), selecting_color=(130, 95, 65)),
            },
            "model-algo-2": {
                "m1": Button("Minimax 1", 15, 30, 150, 80, color=(205, 133, 63), hover_color=(215, 143, 73), selecting_color=(185, 123, 53)),
                "m2": Button("Minimax 2", 195, 30, 150, 80, color=(165, 103, 33), hover_color=(175, 113, 43), selecting_color=(145, 93, 23)),
                "m3": Button("Minimax 3", 375, 30, 150, 80, color=(125, 73, 13), hover_color=(135, 83, 23), selecting_color=(105, 63, 3)),
                "lstm": Button("LSTM-256", 555, 30, 150, 80, color=(150, 105, 75), hover_color=(160, 115, 85), selecting_color=(130, 95, 65)),
                "cnn-lstm": Button("LSTM-640", 735, 30, 150, 80, color=(150, 105, 75), hover_color=(160, 115, 85), selecting_color=(130, 95, 65)),
            },
        }
        self.text = {
            "white": [self.font.render("Checkmate! White wins!", True, (255, 255, 255)),
                      self.font.render("Checkmate! White wins!", True, (255, 255, 255)).get_rect(center=self.above_rect.center)],
            "black": [self.font.render("Checkmate! Black wins!", True, (255, 255, 255)),
                      self.font.render("Checkmate! Black wins!", True, (255, 255, 255)).get_rect(center=self.above_rect.center)],
            "stale": [self.font.render("Stalemate! It's a draw!", True, (255, 255, 255)),
                      self.font.render("Stalemate! It's a draw!", True, (255, 255, 255)).get_rect(center=self.above_rect.center)],
            "none": [self.font.render("It's a draw!", True, (255, 255, 255)),
                     self.font.render("It's a draw!", True, (255, 255, 255)).get_rect(center=self.above_rect.center)],
        }
        self.letters = []
        for i in range (1, self.HEIGHT + 1):
            rect = pygame.Rect(self.padding, self.padding + (8 - i) * self.SQUARE_SIZE, 15, 20)
            self.letters.append([self.letter_font.render(str(i), True, (255, 206, 158) if i % 2 == 1 else (209, 139, 71)),
                               self.letter_font.render(str(i), True, (255, 206, 158) if i % 2 == 1 else (209, 139, 71)).get_rect(center = rect.center)])

        for i, l in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
            rect = pygame.Rect(self.padding + (i + 1) * self.SQUARE_SIZE - 15, self.SCREEN_HEIGHT - self.padding - 20, 15, 20)
            self.letters.append([self.letter_font.render(l, True, (255, 206, 158) if i % 2 == 0 else (209, 139, 71)),
                               self.letter_font.render(l, True, (255, 206, 158) if i % 2 == 0 else (209, 139, 71)).get_rect(center = rect.center)])
        self.ai_black = "m3"
        self.ai_white = "m3"
        self.game_started = False
        self.running = True
        self.selected_square = None
        self.promoting = False
        self.promoting_draw_pos = [-1, -1]
        self.promoting_square = {}
        self.highlight_last_move = []
        self.RNNModel = model.ChessRNN()
        self.RNNModel.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))
        self.LSTMModel256 = model.ChessLSTM256()
        self.LSTMModel256.load_state_dict(torch.load('model/lstm_256_final.pth', map_location=torch.device('cpu')))
        self.LSTMModel640 = model.ChessLSTM640()
        self.LSTMModel640.load_state_dict(torch.load('model/lstm_640_final.pth', map_location=torch.device('cpu')))
        self.CNN_LSTMModel = model.ChessCNNLSTM()
        self.CNN_LSTMModel.load_state_dict(torch.load('model/chess_cnn_lstm_3m_2.pth', map_location=torch.device('cpu')))
#        self.finetunedModel = model.DQNChessLSTM(self.LSTMModel)
#        self.finetunedModel.load_state_dict(torch.load('model/finetuned_chess_lstm_dqn_100.pth'))
        self.frame = pygame.image.load('img/frame.png')
        self.bg = pygame.image.load('img/background.png')
        self.bg = pygame.transform.scale(self.bg, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    
    def handle_click(self, pos):
        if not self.game_started:
            if self.button['play'].rect.collidepoint(pos):
                self.game_started = True
            elif self.button['mode_list']['0P'].rect.collidepoint(pos):
                self.mode = "0P"
            elif self.button['mode_list']['1P'].rect.collidepoint(pos):
                self.mode = "1P"
            elif self.button['mode_list']['2P'].rect.collidepoint(pos):
                self.mode = "2P"
            
            if (self.mode == "0P"):
                for model in self.button['model-algo-1']:
                    if self.button['model-algo-1'][model].rect.collidepoint(pos):
                        self.ai_white = model
                for model in self.button['model-algo-2']:
                    if self.button['model-algo-2'][model].rect.collidepoint(pos):
                        self.ai_black = model
            
            elif (self.mode == "1P"):
                for model in self.button['model-algo-2']:
                    if self.button['model-algo-2'][model].rect.collidepoint(pos):
                        self.ai_black = model
        else:
            if self.promoting:
                for promote, (x, y) in self.promoting_square.items():
                    if x <= pos[0] <= x + self.SQUARE_SIZE // 2 and y <= pos[1] <= y + self.SQUARE_SIZE // 2:
                        move = chess.Move(self.selected_square, chess.square(self.promoting_draw_pos[0], self.promoting_draw_pos[1]), promotion=promote)
                        if move in self.board.legal_moves:
                            self.highlight_last_move.clear()
                            self.highlight_last_move.append(chess.square(self.promoting_draw_pos[0], self.promoting_draw_pos[1]))
                            self.highlight_last_move.append(self.selected_square)
                            self.board.push(move)
                            self.promoting_draw_pos = [-1, -1]
                            self.promoting = False
                            self.selected_square = None
                        return
            if self.button['undo'].rect.collidepoint(pos):
                if len(self.board.move_stack) > 0:
                    self.board.pop()
                    if (self.mode == "1P"): self.board.pop()
                    self.selected_square = None
                    self.promoting_draw_pos = [-1, -1]
                    self.highlight_last_move.clear()
                    self.promoting = False
                return        
            
            if self.button['retry'].rect.collidepoint(pos):
                self.selected_square = None
                self.promoting_draw_pos = [-1, -1]
                self.promoting = False
                self.highlight_last_move.clear()
                self.board = chess.Board()
                return
            
            if self.button['menu'].rect.collidepoint(pos):
                self.selected_square = None
                self.promoting_draw_pos = [-1, -1]
                self.promoting = False
                self.highlight_last_move.clear()
                self.board = chess.Board()
                self.game_started = False
                return

            col = (pos[0] - self.padding) // self.SQUARE_SIZE
            row = 7 - (pos[1] - self.padding) // self.SQUARE_SIZE
            if col < 0 or col >= self.WIDTH or row < 0 or row >= self.HEIGHT:
                return
            square = chess.square(col, row)
            #print(square)
            if self.selected_square is not None:
                available_moves = []
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    available_moves.append(move)
                for promoting in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    move = chess.Move(self.selected_square, square, promotion=promoting)
                    #print(move.uci())
                    if move in self.board.legal_moves:
                        available_moves.append(move)

                #print(available_moves)
                if len(available_moves) == 0:
                    self.selected_square = None
                    self.promoting_draw_pos = [-1, -1]
                    self.promoting = False
                elif len(available_moves) == 1:
                    print(self.board.san(available_moves[0]))
                    self.board.push(available_moves[0])
                    self.promoting_draw_pos = [-1, -1]
                    self.highlight_last_move.clear()
                    self.highlight_last_move.append(square)
                    self.highlight_last_move.append(self.selected_square)
                    self.promoting = False
                    self.selected_square = None
                else:
                    self.promoting = True
                    if (row == 0):
                        self.promoting_draw_pos = [col, row]
                    elif (row == 7):
                        self.promoting_draw_pos = [col, row]
            else:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square

    def available_moves_from_square(self, square):
        moves = []
        for move in self.board.legal_moves:
            if move.from_square == square:
                moves.append(move.to_square)
        return moves

    def run(self):
        while self.running:
            if (self.mode == "2P"):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self.handle_click(event.pos)
            elif (self.mode == "1P"):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self.handle_click(event.pos)
                            if self.board.is_checkmate():
                                print("Checkmate!")
                                #self.running = False
                            elif self.board.is_stalemate():
                                print("Stalemate!")
                                #self.running = False
                            elif self.board.is_seventyfive_moves():
                                print("Draw by 75-move rule!")
                                #self.running = False
                            elif self.board.is_fivefold_repetition():
                                print("Draw by fivefold repetition!")
                                #self.running = False
                            if not self.promoting and not self.board.is_game_over() and self.board.turn == chess.BLACK:
                                gui.draw(self)
                                pygame.display.flip()
                                pygame.event.pump()
                                pygame.time.wait(300)
                                move = AI.AI_move_black(self)
                                if move:
                                    self.highlight_last_move.clear()
                                    self.highlight_last_move.append(move.to_square)
                                    self.highlight_last_move.append(move.from_square)
                                    self.board.push(move)
                            print(self.highlight_last_move)

            elif (self.mode == "0P"):
                #let 2 bots play
                if self.game_started == False:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:
                                self.handle_click(event.pos)
                if self.game_started == True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:
                                self.handle_click(event.pos)
                    if self.board.turn == chess.WHITE:
                        move = AI.AI_move_white(self)
                    else:
                        move = AI.AI_move_black(self)
                    if move and self.board.is_game_over() == False:
                        self.highlight_last_move.clear()
                        self.highlight_last_move.append(move.to_square)
                        self.highlight_last_move.append(move.from_square)
                        self.board.push(move)
                        
                        gui.draw(self)
                        pygame.display.flip()
                        pygame.event.pump()
                        pygame.time.wait(300)
                    else:
                        print("No valid moves available!")
                        #pygame.time.wait(3000)
                        #self.running = False
            gui.draw(self)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

