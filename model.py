import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

direction_map = {
    (0, 1): 0,   # Up
    (0, -1): 1,  # Down
    (1, 0): 2,   # Right
    (-1, 0): 3,  # Left
    (1, 1): 4,   # Up-right
    (-1, 1): 5,  # Up-left
    (1, -1): 6,  # Down-right
    (-1, -1): 7  # Down-left
}

def board_to_tensor(board: chess.Board):
    piece_map = board.piece_map()
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)

    piece_to_index = {
        (chess.PAWN,   True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK,   True): 3,
        (chess.QUEEN,  True): 4,
        (chess.KING,   True): 5,
        (chess.PAWN,   False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK,   False): 9,
        (chess.QUEEN,  False): 10,
        (chess.KING,   False): 11
    }

    for square, piece in piece_map.items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        idx = piece_to_index[(piece.piece_type, piece.color)]
        tensor[idx][row][col] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        row, col = 7 - chess.square_rank(chess.G1), chess.square_file(chess.G1)
        tensor[12][row][col] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        row, col = 7 - chess.square_rank(chess.C1), chess.square_file(chess.C1)
        tensor[12][row][col] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        row, col = 7 - chess.square_rank(chess.G8), chess.square_file(chess.G8)
        tensor[12][row][col] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        row, col = 7 - chess.square_rank(chess.C8), chess.square_file(chess.C8)
        tensor[12][row][col] = 1.0

    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[13][row][col] = 1.0

    return tensor

def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square

    from_x = chess.square_file(from_square)
    from_y = chess.square_rank(from_square)
    to_x = chess.square_file(to_square)
    to_y = chess.square_rank(to_square)

    dx = to_x - from_x
    dy = to_y - from_y
    knight_directions = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2)
    ]
    if knight_directions.count((dx, dy)) > 0:
        idx = knight_directions.index((dx, dy))
        return from_square * 73 + 56 + idx

    elif move.promotion and move.promotion != chess.QUEEN:
        promo_map = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
        promo_type = promo_map.get(move.promotion, -1)
        return from_square * 73 + 64 + promo_type * 3 + (dx + 1)

    else:
        norm_dx = dx // abs(dx) if dx != 0 else 0
        norm_dy = dy // abs(dy) if dy != 0 else 0
        direction = (norm_dx, norm_dy)
        dir_idx = direction_map.get(direction, -1)
        distance = max(abs(dx), abs(dy)) - 1
        return from_square * 73 + dir_idx * 7 + distance

class ChessRNN(nn.Module):
    def __init__(self, input_size=14*8*8, hidden_size=256, num_layers=2, num_classes=4672):
        super(ChessRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 14, 8, 8)
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)  # Reshape to (batch, seq_len, 14*8*8)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # Lấy output của bước thời gian cuối
        out = self.fc(out)
        return out

class ChessLSTM256(nn.Module):
    def __init__(self, input_size=896, lstm_hidden_size=256, num_layers=2, num_classes=4672):
        super(ChessLSTM256, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size, seq_len, -1)

        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)
        return out

class ChessLSTM640(nn.Module):
    def __init__(self, input_size=896, lstm_hidden_size=640, num_layers=2, num_classes=4672):
        super(ChessLSTM640, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size, seq_len, -1)

        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)
        return out

class ChessCNNLSTM(nn.Module):
    def __init__(self, input_channels=14, hidden_size=384, num_layers=2, num_classes=4672):
        super(ChessCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # CNN để trích xuất đặc trưng từ bàn cờ
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Giảm kích thước: 8x8 → 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Giảm kích thước: 4x4 → 2x2
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256)  # 64 channels x 2x2 → 256
        )
        # LSTM để xử lý chuỗi đặc trưng
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()  # x: [batch, 8, 14, 8, 8]
        # Xử lý từng trạng thái bàn cờ qua CNN
        c_in = x.view(batch_size * seq_len, C, H, W)  # [batch*8, 14, 8, 8]
        c_out = self.cnn(c_in)  # [batch*8, 256]
        r_in = c_out.view(batch_size, seq_len, -1)  # [batch, 8, 256]
        # LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        r_out, _ = self.lstm(r_in, (h0, c0))  # [batch, 8, hidden_size]
        r_out = r_out[:, -1, :]  # Lấy đầu ra cuối: [batch, hidden_size]
        # Fully connected
        out = self.relu(self.fc1(r_out))
        out = self.fc2(out)
        return out

class DQNChessLSTM(nn.Module):
    def __init__(self, pretrained_model):
        super(DQNChessLSTM, self).__init__()
        self.lstm = pretrained_model.lstm
        self.fc1 = pretrained_model.fc1
        self.fc2 = nn.Linear(128, 4672)  # Tầng mới cho Q-value
        self.relu = pretrained_model.relu
        self.dropout = pretrained_model.dropout

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Hàm dự đoán nước đi của bot
def predict_move(model, board_sequence, board, device='cpu'):
    model.eval()
    with torch.no_grad():
        board_sequence = board_sequence.unsqueeze(0).to(device)  # Thêm batch dimension
        output = model(board_sequence)
    #softmax
    output = torch.softmax(output, dim=1)
    # Chuyển chỉ số thành nước đi
    best_move = None
    best_value = float('-inf')
    for move in board.legal_moves:
        move_index = move_to_index(move)
        if output[0][move_index] > best_value:
            best_value = output[0][move_index]
            best_move = move
    print(f"Best move: {best_move}, Value: {best_value}")
    return best_move

def predict_move_topk(model, board_sequence, board, k=3, device='cpu'):
    model.eval()
    with torch.no_grad():
        board_sequence = board_sequence.unsqueeze(0).to(device)
        output = model(board_sequence)
    output = torch.softmax(output, dim=1)

    move_scores = []
    for move in board.legal_moves:
        move_index = move_to_index(move)
        score = output[0][move_index]
        move_scores.append((move, score))

    move_scores.sort(key=lambda x: x[1], reverse=True)
    return move_scores[:k]

def extract_board_sequence(board, seq_len=8):
    temp_board = board.copy(stack=True)
    board_states = []
    
    board_states.append(board_to_tensor(temp_board))
    
    move_stack = list(temp_board.move_stack)
    for _ in range(min(len(move_stack), seq_len - 1)):
        temp_board.pop() 
        board_states.append(board_to_tensor(temp_board))
    
    board_states = board_states[::-1]
    
    while len(board_states) < seq_len:
        board_states.insert(0, torch.zeros((14, 8, 8), dtype=torch.float32))
    
    return torch.stack(board_states[-seq_len:])
        