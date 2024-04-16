import dash, random, os, threading
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input as KerasInput

# Initialize the Dash app
app = dash.Dash(__name__)

class Checker:
    def __init__(self, color, king=False):
        self.color = color
        self.king = king

def create_board():
    board = np.full((8, 8), None, dtype=object)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                if i < 3:
                    board[i][j] = Checker('black')
                elif i > 4:
                    board[i][j] = Checker('red')
    return board

def init_neural_network():
    model = Sequential([
        KerasInput(shape=(64,)),  # Corrected to use KerasInput
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

ai_model = init_neural_network()

def retrain_ai_thread(data):
    thread = threading.Thread(target=train_ai, args=(ai_model, data))
    thread.start()

def save_move_data(board, move, player, feedback, file_path='checkers_data.csv'):
    data = {
        'board': [serialize_board_simple(board)],  # Use simple serialization for AI processing
        'move': [move],
        'player': [player],
        'feedback': [feedback]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def serialize_board_simple(board):
    # Simplified serialization for AI processing
    serialized_board = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            if board[i][j] is not None:
                serialized_board[i][j] = 2 if board[i][j].king else 1 if board[i][j].color == 'black' else 4 if board[i][j].king else 3
    return serialized_board.flatten().tolist()

def serialize_board(board):
    return [[{'color': cell.color, 'king': cell.king} if cell else None for cell in row] for row in board]

def deserialize_board(board_data):
    return np.array([[Checker(**cell) if cell else None for cell in row] for row in board_data], dtype=object)

def generate_board_figure(board, selected=None):
    fig = go.Figure()
    colors = ['white', 'grey']
    selected_color = 'lightgreen'
    for i in range(8):
        for j in range(8):
            cell_color = selected_color if selected == (i, j) else colors[(i + j) % 2]
            fig.add_trace(go.Scatter(x=[j], y=[7-i], marker=dict(color=cell_color, size=50), mode='markers', marker_symbol='square'))
            piece = board[i][j]
            if piece:
                piece_color = 'red' if piece.color == 'red' else 'black'
                symbol = 'star' if piece.king else 'circle'
                fig.add_trace(go.Scatter(x=[j], y=[7-i], marker=dict(color=piece_color, size=20), mode='markers', marker_symbol=symbol))
    fig.update_layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, height=400, width=400, showlegend=False)
    return fig

app.layout = html.Div([
    dcc.Graph(id='checkerboard', figure=generate_board_figure(create_board()), config={'staticPlot': False, 'scrollZoom': False, 'editable': False, 'displayModeBar': False}),
    dcc.Store(id='board-state', data=serialize_board(create_board())),
    dcc.Store(id='selected-piece'),
    dcc.Store(id='current-player', data='red'),
    html.Div(id='turn-indicator', children="Red's turn", style={'textAlign': 'center', 'fontSize': 24}),
    html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'}),
    dcc.Graph(id='model-performance', config={'staticPlot': True}),
    html.Div([
        dcc.Interval(
            id='update-performance-event',
            interval=5000,  # in milliseconds
            n_intervals=0
        )
    ])
])

@app.callback(
    Output('model-performance', 'figure'),
    [Input('update-performance-event', 'n_intervals')]
)
def update_performance_graph(_):
    try:
        df = pd.read_csv('model_performance.csv')
    except FileNotFoundError:
        # Create a dummy DataFrame if the file does not exist
        df = pd.DataFrame({'epoch': [1], 'loss': [0], 'accuracy': [0]})
    except pd.errors.EmptyDataError:
        # Handle the case where the file is empty
        df = pd.DataFrame({'epoch': [1], 'loss': [0], 'accuracy': [0]})

    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], name='Loss', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['accuracy'], name='Accuracy', mode='lines+markers'))

    fig.update_layout(title='Model Training Performance',
                      xaxis_title='Epoch',
                      yaxis_title='Value',
                      legend_title='Metrics')
    return fig

@app.callback(
    [Output('checkerboard', 'figure'), Output('board-state', 'data'), Output('selected-piece', 'data'), Output('turn-indicator', 'children'), Output('feedback-message', 'children')],
    [Input('checkerboard', 'clickData')],
    [State('board-state', 'data'), State('selected-piece', 'data'), State('current-player', 'data')]
)
def update_board(clickData, board_data, selected_piece, current_player):
    if not clickData:
        raise PreventUpdate
    col, row = int(clickData['points'][0]['x']), 7 - int(clickData['points'][0]['y'])
    board = deserialize_board(board_data)
    feedback_message = ""
    move_description = f"{current_player} from {selected_piece} to {row, col}" if selected_piece else "No move"
    
    if selected_piece:
        sel_row, sel_col = selected_piece
        if board[row][col] is None and is_valid_move(sel_row, sel_col, row, col, board, current_player):
            execute_move(sel_row, sel_col, row, col, board)
            if is_promotion_row(row, current_player):
                board[row][col].king = True
            feedback_message = "Valid move."
            selected_piece = None
            save_move_data(board, move_description, current_player, feedback_message)  # Save move data
            current_player = 'black' if current_player == 'red' else 'red'
        else:
            feedback_message = "Invalid move, try again."
            return generate_board_figure(board, selected_piece), serialize_board(board), selected_piece, f"{current_player.capitalize()}'s turn", feedback_message
    else:
        if board[row][col] and board[row][col].color == current_player:
            selected_piece = (row, col)
            feedback_message = "Piece selected."
            return generate_board_figure(board, selected_piece), serialize_board(board), selected_piece, f"{current_player.capitalize()}'s turn", feedback_message
        else:
            feedback_message = "Not your turn or no valid piece selected."
            return generate_board_figure(board), serialize_board(board), None, f"{current_player.capitalize()}'s turn", feedback_message

    if current_player == 'black':  # Assuming 'black' is the AI player
        if not ai_move_nn(board, current_player):
            feedback_message = "AI has no valid moves."
        current_player = 'red'
        selected_piece = None

    if check_for_winner(board):
        feedback_message = f"{current_player.capitalize()} wins!"
        next_turn = "Game Over"
    else:
        next_turn = 'Red\'s turn' if current_player == 'black' else 'Black\'s turn'

    save_move_data(board, move_description, current_player, feedback_message)  # Save move data after AI moves or game ends
    return generate_board_figure(board, selected_piece), serialize_board(board), None, next_turn, feedback_message

def is_valid_move(sel_row, sel_col, row, col, board, current_player):
    if row < 0 or row > 7 or col < 0 or col > 7:
        return False
    if abs(sel_row - row) != abs(sel_col - col):
        return False
    if abs(sel_row - row) > 2 or abs(sel_row - row) < 1:
        return False
    piece = board[sel_row][sel_col]
    if piece is None:
        return False
    if not piece.king:
        if (piece.color == 'red' and row > sel_row) or (piece.color == 'black' and row < sel_row):
            return False
    if abs(sel_row - row) == 1:
        return board[row][col] is None
    if abs(sel_row - row) == 2:
        mid_row = (sel_row + row) // 2
        mid_col = (sel_col + col) // 2
        mid_piece = board[mid_row][mid_col]
        if mid_piece and mid_piece.color != current_player and board[row][col] is None:
            return True
    return False

def execute_move(sel_row, sel_col, row, col, board):
    board[row][col] = board[sel_row][sel_col]
    board[sel_row][sel_col] = None
    if abs(sel_row - row) == 2:
        mid_row = (sel_row + row) // 2
        mid_col = (sel_col + col) // 2
        board[mid_row][mid_col] = None

def is_promotion_row(row, current_player):
    return (current_player == 'red' and row == 0) or (current_player == 'black' and row == 7)

def check_for_winner(board):
    red_exists, black_exists = False, False
    for row in board:
        for piece in row:
            if piece:
                if piece.color == 'red':
                    red_exists = True
                elif piece.color == 'black':
                    black_exists = True
                if red_exists and black_exists:
                    return False
    return True

def check_and_trigger_retraining():
    move_data = pd.read_csv('checkers_data.csv')
    if len(move_data) % 100 == 0:  # Every 100 moves
        data = prepare_data_for_training('checkers_data.csv')
        retrain_ai_thread(data)

def prepare_data_for_training(file_path):
    data = pd.read_csv(file_path)
    # Assuming 'board' needs to be converted from serialized form to a flat array suitable for NN input
    features = np.array([eval(board) for board in data['board']])  # Convert string representation back to list
    labels = data['move'].apply(lambda x: 1 if x == 'some_win_condition' else 0)  # Dummy condition
    return {'features': features, 'labels': labels}

def find_all_valid_moves(board, current_player):
    valid_moves = []
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece and piece.color == current_player:
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for d in directions:
                    row, col = i + d[0], j + d[1]
                    if 0 <= row < 8 and 0 <= col < 8 and is_valid_move(i, j, row, col, board, current_player):
                        valid_moves.append(((i, j), (row, col)))
                    jump_row, jump_col = i + 2*d[0], j + 2*d[1]
                    if 0 <= jump_row < 8 and 0 <= jump_col < 8 and is_valid_move(i, j, jump_row, jump_col, board, current_player):
                        valid_moves.append(((i, j), (jump_row, jump_col)))
    return valid_moves

def ai_move_nn(board, current_player):
    valid_moves = find_all_valid_moves(board, current_player)
    if not valid_moves:
        return False

    # Evaluate all valid moves using the neural network
    move_scores = []
    for move in valid_moves:
        simulated_board = simulate_move(board, move[0], move[1])
        serialized = np.array([serialize_board_simple(simulated_board)])
        score = ai_model.predict(serialized)[0]
        move_scores.append((score, move))

    # Select the move with the highest score predicted by the AI
    selected_move = max(move_scores, key=lambda x: x[0])[1]
    execute_move(selected_move[0][0], selected_move[0][1], selected_move[1][0], selected_move[1][1], board)
    return True

def simulate_move(board, from_pos, to_pos):
    simulated_board = np.copy(board)  # Assuming board is a numpy array for simplicity
    piece = simulated_board[from_pos]
    simulated_board[to_pos] = piece
    simulated_board[from_pos] = None
    return simulated_board

if __name__ == '__main__':
    app.run_server(debug=True)
