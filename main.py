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
from threading import Lock
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, F1Score
from tensorflow import lite
from dash import callback_context
import signal
import sys

# Global list to track threads
active_threads = []

# Lock for file access to ensure thread safety
file_lock = Lock()

# Initialize the Dash app
app = dash.Dash(__name__)

class Checker:
    def __init__(self, color, king=False):
        self.color = color
        self.king = king

def create_board():
    board = np.full((8, 8), None, dtype=object)
    for i in range(8):
        if (i % 2 == 0):
            for j in range(1, 8, 2):
                if i < 3:
                    board[i][j] = Checker('black')
                elif i > 4:
                    board[i][j] = Checker('red')
        else:
            for j in range(0, 8, 2):
                if i < 3:
                    board[i][j] = Checker('black')
                elif i > 4:
                    board[i][j] = Checker('red')
    return board

def init_neural_network():
    model = Sequential([
        KerasInput(shape=(64,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[BinaryAccuracy(), Precision(), Recall(), F1Score()])
    return model

ai_model = init_neural_network()

def quantize_model(model):
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('ai_checkers_model.tflite', 'wb') as f:
        f.write(tflite_model)

def train_and_save_model(data, model):
    """Function to train and save the model."""
    try:
        model.fit(data['features'], data['labels'], epochs=10)
        quantize_model(model)
    finally:
        with thread_lock:  # Assuming thread_lock is a threading.Lock() object
            active_threads.remove(threading.current_thread())

def retrain_ai_thread(data, model):
    """Spawn a thread to train AI model."""
    thread = threading.Thread(target=train_and_save_model, args=(data, model))
    thread.start()
    with thread_lock:  # Track this thread
        active_threads.append(thread)

def graceful_exit(signum, frame):
    """Handle graceful shutdown on signal."""
    print("Shutting down gracefully...")
    for thread in active_threads:
        thread.join()  # Ensure all threads have completed
    sys.exit(0)

def save_move_data(board, move, player, feedback, file_path='checkers_data.csv'):
    data = {'board': [serialize_board_simple(board)], 'move': [move], 'player': [player], 'feedback': [feedback]}
    df = pd.DataFrame(data)
    with file_lock:
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def serialize_board_simple(board):
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

def generate_board_figure(board):
    fig = go.Figure()
    colors = ['white', 'grey']
    for i in range(8):
        for j in range(8):
            cell_color = colors[(i + j) % 2]
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
    html.Button("Start Training", id="training-button", n_clicks=0),
    dcc.Store(id='game-mode', data={'mode': 'player'}),
    html.Div(id='turn-indicator', children="Red's turn", style={'textAlign': 'center', 'fontSize': 24}),
    html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'}),
    dcc.Graph(id='model-performance', config={'staticPlot': True}),
    dcc.Interval(id='game-update', interval=1000, n_intervals=0, disabled=True),
    html.Div([
        dcc.Interval(
            id='update-performance-event',
            interval=5000,  # in milliseconds
            n_intervals=0
        )
    ])
])

def check_game_end_conditions(board, current_player):
    if not find_all_valid_moves(board, current_player) and not find_all_valid_moves(board, 'red' if current_player == 'black' else 'black'):
        return True  # No moves left for either player
    return check_for_winner(board)

@app.callback(
    [Output('checkerboard', 'figure', allow_duplicate=True),
     Output('training-button', 'children', allow_duplicate=True),
     Output('board-state', 'data', allow_duplicate=True),
     Output('turn-indicator', 'children', allow_duplicate=True)],
    [Input('training-button', 'n_clicks')],
    [State('board-state', 'data'),
     State('game-mode', 'data')],
    prevent_initial_call=True  # Preventing initial call
)
def toggle_training_mode(n_clicks, board_state, game_mode):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    if game_mode['mode'] == 'player':
        new_mode = 'training'
        button_label = "Stop Training"
        new_board_state = ai_vs_ai_play(deserialize_board(board_state))
        if check_for_winner(new_board_state) or not find_all_valid_moves(new_board_state, 'black') and not find_all_valid_moves(new_board_state, 'red'):
            new_board_state = create_board()  # Reset if game ends
        new_board_state_serialized = serialize_board(new_board_state)
    else:
        new_mode = 'player'
        button_label = "Start Training"
        new_board_state = create_board()  # Always reset the board when stopping training
        new_board_state_serialized = serialize_board(new_board_state)

    return generate_board_figure(new_board_state), button_label, new_board_state_serialized, f"{new_mode.capitalize()}'s turn"

@app.callback(
    Output('model-performance', 'figure'),
    [Input('update-performance-event', 'n_intervals')]
)
def update_performance_graph(_):
    try:
        df = pd.read_csv('model_performance.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame({'epoch': [1], 'loss': [0], 'accuracy': [0]})
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], name='Loss', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['accuracy'], name='Accuracy', mode='lines+markers'))
    fig.update_layout(title='Model Training Performance', xaxis_title='Epoch', yaxis_title='Value', legend_title='Metrics')
    return fig

@app.callback(
    [Output('checkerboard', 'figure'),
     Output('board-state', 'data'),
     Output('selected-piece', 'data'),
     Output('turn-indicator', 'children'),
     Output('feedback-message', 'children')],
    [Input('checkerboard', 'clickData'),
     Input('training-button', 'n_clicks')],
    [State('board-state', 'data'),
     State('selected-piece', 'data'),
     State('current-player', 'data'),
     State('game-mode', 'data')]
)
def update_board(clickData, n_clicks, board_data, selected_piece, current_player, game_mode):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    board = deserialize_board(board_data)

    if triggered_id == 'training-button':
        if game_mode['mode'] == 'player':
            game_mode['mode'] = 'training'
            board = ai_vs_ai_play(board)  # Simulate AI vs AI until game end or reset condition
            feedback_message = "AI training in progress."
            next_turn = "AI's move"
        else:
            game_mode['mode'] = 'player'
            board = create_board()  # Reset board to initial state
            feedback_message = "Player vs. AI mode."
            next_turn = "Player's move"
        return generate_board_figure(board), serialize_board(board), None, f"{game_mode['mode'].capitalize()}'s turn", feedback_message

    if not clickData:
        raise PreventUpdate

    col, row = int(clickData['points'][0]['x']), 7 - int(clickData['points'][0]['y'])
    feedback_message = ""
    
    if selected_piece:
        sel_row, sel_col = selected_piece
        if board[row][col] is None and is_valid_move(sel_row, sel_col, row, col, board, current_player):
            execute_move(sel_row, sel_col, row, col, board)
            if is_promotion_row(row, current_player):
                board[row][col].king = True
            feedback_message = "Valid move."
            selected_piece = None
            save_move_data(board, f"{current_player} from {selected_piece} to {row, col}", current_player, feedback_message)
            current_player = 'black' if current_player == 'red' else 'red'
            next_turn = f"{current_player.capitalize()}'s turn"
        else:
            feedback_message = "Invalid move, try again."
            return generate_board_figure(board), serialize_board(board), selected_piece, f"{current_player.capitalize()}'s turn", feedback_message
    else:
        if board[row][col] and board[row][col].color == current_player:
            selected_piece = (row, col)
            feedback_message = "Piece selected."
            next_turn = f"{current_player.capitalize()}'s turn"
            return generate_board_figure(board), serialize_board(board), selected_piece, next_turn, feedback_message
        else:
            feedback_message = "Not your turn or no valid piece selected."
            return generate_board_figure(board), serialize_board(board), None, f"{current_player.capitalize()}'s turn", feedback_message

    if not find_all_valid_moves(board, current_player):
        feedback_message = "No valid moves available."
        board = create_board()  # Reset the board if no moves are available
        next_turn = "Game reset for new play"
    elif check_for_winner(board):
        feedback_message = f"{current_player.capitalize()} wins!"
        board = create_board()  # Reset the board after a win
        next_turn = "Game reset for new play"

    return generate_board_figure(board), serialize_board(board), None, next_turn, feedback_message

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

def check_and_trigger_retraining(model):
    with file_lock:
        move_data = pd.read_csv('checkers_data.csv')
    if len(move_data) % 100 == 0:  # Every 100 moves
        # Prepare and train
        retrain_ai_thread(prepare_data_for_training('checkers_data.csv'), model)

def prepare_data_for_training(file_path):
    with file_lock:
        data = pd.read_csv(file_path)
    features = np.array([eval(board) for board in data['board']])
    labels = data['move'].apply(lambda x: 1 if x == 'some_win_condition' else 0)
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

def ai_vs_ai_play(board):
    # Initialize variables to control the game flow
    current_player = 'black'
    move_possible = True
    move_count = 0
    max_moves = 100  # Define a maximum move limit to prevent infinite loops in training

    # Loop to let AI play against itself until no moves are possible or max moves are reached
    while move_possible and move_count < max_moves:
        move_possible = ai_move_nn(board, current_player)
        # Switch players after each move
        current_player = 'red' if current_player == 'black' else 'black'
        move_count += 1

    return board  # Return the updated board state after AI moves

def ai_move_nn(board, current_player):
    valid_moves = find_all_valid_moves(board, current_player)
    if not valid_moves:
        return False  # Return False if no valid moves are found

    # Evaluate all valid moves using the neural network
    move_scores = []
    for move in valid_moves:
        simulated_board = simulate_move(board, move[0], move[1])
        serialized = np.array([serialize_board_simple(simulated_board)])
        try:
            score = ai_model.predict(serialized)[0]
            move_scores.append((score, move))
        except Exception as e:
            print(f"Error during model prediction: {e}")
            continue  # Skip this move if there's an error in prediction

    if not move_scores:
        return False  # If all predictions failed, return False

    # Select the move with the highest score predicted by the AI
    selected_move = max(move_scores, key=lambda x: x[0])[1]
    execute_move(selected_move[0][0], selected_move[0][1], selected_move[1][0], selected_move[1][1], board)
    return True  # Return True to indicate that a move was made

def simulate_move(board, from_pos, to_pos):
    simulated_board = np.copy(board)  # Assuming board is a numpy array for simplicity
    piece = simulated_board[from_pos]
    simulated_board[to_pos] = piece
    simulated_board[from_pos] = None
    return simulated_board

# Set up signal handling for graceful shutdown
signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

if __name__ == '__main__':
    app.run_server(debug=True)