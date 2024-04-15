import dash, random, json
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate

# Initialize the Dash app
app = dash.Dash(__name__)


class Checker:
    def __init__(self, color, king=False):
        self.king = king
        self.color = color

def create_board():
    board = np.full((8, 8), None, dtype=object)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:  # This ensures that pieces are placed only on grey squares
                if i < 3:
                    board[i][j] = Checker('black')
                elif i > 4:
                    board[i][j] = Checker('red')
    return board

def serialize_board(board):
    return [[{'color': cell.color, 'king': cell.king} if cell else None for cell in row] for row in board]

def deserialize_board(board_data):
    return np.array([[Checker(**cell) if cell else None for cell in row] for row in board_data], dtype=object)

def generate_board_figure(board, selected=None):
    fig = go.Figure()
    colors = ['white', 'grey']  # Define background colors for the squares
    for i in range(8):
        for j in range(8):
            cell_color = colors[(i + j) % 2]  # This ensures the alternating pattern of the checkerboard
            # Draw the checkerboard square
            fig.add_trace(go.Scatter(
                x=[j],
                y=[7-i],
                marker=dict(color=cell_color, size=50),
                mode='markers',
                marker_symbol='square'
            ))
            piece = board[i][j]
            if piece:
                piece_color = 'red' if piece.color == 'red' else 'black'
                symbol = 'star' if piece.king else 'circle'
                fig.add_trace(go.Scatter(
                    x=[j],
                    y=[7-i],
                    marker=dict(color=piece_color, size=20),
                    mode='markers',
                    marker_symbol=symbol
                ))
    # Update layout to remove the legend
    fig.update_layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, 
        height=400, 
        width=400, 
        showlegend=False  # This disables the legend
    )
    return fig

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='checkerboard',
        figure=generate_board_figure(create_board()),
        config={'staticPlot': False, 'scrollZoom': False, 'editable': False, 'displayModeBar': False}
    ),
    dcc.Store(id='board-state', data=serialize_board(create_board())),
    dcc.Store(id='selected-piece'),
    dcc.Store(id='current-player', data='red'),  # Initialize current player
    html.Div(id='turn-indicator', children='Red\'s turn', style={'textAlign': 'center', 'fontSize': 24}),
    html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'})
])

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
    feedback_message = ""  # Reset feedback message

    if selected_piece:
        # If there's already a selected piece, attempt to move it.
        sel_row, sel_col = selected_piece
        if board[row][col] is None and is_valid_move(sel_row, sel_col, row, col, board, current_player):
            # Execute move if valid.
            execute_move(sel_row, sel_col, row, col, board)
            if is_promotion_row(row, current_player):
                board[row][col].king = True
            feedback_message = "Valid move."
            selected_piece = None  # Clear selected piece after move
            # Switch players only after a successful move.
            current_player = 'black' if current_player == 'red' else 'red'
        else:
            # If the move isn't valid, don't switch players.
            feedback_message = "Invalid move, try again."
            return generate_board_figure(board, selected_piece), serialize_board(board), selected_piece, f"{current_player.capitalize()}'s turn", feedback_message
    else:
        # Selecting a piece.
        if board[row][col] and board[row][col].color == current_player:
            selected_piece = (row, col)  # Update selected piece
            feedback_message = "Piece selected."
            # Do not switch players on piece selection.
            return generate_board_figure(board, selected_piece), serialize_board(board), selected_piece, f"{current_player.capitalize()}'s turn", feedback_message
        else:
            feedback_message = "Not your turn or no valid piece selected."
            return generate_board_figure(board), serialize_board(board), None, f"{current_player.capitalize()}'s turn", feedback_message

    # AI move: Execute this only if the current player is now 'black' and there was a valid move before.
    if current_player == 'black':
        if not ai_move(board, current_player):
            feedback_message = "AI has no valid moves."
        # Switch back to red player after AI's move.
        current_player = 'red'
        selected_piece = None  # Clear selected piece after AI move

    # Check for winner after each move
    if check_for_winner(board):
        feedback_message = f"{current_player.capitalize()} wins!"
        next_turn = "Game Over"
    else:
        next_turn = 'Red\'s turn' if current_player == 'black' else 'Black\'s turn'

    return generate_board_figure(board, selected_piece), serialize_board(board), None, next_turn, feedback_message

def is_valid_move(sel_row, sel_col, row, col, board, current_player):
    # Ensure the move is within board bounds
    if row < 0 or row > 7 or col < 0 or col > 7:
        return False

    # Ensure the move is diagonal
    if abs(sel_row - row) != abs(sel_col - col):
        return False

    # Check if the move is either one or two rows away
    if abs(sel_row - row) > 2 or abs(sel_row - row) < 1:
        return False

    # Get the piece at the selected square
    piece = board[sel_row][sel_col]

    # Ensure there is a piece at the selected square
    if piece is None:
        return False

    # Ensure piece is moving in a correct direction unless it's a king
    if not piece.king:
        if (piece.color == 'red' and row > sel_row) or (piece.color == 'black' and row < sel_row):
            return False

    # Check for simple move
    if abs(sel_row - row) == 1:
        return board[row][col] is None

    # Check for captures
    if abs(sel_row - row) == 2:
        mid_row = (sel_row + row) // 2
        mid_col = (sel_col + col) // 2
        mid_piece = board[mid_row][mid_col]
        if mid_piece and mid_piece.color != current_player and board[row][col] is None:
            return True

    return False

def execute_move(sel_row, sel_col, row, col, board):
    # Move the piece to the new location
    board[row][col] = board[sel_row][sel_col]
    board[sel_row][sel_col] = None
    
    # If the move is a capture, remove the captured piece
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
                # Exit early if both colors are still on the board
                if red_exists and black_exists:
                    return False
    
    # If only pieces of one color are left, that player is the winner
    return True

def find_all_valid_moves(board, current_player):
    valid_moves = []
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece and piece.color == current_player:
                # Possible moves for each piece, including captures
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for d in directions:
                    row, col = i + d[0], j + d[1]
                    if 0 <= row < 8 and 0 <= col < 8 and is_valid_move(i, j, row, col, board, current_player):
                        valid_moves.append(((i, j), (row, col)))
                        # Check for jump moves
                        jump_row, jump_col = i + 2*d[0], j + 2*d[1]
                        if 0 <= jump_row < 8 and 0 <= jump_col < 8 and is_valid_move(i, j, jump_row, jump_col, board, current_player):
                            valid_moves.append(((i, j), (jump_row, jump_col)))
    return valid_moves

def ai_move(board, current_player):
    valid_moves = find_all_valid_moves(board, current_player)
    if valid_moves:
        selected_move = random.choice(valid_moves)
        execute_move(selected_move[0][0], selected_move[0][1], selected_move[1][0], selected_move[1][1], board)
        return True
    return False

if __name__ == '__main__':
    app.run_server(debug=True)
