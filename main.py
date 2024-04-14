import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# CSS style for changing cursor
styles = """
<style>
.js-plotly-plot .plotly .main-svg .draglayer {
    cursor: pointer !important;
}
</style>
"""

class Checker:
    def __init__(self, color, king=False):
        self.king = king
        self.color = color

def create_board():
    board = np.full((8, 8), None, dtype=object)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                if i < 3:
                    board[i][j] = Checker("black")
                elif i > 4:
                    board[i][j] = Checker("red")
    return board

def serialize_board(board):
    return [[{'color': cell.color, 'king': cell.king} if cell else None for cell in row] for row in board]

def deserialize_board(board_data):
    return np.array([[Checker(**cell) if cell else None for cell in row] for row in board_data], dtype=object)

def generate_board_figure(board, selected=None):
    fig = go.Figure()
    for i in range(8):
        for j in range(8):
            color = 'white' if (i + j) % 2 == 0 else 'grey'
            hover_cursor = 'pointer' if board[i][j] else 'default'  # Set cursor to pointer if there's a piece
            hover_template = f"<b>Row:</b> {i}<br><b>Column:</b> {j}<extra></extra>"
            fig.add_trace(go.Scatter(
                x=[j],
                y=[7-i],
                marker=dict(color=color, size=40),
                mode='markers',
                marker_symbol='square',
                name=f"square_{i}_{j}",
                customdata=[[hover_cursor]],  # Pass cursor style as custom data
                hoverinfo='text',  # Show custom hover info
                hovertemplate=hover_template
            ))
            if board[i][j]:
                piece_color = 'red' if board[i][j].color == 'red' else 'black'
                opacity = 1.0 if selected == (i, j) else 0.5  # Highlight the selected piece
                fig.add_trace(go.Scatter(
                    x=[j],
                    y=[7-i],
                    marker=dict(color=piece_color, size=25, opacity=opacity),
                    mode='markers',
                    name=f"piece_{i}_{j}"
                ))
    fig.update_xaxes(range=[-0.5, 7.5], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.5, 7.5], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(height=600, width=600, margin=dict(l=20, r=20, t=20, b=20), showlegend=False, uirevision=True)
    fig.update_layout(dragmode=False)
    return fig

# Define the layout of the app
app.layout = html.Div([
    html.Div(
        styles,
        style={'display': 'none'}
    ),
    dcc.Graph(
        id='checkerboard',
        figure=generate_board_figure(create_board()),
        config={'staticPlot': False, 'scrollZoom': False, 'editable': False, 'displayModeBar': False},
        style={'height': '600px', 'width': '600px'}
    ),
    dcc.Store(id='board-state', data=serialize_board(create_board())),
    dcc.Store(id='selected-piece'),
    html.Script(src='assets/drag.js')
])

# Define the callback to handle interactions
@app.callback(
    [Output('checkerboard', 'figure'), Output('board-state', 'data'), Output('selected-piece', 'data')],
    [Input('checkerboard', 'clickData')],
    [State('board-state', 'data'), State('selected-piece', 'data')]
)
def update_board(clickData, board_data, selected_piece):
    board = deserialize_board(board_data)
    if clickData:
        col, row = int(clickData['points'][0]['x']), 7 - int(clickData['points'][0]['y'])
        if selected_piece:
            sel_row, sel_col = selected_piece
            if board[row][col] is None:
                board[row][col], board[sel_row][sel_col] = board[sel_row][sel_col], None
                selected_piece = None
            else:
                selected_piece = (row, col) if board[row][col] else selected_piece
        else:
            if board[row][col]:
                selected_piece = (row, col)
    return generate_board_figure(board, selected_piece), serialize_board(board), selected_piece

if __name__ == '__main__':
    app.run_server(debug=True)