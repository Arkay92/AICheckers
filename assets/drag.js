
function initializeDragging() {
    let piece = null;
    let selectedPiece = null;
    let initialX, initialY;
    let offsetX, offsetY;
    let doubleClick = false;

    function startDragging(e) {
        if (e.target.nodeName === 'circle') {
            piece = e.target;
            selectedPiece = piece.parentNode;
            initialX = selectedPiece.getAttribute('transform').split(',')[0].split('(')[1];
            initialY = selectedPiece.getAttribute('transform').split(',')[1].split(')')[0];
            offsetX = e.offsetX;
            offsetY = e.offsetY;
            document.getElementById('checkerboard').style.pointerEvents = 'none'; // Disable pointer events on the chart
            document.getElementById('checkerboard').style.cursor = 'pointer'; // Change cursor to pointer
        }
    }

    function dragPiece(e) {
        if (piece) {
            var x = e.clientX - offsetX;
            var y = e.clientY - offsetY;
            selectedPiece.setAttribute('transform', 'translate(' + x + ',' + y + ')');
            e.preventDefault(); // Prevent default behavior (e.g., zooming)
        }
    }

    document.addEventListener('newPiecePosition', function(e) {
        document.getElementById('piece-position').textContent = JSON.stringify(e.detail);
    });

    function endDragging(e) {
        if (piece) {
            var col = Math.round((e.clientX - offsetX) / 75);  // Adjust these values based on your board size
            var row = Math.round((e.clientY - offsetY) / 75);  // Adjust these values based on your board size
            var new_x = col * 75;
            var new_y = row * 75;
            selectedPiece.setAttribute('transform', 'translate(' + new_x + ',' + new_y + ')');
            piece = null;
            selectedPiece = null;
            document.getElementById('checkerboard').style.pointerEvents = 'auto'; // Re-enable pointer events on the chart
            document.getElementById('checkerboard').style.cursor = 'default'; // Change cursor back to default
    
            // Send the new position to Dash
            let newPosition = {col: col, row: row};
            window.dispatchEvent(new CustomEvent('newPiecePosition', {detail: newPosition}));
        }
    }    

    function handleDoubleClick(e) {
        if (!doubleClick) {
            doubleClick = true;
            setTimeout(function() {
                doubleClick = false;
            }, 300);
        } else {
            // Reset zoom level
            Plotly.relayout('checkerboard', {});
        }
    }

    document.getElementById('checkerboard').addEventListener('mousedown', startDragging);
    document.getElementById('checkerboard').addEventListener('mousemove', dragPiece);
    document.getElementById('checkerboard').addEventListener('mouseup', endDragging);
    document.getElementById('checkerboard').addEventListener('dblclick', handleDoubleClick);

    console.log('hellos')
}

window.onload = initializeDragging;
