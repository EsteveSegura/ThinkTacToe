import json
import html
from typing import List, Tuple
import time
from ollama import Client

def get_position_name(row: int, col: int) -> str:
    """
    Returns the natural language name for a position in the board.
    """
    row_names = ["top", "middle", "bottom"]
    col_names = ["left", "center", "right"]
    
    if row == 1 and col == 1:
        return "center"
    elif (row in [0, 2] and col in [0, 2]):
        return f"{row_names[row]} {col_names[col]} corner"
    else:
        return f"{row_names[row]} {col_names[col]}"

def get_board_explanation(board: List[List[str]]) -> str:
    """
    Generates a natural language explanation of the board state.
    """
    explanations = []
    for row in range(3):
        for col in range(3):
            cell = board[row][col]
            position = get_position_name(row, col)
            if cell == "X":
                explanations.append(f"- {position.capitalize()}: Controlled by X")
            elif cell == "O":
                explanations.append(f"- {position.capitalize()}: Controlled by O")
            else:
                explanations.append(f"- {position.capitalize()}: Empty")
    return "\n".join(explanations)

def check_diagonal_threat(board: List[List[str]]) -> bool:
    """
    Checks if O has a potential diagonal threat.
    A diagonal threat exists when:
    1. O controls the center
    2. O controls one corner
    3. The opposite corner is empty
    """
    # Check if O controls the center
    if board[1][1] != "O":
        return False
    
    # Check both diagonals
    # Diagonal 1: top-left to bottom-right
    if board[0][0] == "O" and board[2][2] == "":
        return True
    if board[2][2] == "O" and board[0][0] == "":
        return True
    
    # Diagonal 2: top-right to bottom-left
    if board[0][2] == "O" and board[2][0] == "":
        return True
    if board[2][0] == "O" and board[0][2] == "":
        return True
    
    return False

def generate_think_llm(board: List[List[str]], move: Tuple[int, int], max_retries: int = 3) -> str:
    """
    Generates a thought for a Tic Tac Toe move using Ollama LLM.
    
    Args:
        board: Current board state
        move: Tuple (row, col) representing the move, where:
              row: 0=top, 1=middle, 2=bottom
              col: 0=left, 1=center, 2=right
        max_retries: Maximum number of retry attempts
    
    Returns:
        str: The player's thought about the move
    """
    # Create a copy of the board and apply the move
    board_with_move = [row[:] for row in board]
    row, col = move
    board_with_move[row][col] = "X"
    
    # Convert board states to JSON with proper formatting
    def format_board(board_state):
        # Replace None with empty string and ensure proper JSON formatting
        formatted = []
        for row in board_state:
            formatted_row = []
            for cell in row:
                formatted_row.append(cell if cell is not None else "")
            formatted.append(formatted_row)
        # Use custom formatting to get the desired output format
        json_str = json.dumps(formatted)
        # Replace the default formatting with our desired format
        json_str = json_str.replace("], [", "],\n  [")
        json_str = json_str.replace("[[", "[\n  [")
        json_str = json_str.replace("]]", "]\n]")
        return json_str
    
    # Check for diagonal threats
    has_diagonal_threat = check_diagonal_threat(board)
    extra_notes = "- The opponent has a potential diagonal threat that needs to be blocked." if has_diagonal_threat else "- The opponent has no diagonal threats."
    
    # Prepare the prompt variables
    prompt_vars = {
        "BOARD_INIT_STATE": format_board(board),
        "BOARD_WITH_MOVEMENT_APPLIED": format_board(board_with_move),
        "POSITION": f"({row},{col})",
        "BOARD_EXPLAINED": get_board_explanation(board),
        "EXTRA_NOTES": extra_notes
    }
    
    # Read the prompt template
    with open("src/dataset/utils/prompt.md", "r") as f:
        prompt_template = f.read()
    
    # Replace variables in the template
    for var, value in prompt_vars.items():
        # Handle both underscore and space versions of the variable
        prompt_template = prompt_template.replace(f"{{{{{var}}}}}", value)
        prompt_template = prompt_template.replace(f"{{{{{var.replace('_', ' ')}}}}}", value)
    
    print(prompt_template)
    
    # Initialize Ollama client
    client = Client(host='http://localhost:11434')
    
    # Call Ollama API with retries
    for attempt in range(max_retries):
        try:
            # Process the streaming response
            full_response = ""
            print("\nStreaming response:")
            print("------------------")
            
            # Use the native client for streaming
            for chunk in client.generate(
                model='gemma3:12b',
                prompt=prompt_template,
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.85
                }
            ):
                if 'response' in chunk:
                    print(chunk['response'], end='', flush=True)
                    full_response += chunk['response']
            
            print("\n------------------")
            
            # Get the response and decode HTML entities
            decoded_response = html.unescape(full_response)
            
            # Extract the thought between the tags
            try:
                # Find the last occurrence of the tags
                start_tag = decoded_response.rindex("<player_think>")
                end_tag = decoded_response.rindex("</player_think>")
                
                # Extract the content between the tags
                thought = decoded_response[start_tag + len("<player_think>"):end_tag].strip()
                print(thought)
                print("--------------------------------")
                return f"<player_think> {thought} </player_think>"
            except ValueError:
                raise Exception("Could not find player_think tags in the response")
                
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)  # Wait 1 second before retrying
