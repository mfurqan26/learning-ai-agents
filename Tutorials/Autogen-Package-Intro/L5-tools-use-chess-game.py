from utils import get_openai_api_key
from autogen import ConversableAgent, initiate_chats, AssistantAgent, register_function
import pprint
import chess
import chess.svg
from typing_extensions import Annotated
#from IPython.display import display

# get the openai api key
OPENAI_API_KEY = get_openai_api_key()
llm_config = {"model": "gpt-4.1-nano"}

# Initialize the chess board
board = chess.Board()
made_move = False

# Define tools to get moves and make moves for chess
def get_legal_moves(
    
) -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Possible moves are: " + ",".join(
        [str(move) for move in board.legal_moves]
    )
    
def make_move(
    move: Annotated[str, "A move in UCI format."]
) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)
    board.push_uci(str(move))
    global made_move
    made_move = True
    
    # Display the board.
    #display(
    #    chess.svg.board(
    #        board,
    #        arrows=[(move.from_square, move.to_square)],
    #        fill={move.from_square: "gray"},
    #        size=200
    #    )
    #)
    
    # Get the piece name.
    piece = board.piece_at(move.to_square)
    piece_symbol = piece.unicode_symbol()
    piece_name = (
        chess.piece_name(piece.piece_type).capitalize()
        if piece_symbol.isupper()
        else chess.piece_name(piece.piece_type)
    )
    return f"Moved {piece_name} ({piece_symbol}) from "\
    f"{chess.SQUARE_NAMES[move.from_square]} to "\
    f"{chess.SQUARE_NAMES[move.to_square]}."
    
# Defien the two chess agent players
player_white = ConversableAgent(
    name="player_white",
    system_message="You are a chess player and you play as white. "
    "First call get_legal_moves(), to get a list of legal moves. "
    "Then call make_move(move) to make a move.",
    llm_config=llm_config,
)
player_black = ConversableAgent(
    name="player_black",
    system_message="You are a chess player and you play as black. "
    "First call get_legal_moves(), to get a list of legal moves. "
    "Then call make_move(move) to make a move.",
    llm_config=llm_config,
)

# Define chess proxy agent and check if moves made func
def check_made_move(msg):
    global made_move
    if made_move:
        made_move = False
        return True
    else:
        return False

# Define the board proxy agent
board_proxy = ConversableAgent(
    name="board_proxy",
    llm_config=False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)

# Register the tools for the board agent players
for caller in [player_white, player_black]:
    register_function(
        get_legal_moves,
        caller=caller,
        executor=board_proxy,
        name="get_legal_moves",
        description="Get legal moves.",
    )
    
    register_function(
        make_move,
        caller=caller,
        executor=board_proxy,
        name="make_move",
        description="Call this tool to make a move.",
    )


# Check the tools if you want that were register for the agent player
player_black.llm_config["tools"]

# Register nested chats for agent players
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_white,
            "summary_method": "last_msg",
            # if you want to hide the inner chat reasoning and just see output, set silent to True
            "silent": True
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_black,
            "summary_method": "last_msg",
            # if you want to hide the inner chat reasoning and just see output, set silent to True
            "silent": True
        }
    ],
)

# start the chess game
board = chess.Board()
chat_result = player_black.initiate_chat(
    player_white,
    message="Let's play chess! Your move.",
    max_turns=2,
)