import os
import uuid
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chess_ai import ChessBoard, AutonomousChessAgent

app = FastAPI(
    title="NeuroChess Autonomous AI Engine API",
    description="Intelligent Chess Web Application with PyTorch Deep Neural Networks (Stockfish NNUE SFNNv9 + Lc0 SE-ResNet Transformer + Dueling DQN + Neuro-Evolution), PVS/MCTS, and Kubernetes readiness.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Autonomous Neural Agent & Active Game Sessions
agent = AutonomousChessAgent()
sessions: Dict[str, ChessBoard] = {}

DEFAULT_SESSION_ID = "default"
sessions[DEFAULT_SESSION_ID] = ChessBoard()


class NewGameRequest(BaseModel):
    session_id: Optional[str] = DEFAULT_SESSION_ID
    fen: Optional[str] = None
    player_color: str = "w"
    learning_model: str = "alphazero_mcts"
    reaction_profile: str = "tactical"
    difficulty: int = 3


class PlayerMoveRequest(BaseModel):
    session_id: str = DEFAULT_SESSION_ID
    uci: str
    auto_reply: bool = True
    learning_model: str = "alphazero_mcts"
    reaction_profile: str = "tactical"
    difficulty: int = 3


class AgentMoveRequest(BaseModel):
    session_id: str = DEFAULT_SESSION_ID
    learning_model: str = "alphazero_mcts"
    reaction_profile: str = "tactical"
    difficulty: int = 3


class TrainAgentRequest(BaseModel):
    learning_model: str = "alphazero_mcts"
    iterations: int = 25


def serialize_board_state(
    board: ChessBoard,
    telemetry: Optional[Dict[str, Any]] = None,
    learning_model: str = "alphazero_mcts",
    reaction_profile: str = "tactical"
) -> Dict[str, Any]:
    legal = board.generate_legal_moves()
    legal_dicts = [m.to_dict(board.format_san(m, legal)) for m in legal]
    status = board.get_game_status()

    # If no telemetry provided, compute quick baseline evaluation & neural activations
    if telemetry is None:
        brain = agent.networks.get(learning_model, agent.networks["alphazero_mcts"])
        _, _, nn_out = brain.evaluate_board(board)
        telemetry = agent._build_neural_telemetry(
            board=board,
            brain=brain,
            nn_out=nn_out,
            chosen_move=legal[0] if legal else None,
            chosen_san=board.format_san(legal[0], legal) if legal else "-",
            candidate_rankings=[],
            learning_model=learning_model,
            reaction_profile=reaction_profile,
            td_loss=brain.loss_history[-1] if brain.loss_history else 0.025,
            elapsed_ms=2,
            reasoning_steps=[f"Red PyTorch ({agent.LEARNING_MODELS[learning_model]['name']}) sincronizada y lista."]
        ) if legal else {}

    return {
        "fen": board.to_fen(),
        "turn": board.turn,
        "board": board.board,
        "in_check": board.in_check(board.turn),
        "king_sq": board.find_king(board.turn),
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "legal_moves": legal_dicts,
        "move_history": board.move_history,
        "status": status,
        "telemetry": telemetry
    }


@app.get("/api/health")
def health_check():
    """Kubernetes liveness & readiness probe endpoint."""
    return {
        "status": "healthy",
        "service": "neuro-chess-engine",
        "models_loaded": list(agent.networks.keys()),
        "active_sessions": len(sessions)
    }


@app.get("/api/models")
def get_models_and_profiles():
    """Returns all available Neural Learning Models and Reaction Profiles."""
    model_stats = {}
    for mid, brain in agent.networks.items():
        model_stats[mid] = {
            **agent.LEARNING_MODELS[mid],
            "param_count": brain.param_count,
            "training_epochs": brain.training_epochs,
            "games_learned": brain.games_learned,
            "latest_loss": brain.loss_history[-1] if brain.loss_history else 0.0
        }
    return {
        "learning_models": model_stats,
        "reaction_profiles": agent.REACTION_PROFILES
    }


@app.post("/api/game/new")
def start_new_game(req: NewGameRequest):
    sid = req.session_id or str(uuid.uuid4())
    board = ChessBoard(req.fen) if req.fen else ChessBoard()
    sessions[sid] = board

    agent_move_info = None
    telemetry = None

    # If player chose Black ('b') and it is White's turn, autonomous agent makes the first move
    if req.player_color == "b" and board.turn == "w":
        res = agent.select_move(
            board,
            learning_model=req.learning_model,
            reaction_profile=req.reaction_profile,
            difficulty=req.difficulty
        )
        if res["move"]:
            agent_move_info = board.push_uci(res["move"].uci())
            telemetry = res["telemetry"]

    state = serialize_board_state(
        board,
        telemetry,
        learning_model=req.learning_model,
        reaction_profile=req.reaction_profile
    )
    state["session_id"] = sid
    state["agent_move"] = agent_move_info
    return state


@app.get("/api/game/{session_id}")
def get_game_state(session_id: str):
    board = sessions.get(session_id)
    if not board:
        board = ChessBoard()
        sessions[session_id] = board
    state = serialize_board_state(board)
    state["session_id"] = session_id
    return state


@app.post("/api/game/move")
def make_player_move(req: PlayerMoveRequest):
    board = sessions.get(req.session_id)
    if not board:
        board = ChessBoard()
        sessions[req.session_id] = board

    player_move_info = board.push_uci(req.uci)
    if not player_move_info:
        raise HTTPException(status_code=400, detail=f"Movimiento ilegal: {req.uci}")

    agent_move_info = None
    telemetry = None
    status = board.get_game_status()

    if req.auto_reply and not status["is_game_over"]:
        res = agent.select_move(
            board,
            learning_model=req.learning_model,
            reaction_profile=req.reaction_profile,
            difficulty=req.difficulty
        )
        if res["move"]:
            agent_move_info = board.push_uci(res["move"].uci())
            telemetry = res["telemetry"]

    state = serialize_board_state(
        board,
        telemetry,
        learning_model=req.learning_model,
        reaction_profile=req.reaction_profile
    )
    state["session_id"] = req.session_id
    state["player_move"] = player_move_info
    state["agent_move"] = agent_move_info
    return state


@app.post("/api/game/agent-move")
def trigger_autonomous_agent_move(req: AgentMoveRequest):
    """Allows autonomous Agent vs Agent self-play or on-demand AI move."""
    board = sessions.get(req.session_id)
    if not board:
        board = ChessBoard()
        sessions[req.session_id] = board

    status = board.get_game_status()
    if status["is_game_over"]:
        state = serialize_board_state(board)
        state["session_id"] = req.session_id
        return state

    res = agent.select_move(
        board,
        learning_model=req.learning_model,
        reaction_profile=req.reaction_profile,
        difficulty=req.difficulty
    )
    agent_move_info = None
    if res["move"]:
        agent_move_info = board.push_uci(res["move"].uci())

    state = serialize_board_state(board, res["telemetry"])
    state["session_id"] = req.session_id
    state["agent_move"] = agent_move_info
    return state


@app.post("/api/agent/train")
def train_neural_agent(req: TrainAgentRequest):
    """Executes autonomous Reinforcement Learning / Self-Play epochs on the selected neural network."""
    iters = min(max(req.iterations, 5), 100)
    stats = agent.train_batch_self_play(req.learning_model, iterations=iters)
    return stats


# Serve Frontend Static Assets
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
