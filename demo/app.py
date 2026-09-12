"""Playable web demo of the engine: two nets, one search, per-move telemetry.

Runs the same `search.Searcher` the competition zip ships, with fixed think times
instead of the tournament clock, and reports what every search did (depth, nodes,
iteration by iteration, principal variation) so the page can show the arithmetic
behind a move. Serves on port 7860 for a Hugging Face Docker Space; `uvicorn
demo.app:app` from the repository root runs it locally.
"""

import math
import secrets
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT: Final = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bitboard as bb  # noqa: E402
from nnue_net import NetworkWeights, load_network  # noqa: E402
from search import Searcher  # noqa: E402
from search_kernel import MATE, MATE_THRESHOLD, TT_BITS  # noqa: E402

ENGINES: Final[dict[str, tuple[str, str]]] = {
    "net2": ("Gators (net2)", "nnue.npz"),
    "net5": ("net5 (latest)", "nnue-net5.npz"),
}
THINK_OPTIONS_MS: Final = (500, 1000, 3000, 8000)
MAX_SESSIONS: Final = 16
SCORE_PER_CP: Final = 32
PROBE_FEN: Final = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8"
LIMITS: Final = (
    "Competition build: one CPU core, 2 GB, 120 s + 0.5 s per move, 60 s import budget, "
    "no engines inside the zip. The same code and weights run here with a fixed think time."
)


@dataclass
class Session:
    engine: str
    searcher: Searcher
    board: chess.Board
    think_ms: int
    human: chess.Color
    created: float = field(default_factory=time.monotonic)
    nodes: int = 0
    seconds: float = 0.0


class NewGame(BaseModel):
    engine: str = "net2"
    color: str = "white"
    think_ms: int = 1000
    fen: str | None = None


class HumanMove(BaseModel):
    session: str
    uci: str


_lock = threading.Lock()
_nets: dict[str, NetworkWeights] = {}
_boot_nps: dict[str, int] = {}
_sessions: OrderedDict[str, Session] = OrderedDict()


def load_engines() -> None:
    """Load both nets, compile the kernels once, and measure boot speed."""
    for key, (_, filename) in ENGINES.items():
        net = load_network(ROOT / "weights" / filename)
        searcher = Searcher(net)
        searcher.warm_up()  # compiles the kernels on the first net; the second is instant
        searcher.pick(PROBE_FEN, 0, move_time_ms=1000)
        _nets[key] = net
        _boot_nps[key] = int(searcher.nodes / max(searcher.elapsed, 1e-6))
        print(f"{key}: warm, {_boot_nps[key]:,} nodes/s on the probe position")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Warm up in a thread so /api/info can answer "not ready yet" while kernels compile.
    worker = threading.Thread(target=load_engines, name="warm-up", daemon=True)
    worker.start()
    yield


app = FastAPI(title="AI Chessathon Engine", lifespan=_lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/info")
def info() -> dict[str, Any]:
    return {
        "engines": [{"key": key, "label": label} for key, (label, _) in ENGINES.items()],
        "think_options_ms": list(THINK_OPTIONS_MS),
        "boot_nps": dict(_boot_nps),
        "tt_entries": 1 << TT_BITS,
        "limits": LIMITS,
        "ready": len(_nets) == len(ENGINES),
    }


def _legal(board: chess.Board) -> list[str]:
    return [move.uci() for move in board.legal_moves]


def _outcome(board: chess.Board) -> tuple[str | None, str | None]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, None
    return outcome.result(), outcome.termination.name.lower().replace("_", " ")


def _san_list(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    position = board.copy(stack=False)
    out: list[str] = []
    for move in moves:
        if move not in position.legal_moves:
            break
        out.append(position.san(move))
        position.push(move)
    return out


def _engine_move(session: Session) -> dict[str, Any]:
    board = session.board
    searcher = session.searcher
    with _lock:
        move = searcher.pick(board.fen(), 60_000, move_time_ms=session.think_ms)
    nodes = searcher.nodes
    elapsed = max(searcher.elapsed, 1e-6)
    depth = searcher.iterations[-1][0] if searcher.iterations else 0
    score = searcher.score
    mate: int | None = None
    if abs(score) >= MATE_THRESHOLD:
        plies = MATE - abs(score)
        mate = (plies + 1) // 2 * (1 if score > 0 else -1)
    pv = _san_list(board, searcher.principal_variation(board))
    iterations = []
    for it_depth, it_nodes, it_score, it_move, seconds in searcher.iterations:
        candidate = bb.move_to_chess(it_move) if it_move >= 0 else None
        iterations.append(
            {
                "depth": it_depth,
                "nodes": it_nodes,
                "score_cp": int(it_score / SCORE_PER_CP),
                "time_ms": int(seconds * 1000),
                "move": board.san(candidate)
                if candidate is not None and candidate in board.legal_moves
                else "",
            }
        )
    san = board.san(move)
    board.push(move)
    session.nodes += nodes
    session.seconds += elapsed
    return {
        "uci": move.uci(),
        "san": san,
        "depth": depth,
        "nodes": nodes,
        "time_ms": int(elapsed * 1000),
        "nps": int(nodes / elapsed),
        "score_cp": int(score / SCORE_PER_CP),
        "mate": mate,
        "pv": pv,
        "iterations": iterations,
        "ebf": round(nodes ** (1 / depth), 2) if depth > 0 and nodes > 0 else None,
    }


def _state(session: Session, extra: dict[str, Any]) -> dict[str, Any]:
    result, reason = _outcome(session.board)
    payload: dict[str, Any] = {
        "fen": session.board.fen(),
        "legal": _legal(session.board) if result is None else [],
        "result": result,
        "reason": reason,
        "totals": {"nodes": session.nodes, "time_s": round(session.seconds, 2)},
    }
    payload.update(extra)
    return payload


def _get(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown or expired session; start a new game")
    _sessions.move_to_end(session_id)
    return session


@app.post("/api/new")
def new_game(request: NewGame) -> dict[str, Any]:
    if len(_nets) < len(ENGINES):
        raise HTTPException(503, "engines are still warming up")
    if request.engine not in ENGINES:
        raise HTTPException(400, f"unknown engine {request.engine!r}")
    if request.color not in ("white", "black"):
        raise HTTPException(400, "color must be 'white' or 'black'")
    if request.think_ms not in THINK_OPTIONS_MS:
        raise HTTPException(400, f"think_ms must be one of {THINK_OPTIONS_MS}")
    try:
        board = chess.Board(request.fen) if request.fen else chess.Board()
    except ValueError as error:
        raise HTTPException(400, f"bad FEN: {error}") from error
    if not board.is_valid():
        raise HTTPException(400, "that position is not legal")
    session = Session(
        engine=request.engine,
        searcher=Searcher(_nets[request.engine]),
        board=board,
        think_ms=request.think_ms,
        human=chess.WHITE if request.color == "white" else chess.BLACK,
    )
    session_id = secrets.token_urlsafe(8)
    _sessions[session_id] = session
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    engine_move = None
    if board.turn != session.human and _outcome(board)[0] is None:
        engine_move = _engine_move(session)
    return _state(session, {"session": session_id, "engine_move": engine_move})


@app.post("/api/move")
def human_move(request: HumanMove) -> dict[str, Any]:
    session = _get(request.session)
    board = session.board
    if _outcome(board)[0] is not None:
        raise HTTPException(400, "the game is over; start a new one")
    if board.turn != session.human:
        raise HTTPException(400, "it is the engine's turn")
    try:
        move = chess.Move.from_uci(request.uci)
    except ValueError as error:
        raise HTTPException(400, f"malformed move {request.uci!r}") from error
    if move not in board.legal_moves:
        raise HTTPException(400, f"{request.uci} is not legal here")
    san = board.san(move)
    board.push(move)
    engine_move = None
    if _outcome(board)[0] is None:
        engine_move = _engine_move(session)
    return _state(session, {"san": san, "engine_move": engine_move})


def _si(value: float) -> str:
    """Compact number for log lines; the page formats its own."""
    if value >= 1e9:
        return f"{value / 1e9:.1f}G"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


def branching_note(nodes: int, depth: int, branching: float = 35.0) -> str:
    """The comparison the page draws: minimax, ideal alpha-beta, and this search."""
    minimax = branching**depth
    alpha_beta = branching ** (depth / 2)
    ebf = nodes ** (1 / depth) if depth > 0 else float("nan")
    return (
        f"depth {depth}: minimax ~{_si(minimax)}, alpha-beta ~{_si(alpha_beta)}, "
        f"searched {_si(nodes)} nodes, effective branching factor {ebf:.2f} "
        f"(log10 nodes = {math.log10(max(nodes, 1)):.2f})"
    )
