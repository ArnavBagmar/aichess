"""Run repeatable, competition-shaped searches and print engine telemetry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import chess

from agent import Engine


@dataclass(frozen=True, slots=True)
class Position:
    name: str
    phase: str
    fen: str


POSITIONS = (
    Position("start", "opening", chess.STARTING_FEN),
    Position(
        "kiwipete",
        "tactical",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    Position(
        "development",
        "opening",
        "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2PP1N2/PP3PPP/RNBQR1K1 w - - 4 9",
    ),
    Position(
        "opposite_castles",
        "tactical",
        "r3kb1r/ppp2ppp/2n1bn2/3qp3/8/2NP1NP1/PPP2PBP/R1BQR1K1 w kq - 4 10",
    ),
    Position(
        "isolated_queen_pawn",
        "middlegame",
        "r2q1rk1/pp2bppp/2n1pn2/2bp4/8/2N1PN2/PPQB1PPP/2RR2K1 w - - 4 13",
    ),
    Position(
        "closed_center",
        "middlegame",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B1K2R w KQ - 4 9",
    ),
    Position(
        "check_evasion",
        "tactical",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ),
    Position(
        "rook_endgame",
        "endgame",
        "8/5pk1/6p1/3R3p/3r3P/5PP1/6K1/8 w - - 0 40",
    ),
    Position(
        "pawn_race",
        "endgame",
        "8/5pk1/8/3p4/3P4/4K3/5P2/8 w - - 0 45",
    ),
    Position(
        "minor_endgame",
        "endgame",
        "8/5pk1/3b2p1/3P3p/4P3/3B1P2/6PP/5K2 w - - 2 35",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clock-ms",
        type=int,
        default=3_000,
        help="remaining clock passed to the competition API for every benchmark position",
    )
    arguments = parser.parse_args()

    print(
        "name                 phase       move  depth   score    nodes      nps  "
        "q%  tt%  eval% pawn% first%  null  rfp  lmr  qprune"
    )
    totals = {"nodes": 0, "elapsed": 0.0, "qnodes": 0}
    for position in POSITIONS:
        board = chess.Board(position.fen)
        engine = Engine()
        move = engine.choose(board, arguments.clock_ms)
        stats = engine.stats
        nps = round(stats.nodes / stats.elapsed_s) if stats.elapsed_s else 0
        q_percent = 100 * stats.qnodes / stats.nodes if stats.nodes else 0
        tt_percent = 100 * stats.tt_hits / stats.tt_probes if stats.tt_probes else 0
        eval_percent = 100 * stats.eval_hits / stats.eval_calls if stats.eval_calls else 0
        pawn_percent = 100 * stats.pawn_hits / stats.pawn_calls if stats.pawn_calls else 0
        first_percent = (
            100 * stats.first_move_cutoffs / stats.beta_cutoffs if stats.beta_cutoffs else 0
        )
        print(
            f"{position.name:20} {position.phase:10} {move.uci():5} "
            f"{stats.completed_depth:5} {stats.score:7} {stats.nodes:8} {nps:8} "
            f"{q_percent:3.0f} {tt_percent:4.0f} {eval_percent:5.0f} {pawn_percent:5.0f} "
            f"{first_percent:6.0f} "
            f"{stats.null_cutoffs:4}/{stats.null_tries:<4} "
            f"{stats.reverse_futility_prunes:4} "
            f"{stats.lmr_researches:3}/{stats.lmr_reductions:<3} "
            f"{stats.see_prunes + stats.delta_prunes:6}"
        )
        totals["nodes"] += stats.nodes
        totals["elapsed"] += stats.elapsed_s
        totals["qnodes"] += stats.qnodes

    total_nps = round(totals["nodes"] / totals["elapsed"]) if totals["elapsed"] else 0
    total_q = 100 * totals["qnodes"] / totals["nodes"] if totals["nodes"] else 0
    print(
        f"\ntotal: {totals['nodes']:.0f} nodes in {totals['elapsed']:.3f}s, "
        f"{total_nps} nps, {total_q:.1f}% quiescence"
    )


if __name__ == "__main__":
    main()
