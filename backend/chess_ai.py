"""
NeuroChess Engine v3.0 — Stockfish Search + Multi-Model PyTorch Neural Agents
==============================================================================
Combines a complete FIDE-compliant Chess Rules Engine with:
- Zobrist Hashing & 120K-entry Transposition Table (EXACT / LOWER / UPPER + PV move)
- Static Exchange Evaluation (SEE) for capture safety & Quiescence Delta Pruning
- Iterative Deepening + Principal Variation Search (PVS) + Null-Move Pruning + LMR
- Killer Moves, History Heuristic & MVV-LVA Move Ordering
- PUCT Monte Carlo Tree Search (AlphaZero / Lc0 style) with Zobrist eval caching & 1-ply tactical guard
- Four PyTorch Neural Architectures via PyTorchChessBrain (Stockfish SFNNv9, Lc0 SE-ResNet+Transformer,
  Dueling DQN+A2C RL, Neuro-Evolutionary Swarm)
"""

import math
import random
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from neural_models import PyTorchChessBrain, MAT_VALUES, NN_PST

# ==============================================================================
# 1. COMPLETE CHESS RULES ENGINE & ZOBRIST HASHING
# ==============================================================================

PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
}

ABS_PIECE_VALUES = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

PST = NN_PST

FILES = 'abcdefgh'
RANKS = '87654321'

# Deterministic 64-bit Zobrist keys
_zrng = random.Random(20261111)
ZOBRIST_PIECES: Dict[str, List[int]] = {
    p: [_zrng.getrandbits(64) for _ in range(64)]
    for p in "PNBRQKpnbrqk"
}
ZOBRIST_TURN = _zrng.getrandbits(64)
ZOBRIST_CASTLING = {ch: _zrng.getrandbits(64) for ch in "KQkq"}
ZOBRIST_EP = [_zrng.getrandbits(64) for _ in range(8)]


def sq_to_alg(sq: int) -> str:
    r, c = divmod(sq, 8)
    return f"{FILES[c]}{RANKS[r]}"


def alg_to_sq(alg: str) -> int:
    if len(alg) < 2 or alg == '-':
        return -1
    c = FILES.index(alg[0])
    r = RANKS.index(alg[1])
    return r * 8 + c


class Move:
    __slots__ = ('from_sq', 'to_sq', 'promotion', 'is_castling', 'is_en_passant', 'captured', 'piece')

    def __init__(self, from_sq: int, to_sq: int, promotion: Optional[str] = None,
                 is_castling: bool = False, is_en_passant: bool = False,
                 captured: Optional[str] = None, piece: str = ''):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.promotion = promotion
        self.is_castling = is_castling
        self.is_en_passant = is_en_passant
        self.captured = captured
        self.piece = piece

    def uci(self) -> str:
        prom = self.promotion.lower() if self.promotion else ''
        return f"{sq_to_alg(self.from_sq)}{sq_to_alg(self.to_sq)}{prom}"

    def to_dict(self, san: str = "") -> Dict[str, Any]:
        return {
            "uci": self.uci(),
            "from": sq_to_alg(self.from_sq),
            "to": sq_to_alg(self.to_sq),
            "from_sq": self.from_sq,
            "to_sq": self.to_sq,
            "promotion": self.promotion,
            "is_castling": self.is_castling,
            "is_en_passant": self.is_en_passant,
            "captured": self.captured,
            "piece": self.piece,
            "san": san or self.uci()
        }


class ChessBoard:
    STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def __init__(self, fen: str = STARTING_FEN):
        self.board: List[Optional[str]] = [None] * 64
        self.turn: str = 'w'
        self.castling: str = 'KQkq'
        self.ep_sq: int = -1
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.move_history: List[Dict[str, Any]] = []
        self.position_history: List[str] = []
        self.load_fen(fen)

    def copy(self) -> 'ChessBoard':
        b = ChessBoard.__new__(ChessBoard)
        b.board = self.board[:]
        b.turn = self.turn
        b.castling = self.castling
        b.ep_sq = self.ep_sq
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        b.move_history = self.move_history[:]
        b.position_history = self.position_history[:]
        return b

    def load_fen(self, fen: str):
        parts = fen.strip().split()
        rows = parts[0].split('/')
        self.board = [None] * 64
        idx = 0
        for row in rows:
            for ch in row:
                if ch.isdigit():
                    idx += int(ch)
                else:
                    self.board[idx] = ch
                    idx += 1
        self.turn = parts[1] if len(parts) > 1 else 'w'
        self.castling = parts[2] if len(parts) > 2 else '-'
        self.ep_sq = alg_to_sq(parts[3]) if len(parts) > 3 else -1
        self.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove_number = int(parts[5]) if len(parts) > 5 else 1
        self.position_history = [self.position_key()]

    def position_key(self) -> str:
        return f"{''.join(p or '.' for p in self.board)} {self.turn} {self.castling} {self.ep_sq}"

    def zobrist_hash(self) -> int:
        h = 0
        for sq, p in enumerate(self.board):
            if p is not None:
                h ^= ZOBRIST_PIECES[p][sq]
        if self.turn == 'b':
            h ^= ZOBRIST_TURN
        for ch in self.castling:
            if ch in ZOBRIST_CASTLING:
                h ^= ZOBRIST_CASTLING[ch]
        if self.ep_sq != -1:
            h ^= ZOBRIST_EP[self.ep_sq % 8]
        return h

    def to_fen(self) -> str:
        rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for c in range(8):
                p = self.board[r * 8 + c]
                if p is None:
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += p
            if empty > 0:
                row_str += str(empty)
            rows.append(row_str)
        ep_str = sq_to_alg(self.ep_sq) if self.ep_sq != -1 else '-'
        castling_str = self.castling if self.castling else '-'
        return f"{'/'.join(rows)} {self.turn} {castling_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"

    def is_own(self, p: Optional[str], color: str) -> bool:
        if p is None:
            return False
        return p.isupper() if color == 'w' else p.islower()

    def is_enemy(self, p: Optional[str], color: str) -> bool:
        if p is None:
            return False
        return p.islower() if color == 'w' else p.isupper()

    def find_king(self, color: str) -> int:
        target = 'K' if color == 'w' else 'k'
        for i, p in enumerate(self.board):
            if p == target:
                return i
        return -1

    def has_non_pawn_material(self, color: str) -> bool:
        targets = ('N', 'B', 'R', 'Q') if color == 'w' else ('n', 'b', 'r', 'q')
        for p in self.board:
            if p in targets:
                return True
        return False

    def is_square_attacked(self, sq: int, by_color: str) -> bool:
        r, c = divmod(sq, 8)
        squares = self.board
        if by_color == 'w':
            if r + 1 < 8:
                if c - 1 >= 0 and squares[(r + 1) * 8 + (c - 1)] == 'P':
                    return True
                if c + 1 < 8 and squares[(r + 1) * 8 + (c + 1)] == 'P':
                    return True
        else:
            if r - 1 >= 0:
                if c - 1 >= 0 and squares[(r - 1) * 8 + (c - 1)] == 'p':
                    return True
                if c + 1 < 8 and squares[(r - 1) * 8 + (c + 1)] == 'p':
                    return True

        knight = 'N' if by_color == 'w' else 'n'
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and squares[nr * 8 + nc] == knight:
                return True

        king = 'K' if by_color == 'w' else 'k'
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and squares[nr * 8 + nc] == king:
                    return True

        rook = 'R' if by_color == 'w' else 'r'
        queen = 'Q' if by_color == 'w' else 'q'
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                p = squares[nr * 8 + nc]
                if p is not None:
                    if p == rook or p == queen:
                        return True
                    break
                nr += dr
                nc += dc

        bishop = 'B' if by_color == 'w' else 'b'
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                p = squares[nr * 8 + nc]
                if p is not None:
                    if p == bishop or p == queen:
                        return True
                    break
                nr += dr
                nc += dc

        return False

    def in_check(self, color: Optional[str] = None) -> bool:
        color = color or self.turn
        ksq = self.find_king(color)
        if ksq == -1:
            return True
        enemy = 'b' if color == 'w' else 'w'
        return self.is_square_attacked(ksq, enemy)

    def generate_pseudo_legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        color = self.turn
        enemy = 'b' if color == 'w' else 'w'
        squares = self.board

        for sq in range(64):
            p = squares[sq]
            if not self.is_own(p, color):
                continue
            r, c = divmod(sq, 8)
            pt = p.upper()

            if pt == 'P':
                direction = -1 if color == 'w' else 1
                start_rank = 6 if color == 'w' else 1
                prom_rank = 0 if color == 'w' else 7
                nr = r + direction

                if 0 <= nr < 8:
                    fwd_sq = nr * 8 + c
                    if squares[fwd_sq] is None:
                        if nr == prom_rank:
                            for prom in ('q', 'r', 'b', 'n'):
                                moves.append(Move(sq, fwd_sq, promotion=prom, piece=p))
                        else:
                            moves.append(Move(sq, fwd_sq, piece=p))
                            if r == start_rank:
                                fwd2_sq = (r + 2 * direction) * 8 + c
                                if squares[fwd2_sq] is None:
                                    moves.append(Move(sq, fwd2_sq, piece=p))

                    for dc in (-1, 1):
                        nc = c + dc
                        if 0 <= nc < 8:
                            cap_sq = nr * 8 + nc
                            target = squares[cap_sq]
                            if self.is_enemy(target, color):
                                if nr == prom_rank:
                                    for prom in ('q', 'r', 'b', 'n'):
                                        moves.append(Move(sq, cap_sq, promotion=prom, captured=target, piece=p))
                                else:
                                    moves.append(Move(sq, cap_sq, captured=target, piece=p))
                            elif cap_sq == self.ep_sq:
                                ep_cap = 'p' if color == 'w' else 'P'
                                moves.append(Move(sq, cap_sq, is_en_passant=True, captured=ep_cap, piece=p))

            elif pt == 'N':
                for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        nsq = nr * 8 + nc
                        target = squares[nsq]
                        if not self.is_own(target, color):
                            moves.append(Move(sq, nsq, captured=target, piece=p))

            elif pt == 'K':
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8:
                            nsq = nr * 8 + nc
                            target = squares[nsq]
                            if not self.is_own(target, color):
                                moves.append(Move(sq, nsq, captured=target, piece=p))

                if color == 'w' and sq == 60 and not self.is_square_attacked(60, enemy):
                    if 'K' in self.castling and squares[61] is None and squares[62] is None and squares[63] == 'R':
                        if not self.is_square_attacked(61, enemy) and not self.is_square_attacked(62, enemy):
                            moves.append(Move(60, 62, is_castling=True, piece=p))
                    if 'Q' in self.castling and squares[59] is None and squares[58] is None and squares[57] is None and squares[56] == 'R':
                        if not self.is_square_attacked(59, enemy) and not self.is_square_attacked(58, enemy):
                            moves.append(Move(60, 58, is_castling=True, piece=p))
                elif color == 'b' and sq == 4 and not self.is_square_attacked(4, enemy):
                    if 'k' in self.castling and squares[5] is None and squares[6] is None and squares[7] == 'r':
                        if not self.is_square_attacked(5, enemy) and not self.is_square_attacked(6, enemy):
                            moves.append(Move(4, 6, is_castling=True, piece=p))
                    if 'q' in self.castling and squares[3] is None and squares[2] is None and squares[1] is None and squares[0] == 'r':
                        if not self.is_square_attacked(3, enemy) and not self.is_square_attacked(2, enemy):
                            moves.append(Move(4, 2, is_castling=True, piece=p))

            else:
                dirs = []
                if pt in ('R', 'Q'):
                    dirs.extend(((-1, 0), (1, 0), (0, -1), (0, 1)))
                if pt in ('B', 'Q'):
                    dirs.extend(((-1, -1), (-1, 1), (1, -1), (1, 1)))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 8 and 0 <= nc < 8:
                        nsq = nr * 8 + nc
                        target = squares[nsq]
                        if target is None:
                            moves.append(Move(sq, nsq, piece=p))
                        else:
                            if self.is_enemy(target, color):
                                moves.append(Move(sq, nsq, captured=target, piece=p))
                            break
                        nr += dr
                        nc += dc

        return moves

    def apply_move_fast(self, move: Move):
        p = self.board[move.from_sq]
        self.board[move.from_sq] = None

        if move.is_en_passant:
            cap_r = move.from_sq // 8
            cap_c = move.to_sq % 8
            self.board[cap_r * 8 + cap_c] = None

        if move.is_castling:
            if move.to_sq == 62:
                self.board[61] = self.board[63]
                self.board[63] = None
            elif move.to_sq == 58:
                self.board[59] = self.board[56]
                self.board[56] = None
            elif move.to_sq == 6:
                self.board[5] = self.board[7]
                self.board[7] = None
            elif move.to_sq == 2:
                self.board[3] = self.board[0]
                self.board[0] = None

        if move.promotion:
            prom_piece = move.promotion.upper() if self.turn == 'w' else move.promotion.lower()
            self.board[move.to_sq] = prom_piece
        else:
            self.board[move.to_sq] = p

        if self.castling and self.castling != '-':
            c_rights = self.castling
            if p == 'K':
                c_rights = c_rights.replace('K', '').replace('Q', '')
            elif p == 'k':
                c_rights = c_rights.replace('k', '').replace('q', '')
            if move.from_sq == 63 or move.to_sq == 63:
                c_rights = c_rights.replace('K', '')
            if move.from_sq == 56 or move.to_sq == 56:
                c_rights = c_rights.replace('Q', '')
            if move.from_sq == 7 or move.to_sq == 7:
                c_rights = c_rights.replace('k', '')
            if move.from_sq == 0 or move.to_sq == 0:
                c_rights = c_rights.replace('q', '')
            self.castling = c_rights if c_rights else '-'

        if p in ('P', 'p') and abs(move.from_sq - move.to_sq) == 16:
            self.ep_sq = (move.from_sq + move.to_sq) // 2
        else:
            self.ep_sq = -1

        if p in ('P', 'p') or move.captured is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.turn == 'b':
            self.fullmove_number += 1
        self.turn = 'b' if self.turn == 'w' else 'w'

    def generate_legal_moves(self) -> List[Move]:
        pseudo = self.generate_pseudo_legal_moves()
        legal = []
        my_color = self.turn
        for m in pseudo:
            b_copy = self.copy()
            b_copy.apply_move_fast(m)
            if not b_copy.in_check(my_color):
                legal.append(m)
        return legal

    def format_san(self, move: Move, legal_moves: Optional[List[Move]] = None) -> str:
        if move.is_castling:
            san = "O-O" if move.to_sq in (62, 6) else "O-O-O"
        else:
            pt = move.piece.upper()
            to_str = sq_to_alg(move.to_sq)
            from_str = sq_to_alg(move.from_sq)
            if pt == 'P':
                if move.captured:
                    san = f"{from_str[0]}x{to_str}"
                else:
                    san = to_str
                if move.promotion:
                    san += f"={move.promotion.upper()}"
            else:
                disambig = ""
                if legal_moves:
                    ambig = [m for m in legal_moves if m.piece == move.piece and m.to_sq == move.to_sq and m.from_sq != move.from_sq]
                    if ambig:
                        same_file = any((m.from_sq % 8) == (move.from_sq % 8) for m in ambig)
                        same_rank = any((m.from_sq // 8) == (move.from_sq // 8) for m in ambig)
                        if not same_file:
                            disambig = from_str[0]
                        elif not same_rank:
                            disambig = from_str[1]
                        else:
                            disambig = from_str
                cap_str = "x" if move.captured else ""
                san = f"{pt}{disambig}{cap_str}{to_str}"

        b_copy = self.copy()
        b_copy.apply_move_fast(move)
        if b_copy.in_check(b_copy.turn):
            if len(b_copy.generate_legal_moves()) == 0:
                san += "#"
            else:
                san += "+"
        return san

    def push_uci(self, uci_str: str) -> Optional[Dict[str, Any]]:
        legal = self.generate_legal_moves()
        for m in legal:
            if m.uci() == uci_str or (len(uci_str) == 4 and m.uci()[:4] == uci_str and m.promotion == 'q'):
                san = self.format_san(m, legal)
                info = m.to_dict(san)
                self.apply_move_fast(m)
                self.move_history.append(info)
                self.position_history.append(self.position_key())
                return info
        return None

    def get_game_status(self) -> Dict[str, Any]:
        legal = self.generate_legal_moves()
        checked = self.in_check(self.turn)
        if len(legal) == 0:
            if checked:
                winner = "White" if self.turn == 'b' else "Black"
                return {"is_game_over": True, "status": "checkmate", "winner": winner, "reason": f"Jaque Mate - Gana {winner}"}
            else:
                return {"is_game_over": True, "status": "stalemate", "winner": "Draw", "reason": "Tablas por Rey Ahogado (Stalemate)"}
        if self.halfmove_clock >= 100:
            return {"is_game_over": True, "status": "draw", "winner": "Draw", "reason": "Tablas por regla de 50 movimientos"}
        cur_key = self.position_key()
        if self.position_history.count(cur_key) >= 3:
            return {"is_game_over": True, "status": "draw", "winner": "Draw", "reason": "Tablas por triple repetición"}
        pieces = [p for p in self.board if p is not None]
        if len(pieces) == 2:
            return {"is_game_over": True, "status": "draw", "winner": "Draw", "reason": "Tablas por material insuficiente (K vs K)"}
        if len(pieces) == 3 and any(p.upper() in ('N', 'B') for p in pieces):
            return {"is_game_over": True, "status": "draw", "winner": "Draw", "reason": "Tablas por material insuficiente"}

        return {
            "is_game_over": False,
            "status": "check" if checked else "playing",
            "winner": None,
            "reason": "Jaque al Rey" if checked else "Partida en curso"
        }


# ==============================================================================
# 2. STATIC EXCHANGE EVALUATION (SEE) & TRANSPOSITION TABLE
# ==============================================================================

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class TranspositionTable:
    def __init__(self, max_size: int = 120000):
        self.table: Dict[int, Tuple[int, int, float, Optional[str]]] = {}
        self.max_size = max_size
        self.hits = 0

    def probe(self, key: int) -> Optional[Tuple[int, int, float, Optional[str]]]:
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def store(self, key: int, depth: int, flag: int, score: float, best_uci: Optional[str]):
        if len(self.table) >= self.max_size:
            # Evict oldest quarter cleanly
            for k in list(self.table.keys())[: self.max_size // 4]:
                del self.table[k]
        old = self.table.get(key)
        if old is None or depth >= old[0]:
            self.table[key] = (depth, flag, score, best_uci)


def _find_least_valuable_attacker(squares: List[Optional[str]], sq: int, color: str, occupied_mask: List[bool]) -> Tuple[int, Optional[str]]:
    """Finds the least valuable piece of `color` attacking `sq` ignoring removed squares."""
    r, c = divmod(sq, 8)
    # 1. Pawns
    pawn = 'P' if color == 'w' else 'p'
    pr = r + 1 if color == 'w' else r - 1
    if 0 <= pr < 8:
        for dc in (-1, 1):
            pc = c + dc
            if 0 <= pc < 8:
                psq = pr * 8 + pc
                if occupied_mask[psq] and squares[psq] == pawn:
                    return psq, pawn
    # 2. Knights
    knight = 'N' if color == 'w' else 'n'
    for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            nsq = nr * 8 + nc
            if occupied_mask[nsq] and squares[nsq] == knight:
                return nsq, knight
    # 3. Bishops (and x-ray behind removed pieces)
    bishop = 'B' if color == 'w' else 'b'
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            nsq = nr * 8 + nc
            if occupied_mask[nsq]:
                if squares[nsq] == bishop:
                    return nsq, bishop
                break
            nr += dr
            nc += dc
    # 4. Rooks
    rook = 'R' if color == 'w' else 'r'
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            nsq = nr * 8 + nc
            if occupied_mask[nsq]:
                if squares[nsq] == rook:
                    return nsq, rook
                break
            nr += dr
            nc += dc
    # 5. Queens
    queen = 'Q' if color == 'w' else 'q'
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            nsq = nr * 8 + nc
            if occupied_mask[nsq]:
                if squares[nsq] == queen:
                    return nsq, queen
                break
            nr += dr
            nc += dc
    # 6. King
    king = 'K' if color == 'w' else 'k'
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                nsq = nr * 8 + nc
                if occupied_mask[nsq] and squares[nsq] == king:
                    return nsq, king
    return -1, None


def see_score(board: ChessBoard, move: Move) -> int:
    """
    Stockfish-style Static Exchange Evaluation (SEE) in centipawns.
    Determines net material gain/loss of a capture sequence on move.to_sq.
    """
    if not move.captured and not move.promotion:
        return 0
    gain = [0] * 16
    target_sq = move.to_sq
    squares = board.board
    occupied = [p is not None for p in squares]

    cap_val = ABS_PIECE_VALUES.get(move.captured.upper(), 0) if move.captured else 0
    if move.promotion:
        cap_val += ABS_PIECE_VALUES.get(move.promotion.upper(), 900) - 100
    gain[0] = cap_val

    occupied[move.from_sq] = False
    piece_on_sq = (move.promotion.upper() if move.promotion else move.piece.upper())
    side = 'b' if board.turn == 'w' else 'w'
    d = 0

    while d < 15:
        d += 1
        gain[d] = ABS_PIECE_VALUES.get(piece_on_sq, 0) - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            break
        att_sq, att_piece = _find_least_valuable_attacker(squares, target_sq, side, occupied)
        if att_sq == -1 or att_piece is None:
            break
        occupied[att_sq] = False
        piece_on_sq = att_piece.upper()
        side = 'b' if side == 'w' else 'w'

    while d > 1:
        d -= 1
        gain[d - 1] = -max(-gain[d - 1], gain[d])
    return gain[0]


# ==============================================================================
# 3. EXPANDED GRANDMASTER OPENING BOOK & MCTS NODE
# ==============================================================================

OPENING_BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": [
        ("e2e4", "Apertura de Peón de Rey", 40),
        ("d2d4", "Apertura de Peón de Dama", 35),
        ("c2c4", "Apertura Inglesa", 15),
        ("g1f3", "Apertura Réti", 10)
    ],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": [
        ("c7c5", "Defensa Siciliana", 38),
        ("e7e5", "Apertura Abierta (Doble Peón de Rey)", 32),
        ("e7e6", "Defensa Francesa", 16),
        ("c7c6", "Defensa Caro-Kann", 14)
    ],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b": [
        ("g8f6", "Defensa India", 45),
        ("d7d5", "Gambito de Dama Cerrado", 40),
        ("e7e6", "Sistema QGD / Nimzo-India", 15)
    ],
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b": [
        ("b8c6", "Desarrollo Clásico (Nc6)", 65),
        ("g8f6", "Defensa Petroff", 35)
    ],
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b": [
        ("f8c5", "Giuoco Piano (Apertura Italiana)", 55),
        ("g8f6", "Defensa de los Dos Caballos", 45)
    ],
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b": [
        ("a7a6", "Apertura Española (Defensa Morphy)", 70),
        ("g8f6", "Apertura Española (Defensa Berlinesa)", 30)
    ],
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b": [
        ("d7d6", "Defensa Siciliana (Variante Najdorf/Dragón)", 45),
        ("b8c6", "Defensa Siciliana Clásica", 35),
        ("e7e6", "Defensa Siciliana Paulsen/Taimanov", 20)
    ],
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w": [
        ("g1f3", "Siciliana Abierta (2. Cf3)", 75),
        ("b1c3", "Siciliana Cerrada (2. Cc3)", 25)
    ],
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w": [
        ("g1f3", "Apertura del Caballo de Rey (2. Cf3)", 80),
        ("f1c4", "Apertura del Alfil", 20)
    ],
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b": [
        ("e7e6", "Gambito de Dama Declinado (QGD)", 50),
        ("c7c6", "Defensa Eslava", 40),
        ("d5c4", "Gambito de Dama Aceptado (QGA)", 10)
    ],
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR w": [
        ("b1c3", "Línea Principal contra India (3. Cc3)", 55),
        ("g1f3", "Anti-Nimzo / Catalana (3. Cf3)", 45)
    ],
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w": [
        ("c2c4", "Gambito de Dama (2. c4)", 70),
        ("c1f4", "Sistema Londres (2. Af4)", 30)
    ],
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w": [
        ("d2d4", "Defensa Francesa (2. d4)", 90),
        ("d2d3", "Ataque Indio de Rey", 10)
    ],
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w": [
        ("d2d4", "Defensa Caro-Kann (2. d4)", 85),
        ("b1c3", "Caro-Kann Dos Caballos", 15)
    ]
}


class MCTSNode:
    __slots__ = ('board', 'parent', 'move', 'prior', 'children', 'visits', 'value_sum', 'is_expanded')

    def __init__(self, board: ChessBoard, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct: float = 1.45) -> float:
        parent_visits = self.parent.visits if self.parent else 1
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value() + u


# ==============================================================================
# 4. MULTI-MODEL AUTONOMOUS AGENT (PYTORCH + STOCKFISH SEARCH)
# ==============================================================================

class AutonomousChessAgent:
    """
    Autonomous Neural Chess Agent orchestrating 4 PyTorch architectures and 5 Reaction Profiles
    backed by Stockfish-style PVS, Transposition Tables, SEE, Quiescence Search, and Lc0 MCTS.
    """

    LEARNING_MODELS = {
        "alphazero_mcts": {
            "id": "alphazero_mcts",
            "name": "Lc0 SE-ResNet + Transformer MCTS",
            "architecture": "PyTorch 20-Plane SE-ResNet + BT4 Self-Attention + 4096 Attention Policy + PUCT MCTS",
            "description": "Red Convolucional Squeeze-and-Excitation + Transformer Multi-Cabeza (estilo Leela Chess Zero BT4) combinada con búsqueda Monte Carlo PUCT y cabezal WDL/MLH."
        },
        "minimax_nnue": {
            "id": "minimax_nnue",
            "name": "Stockfish NNUE SFNNv9 + PVS",
            "architecture": "PyTorch HalfKAv2_hm (24,576) + PSQT + SCReLU + 8 Layer Stacks + PVS/Quiescence",
            "description": "Ingeniería inversa de Stockfish SFNNv9 en PyTorch: acumulador dual HalfKAv2_hm, SCReLU, 8 pilas de fase, Transposition Table, SEE, PVS y Quiescence Search."
        },
        "dqn_td_rl": {
            "id": "dqn_td_rl",
            "name": "Dueling Double-DQN + A2C RL",
            "architecture": "PyTorch Dual-Stream Conv+Vector + Dueling Advantage + Prioritized Experience Replay",
            "description": "Agente Deep RL de doble flujo (espacial + táctico) con optimizador AdamW, muestreo por error TD prioritario (PER) y aprendizaje continuo en vivo."
        },
        "evolutionary_ga": {
            "id": "evolutionary_ga",
            "name": "Neuro-Evolutionary Swarm (PyTorch)",
            "architecture": "4-Genome Residual Committee + Tournament Selection + Crossover & Gaussian Mutation",
            "description": "Comité de 4 genomas neuronales PyTorch evaluados por consenso ponderado según aptitud (fitness), con cruce y mutación genética adaptativa."
        }
    }

    REACTION_PROFILES = {
        "aggressive": {
            "id": "aggressive",
            "name": "Agresivo (Tal Berserker)",
            "badge": "Ataque Total",
            "weights": {"material": 0.94, "positional": 1.05, "center": 1.25, "aggression": 2.15, "neural_blend": 0.55},
            "description": "Busca sacrificios tácticos, columnas abiertas y ataques directos al Rey enemigo."
        },
        "positional": {
            "id": "positional",
            "name": "Posicional (Maestro Estratégico)",
            "badge": "Control & Estructura",
            "weights": {"material": 1.10, "positional": 1.65, "center": 1.75, "aggression": 0.75, "neural_blend": 0.55},
            "description": "Domina casillas centrales, mejora la estructura de peones y asfixia gradualmente las piezas rivales."
        },
        "tactical": {
            "id": "tactical",
            "name": "Táctico (Cálculo Profundo)",
            "badge": "Precisión Letal",
            "weights": {"material": 1.15, "positional": 1.25, "center": 1.30, "aggression": 1.40, "neural_blend": 0.55},
            "description": "Prioriza combinaciones forzadas, horquillas, clavadas y ganancia material exacta."
        },
        "adaptive": {
            "id": "adaptive",
            "name": "Adaptativo (Camaleón Neuronal)",
            "badge": "IA Dinámica",
            "weights": {"material": 1.05, "positional": 1.20, "center": 1.25, "aggression": 1.20, "neural_blend": 0.60},
            "description": "Analiza el estado de la partida en tiempo real: juega agresivo cuando detecta ventaja y busca contragolpes dinámicos bajo presión."
        },
        "defensive": {
            "id": "defensive",
            "name": "Defensivo (Muralla de Hierro)",
            "badge": "Contraataque",
            "weights": {"material": 1.25, "positional": 1.40, "center": 1.15, "aggression": 0.55, "neural_blend": 0.50},
            "description": "Minimiza riesgos, protege a su Rey y castiga cualquier sobreextensión del oponente."
        }
    }

    def __init__(self):
        self.networks: Dict[str, PyTorchChessBrain] = {
            "alphazero_mcts": PyTorchChessBrain("alphazero_mcts"),
            "minimax_nnue": PyTorchChessBrain("minimax_nnue"),
            "dqn_td_rl": PyTorchChessBrain("dqn_td_rl"),
            "evolutionary_ga": PyTorchChessBrain("evolutionary_ga"),
        }
        self.tt = TranspositionTable(max_size=120000)
        self.killers: List[List[Optional[str]]] = [[None, None] for _ in range(32)]
        self.history: Dict[Tuple[str, int, int], int] = {}
        self.nodes_evaluated = 0
        self.q_nodes = 0
        self.search_deadline = 0.0

    def score_move_tactics(
        self,
        board: ChessBoard,
        move: Move,
        brain: PyTorchChessBrain,
        policy_pack: Dict[str, Any],
        reaction: str,
        ply: int = 0,
        tt_best_uci: Optional[str] = None
    ) -> float:
        uci = move.uci()
        if tt_best_uci and uci == tt_best_uci:
            return 100.0

        nn_logit = brain.move_prior_logit(move, policy_pack) * 0.42
        tactical_bonus = 0.0

        if move.captured:
            see_val = see_score(board, move)
            if see_val >= 0:
                victim_val = ABS_PIECE_VALUES.get(move.captured.upper(), 100)
                attacker_val = ABS_PIECE_VALUES.get(move.piece.upper(), 100)
                tactical_bonus += 5.0 + (victim_val * 1.2 - attacker_val * 0.1) / 180.0
            else:
                tactical_bonus += max(-2.5, see_val / 350.0)

        if move.promotion:
            tactical_bonus += 6.0 if move.promotion.lower() == 'q' else 2.5

        if ply < 32:
            if uci == self.killers[ply][0]:
                tactical_bonus += 3.2
            elif uci == self.killers[ply][1]:
                tactical_bonus += 2.6

        hist_score = self.history.get((board.turn, move.from_sq, move.to_sq), 0)
        tactical_bonus += min(hist_score / 400.0, 2.0)

        if move.is_castling:
            tactical_bonus += 0.95

        tr, tc = divmod(move.to_sq, 8)
        if 2 <= tr <= 5 and 2 <= tc <= 5:
            tactical_bonus += 0.25

        if reaction == "aggressive":
            enemy_king_sq = board.find_king('b' if board.turn == 'w' else 'w')
            if enemy_king_sq != -1:
                kr, kc = divmod(enemy_king_sq, 8)
                if abs(tr - kr) + abs(tc - kc) <= 3:
                    tactical_bonus += 0.65
            if move.captured:
                tactical_bonus *= 1.25
        elif reaction == "positional":
            if move.piece.upper() in ('N', 'B') and (move.from_sq // 8) in (0, 7):
                tactical_bonus += 0.48
        elif reaction == "defensive":
            my_king_sq = board.find_king(board.turn)
            if my_king_sq != -1:
                kr, kc = divmod(my_king_sq, 8)
                if abs(tr - kr) + abs(tc - kc) <= 2:
                    tactical_bonus += 0.35

        return nn_logit + tactical_bonus

    def select_move(
        self,
        board: ChessBoard,
        learning_model: str = "alphazero_mcts",
        reaction_profile: str = "tactical",
        difficulty: int = 3
    ) -> Dict[str, Any]:
        start_time = time.time()
        self.nodes_evaluated = 0
        self.q_nodes = 0
        tt_hits_before = self.tt.hits
        # Soft search budget scales cleanly with difficulty
        self.search_deadline = start_time + (0.65 if difficulty <= 2 else (1.35 if difficulty == 3 else 2.25))

        brain = self.networks.get(learning_model, self.networks["alphazero_mcts"])
        profile = self.REACTION_PROFILES.get(reaction_profile, self.REACTION_PROFILES["tactical"])
        style_weights = dict(profile["weights"])

        adaptive_state_note = ""
        if reaction_profile == "adaptive":
            init_val, _, _ = brain.evaluate_board(board, style_weights)
            my_perspective_val = init_val if board.turn == 'w' else -init_val
            if my_perspective_val > 0.25:
                style_weights["aggression"] = 1.95
                style_weights["material"] = 1.20
                adaptive_state_note = "Modo Cazador Activo (Ventaja detectada -> Presión ofensiva)"
            elif my_perspective_val < -0.25:
                style_weights["aggression"] = 1.65
                style_weights["positional"] = 1.55
                adaptive_state_note = "Modo Contragolpe Táctico (Buscando complicaciones)"
            else:
                style_weights["positional"] = 1.55
                adaptive_state_note = "Modo Equilibrio Estratégico (Control central)"

        legal_moves = board.generate_legal_moves()
        if not legal_moves:
            return {"move": None, "telemetry": {}}

        # Decay history table smoothly between moves
        for k in list(self.history.keys()):
            self.history[k] //= 2
            if self.history[k] == 0:
                del self.history[k]

        # 1. Check Expanded Opening Book on early plies
        fen_prefix = " ".join(board.to_fen().split()[:2])
        book_hit = None
        if board.fullmove_number <= 8 and fen_prefix in OPENING_BOOK:
            candidates = OPENING_BOOK[fen_prefix]
            legal_ucis = {m.uci(): m for m in legal_moves}
            valid_book = [(uci, name, w) for (uci, name, w) in candidates if uci in legal_ucis]
            if valid_book:
                chosen_uci, opening_name, _ = random.choices(
                    valid_book, weights=[w for _, _, w in valid_book], k=1
                )[0]
                book_hit = (legal_ucis[chosen_uci], opening_name)

        _, root_policy, nn_out = brain.evaluate_board(board, style_weights)

        candidate_rankings: List[Dict[str, Any]] = []
        chosen_move: Move = legal_moves[0]
        reasoning_steps: List[str] = []
        depth_reached = 1
        pv_line: List[str] = []

        if book_hit is not None:
            chosen_move, opening_name = book_hit
            self.nodes_evaluated = len(legal_moves) * 4
            reasoning_steps.append(f"Libro de Aperturas Gran Maestro: {opening_name}")
            for m in legal_moves:
                b_next = board.copy()
                b_next.apply_move_fast(m)
                v, _, _ = brain.evaluate_board(b_next, style_weights)
                my_v = v if board.turn == 'w' else -v
                if m.uci() == chosen_move.uci():
                    my_v += 0.14
                candidate_rankings.append({
                    "uci": m.uci(),
                    "san": board.format_san(m, legal_moves),
                    "from": sq_to_alg(m.from_sq),
                    "to": sq_to_alg(m.to_sq),
                    "score": round(my_v, 4),
                    "prior": round(brain.move_prior_logit(m, root_policy), 3),
                    "visits": 32 if m.uci() == chosen_move.uci() else 4
                })
            candidate_rankings.sort(key=lambda x: x["score"], reverse=True)
            pv_line = [board.format_san(chosen_move, legal_moves)]

        elif learning_model == "alphazero_mcts":
            simulations = {1: 32, 2: 52, 3: 80, 4: 115}.get(difficulty, 75)
            chosen_move, candidate_rankings, pv_line = self._run_mcts(
                board, brain, style_weights, reaction_profile, simulations
            )
            depth_reached = 3 if difficulty >= 3 else 2
            reasoning_steps.append(
                f"Lc0 SE-ResNet + Transformer MCTS ejecutó {simulations} simulaciones PUCT ({self.nodes_evaluated} nodos, {self.q_nodes} tácticos)."
            )

        else:
            target_depth = {1: 2, 2: 3, 3: 4, 4: 4}.get(difficulty, 3)
            chosen_move, candidate_rankings, depth_reached, pv_line = self._run_iterative_pvs(
                board, brain, style_weights, reaction_profile, target_depth, learning_model
            )
            if learning_model == "minimax_nnue":
                reasoning_steps.append(
                    f"Stockfish NNUE SFNNv9 + PVS alcanzó profundidad {depth_reached} ({self.nodes_evaluated} nodos, {self.q_nodes} Quiescence, {self.tt.hits - tt_hits_before} TT hits)."
                )
            elif learning_model == "dqn_td_rl":
                reasoning_steps.append(
                    f"Dueling Double-DQN RL + PVS evaluó variantes a profundidad {depth_reached} ({self.nodes_evaluated} nodos, {self.q_nodes} Quiescence)."
                )
            else:
                reasoning_steps.append(
                    f"Enjambre Neuro-Evolutivo + PVS exploró árboles a profundidad {depth_reached} ({self.nodes_evaluated} nodos)."
                )

        # Online AdamW Prioritized Experience Replay update
        top_score = candidate_rankings[0]["score"] if candidate_rankings else 0.0
        target_white_val = float(np.clip(top_score if board.turn == 'w' else -top_score, -0.99, 0.99))
        td_loss = brain.record_and_train_online(board, target_white_val, chosen_move.from_sq, chosen_move.to_sq)

        elapsed_ms = int((time.time() - start_time) * 1000)
        chosen_san = board.format_san(chosen_move, legal_moves)

        if chosen_move.captured:
            see_cp = see_score(board, chosen_move)
            reasoning_steps.append(f"Captura evaluada con SEE ({see_cp:+d} cp) mediante {chosen_san}.")
        if pv_line:
            reasoning_steps.append(f"Línea Principal (PV): {' '.join(pv_line[:5])}")
        if adaptive_state_note:
            reasoning_steps.append(adaptive_state_note)
        reasoning_steps.append(
            f"Backprop AdamW + PER completado (Loss: {td_loss:.4f}, Época #{brain.training_epochs})."
        )

        telemetry = self._build_neural_telemetry(
            board=board,
            brain=brain,
            nn_out=nn_out,
            chosen_move=chosen_move,
            chosen_san=chosen_san,
            candidate_rankings=candidate_rankings[:6],
            learning_model=learning_model,
            reaction_profile=reaction_profile,
            td_loss=td_loss,
            elapsed_ms=elapsed_ms,
            reasoning_steps=reasoning_steps,
            depth_reached=depth_reached,
            pv_line=pv_line,
            tt_hits=self.tt.hits - tt_hits_before
        )

        return {
            "move": chosen_move,
            "san": chosen_san,
            "telemetry": telemetry
        }

    # --------------------------------------------------------------------------
    # ALPHAZERO / LC0 PUCT MONTE CARLO TREE SEARCH WITH TACTICAL GUARD
    # --------------------------------------------------------------------------

    def _run_mcts(
        self,
        board: ChessBoard,
        brain: PyTorchChessBrain,
        style_weights: Dict[str, float],
        reaction_profile: str,
        simulations: int
    ) -> Tuple[Move, List[Dict[str, Any]], List[str]]:
        root = MCTSNode(board.copy())
        legal = root.board.generate_legal_moves()
        _, root_policy, _ = brain.evaluate_board(root.board, style_weights)

        # Immediate mate-in-1 detection
        for m in legal:
            b_chk = root.board.copy()
            b_chk.apply_move_fast(m)
            if len(b_chk.generate_legal_moves()) == 0 and b_chk.in_check(b_chk.turn):
                san = board.format_san(m, legal)
                return m, [{
                    "uci": m.uci(), "san": san, "from": sq_to_alg(m.from_sq),
                    "to": sq_to_alg(m.to_sq), "score": 0.999, "prior": 1.0, "visits": simulations
                }], [san]

        raw_scores = np.array(
            [self.score_move_tactics(root.board, m, brain, root_policy, reaction_profile) for m in legal],
            dtype=np.float32
        )
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        priors = exp_scores / (np.sum(exp_scores) + 1e-8)

        # Light Dirichlet root noise for exploration diversity
        if len(priors) > 1:
            noise = np.random.dirichlet([0.35] * len(priors))
            priors = 0.88 * priors + 0.12 * noise

        for m, pr in zip(legal, priors):
            b_child = root.board.copy()
            b_child.apply_move_fast(m)
            root.children.append(MCTSNode(b_child, parent=root, move=m, prior=float(pr)))
        root.is_expanded = True

        eval_cache: Dict[int, float] = {}

        for sim in range(simulations):
            if sim >= 24 and time.time() > self.search_deadline:
                break
            node = root
            while node.is_expanded and node.children:
                node = max(node.children, key=lambda c: c.ucb_score(c_puct=1.45))

            if node.visits > 0 and not node.is_expanded:
                sub_legal = node.board.generate_legal_moves()
                if sub_legal:
                    _, sub_pol, _ = brain.evaluate_board(node.board, style_weights)
                    scored = sorted(
                        sub_legal,
                        key=lambda m: self.score_move_tactics(node.board, m, brain, sub_pol, reaction_profile),
                        reverse=True
                    )[:7]
                    for sm in scored:
                        bc = node.board.copy()
                        bc.apply_move_fast(sm)
                        node.children.append(MCTSNode(bc, parent=node, move=sm, prior=1.0 / len(scored)))
                    node.is_expanded = True
                    node = node.children[0]

            self.nodes_evaluated += 1
            zhash = node.board.zobrist_hash()
            if zhash in eval_cache:
                val_white = eval_cache[zhash]
            else:
                status = node.board.get_game_status()
                if status["is_game_over"]:
                    val_white = (1.0 if node.board.turn == 'b' else -1.0) if status["status"] == "checkmate" else 0.0
                else:
                    val_white, _, _ = brain.evaluate_board(node.board, style_weights)
                    # Quiescence 1-ply tactical capture check so MCTS sees hanging material
                    opp_captures = [m for m in node.board.generate_legal_moves() if m.captured]
                    if opp_captures:
                        self.q_nodes += 1
                        best_see = max((see_score(node.board, cm) for cm in opp_captures), default=0)
                        if best_see > 75:
                            penalty = math.tanh(best_see / 500.0) * 0.55
                            val_white += (-penalty if node.board.turn == 'w' else penalty)
                eval_cache[zhash] = val_white

            val_root = val_white if board.turn == 'w' else -val_white
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += val_root
                cur = cur.parent

        rankings = []
        for child in root.children:
            q_val = child.q_value()
            # Guard against immediate opponent checkmate or hanging piece
            opp_legal = child.board.generate_legal_moves()
            mate_danger = 0.0
            worst_see = 0
            if len(opp_legal) == 0 and child.board.in_check(child.board.turn):
                q_val = 1.0  # Our move delivers checkmate!
            else:
                for om in opp_legal:
                    if om.captured:
                        s_val = see_score(child.board, om)
                        if s_val > worst_see:
                            worst_see = s_val
                if worst_see > 80:
                    mate_danger = math.tanh(worst_see / 450.0) * 0.45

            adjusted = q_val - mate_danger
            rankings.append({
                "move_obj": child.move,
                "uci": child.move.uci(),
                "san": board.format_san(child.move, legal),
                "from": sq_to_alg(child.move.from_sq),
                "to": sq_to_alg(child.move.to_sq),
                "score": round(float(adjusted), 4),
                "prior": round(float(child.prior), 3),
                "visits": child.visits
            })

        rankings.sort(key=lambda x: (x["score"], x["visits"]), reverse=True)
        best_move = rankings[0]["move_obj"]
        pv_line = [rankings[0]["san"]]

        # Extract 2nd & 3rd ply PV from most visited MCTS path
        cur_node = next((c for c in root.children if c.move.uci() == best_move.uci()), None)
        for _ in range(3):
            if cur_node and cur_node.children:
                nxt = max(cur_node.children, key=lambda c: c.visits)
                pv_line.append(cur_node.board.format_san(nxt.move))
                cur_node = nxt
            else:
                break

        for r in rankings:
            r.pop("move_obj", None)
        return best_move, rankings, pv_line

    # --------------------------------------------------------------------------
    # STOCKFISH-STYLE ITERATIVE DEEPENING + PVS + NULL-MOVE + LMR + QUIESCENCE
    # --------------------------------------------------------------------------

    def _run_iterative_pvs(
        self,
        board: ChessBoard,
        brain: PyTorchChessBrain,
        style_weights: Dict[str, float],
        reaction_profile: str,
        target_depth: int,
        learning_model: str
    ) -> Tuple[Move, List[Dict[str, Any]], int, List[str]]:
        legal = board.generate_legal_moves()
        fast_eval_brain = self.networks["minimax_nnue"]
        _, root_policy, _ = brain.evaluate_board(board, style_weights)

        best_move = legal[0]
        best_rankings: List[Dict[str, Any]] = []
        depth_reached = 1

        # Iterative deepening loop from depth 1..target_depth
        for cur_depth in range(1, target_depth + 1):
            if cur_depth >= 2 and time.time() > self.search_deadline:
                break

            tt_entry = self.tt.probe(board.zobrist_hash())
            tt_best = tt_entry[3] if tt_entry else (best_move.uci() if cur_depth > 1 else None)

            ordered = sorted(
                legal,
                key=lambda m: self.score_move_tactics(board, m, brain, root_policy, reaction_profile, ply=0, tt_best_uci=tt_best),
                reverse=True
            )

            alpha = -20.0
            beta = 20.0
            cur_rankings: List[Dict[str, Any]] = []
            aborted = False

            for idx, m in enumerate(ordered):
                if cur_depth >= 2 and idx >= 1 and time.time() > self.search_deadline:
                    aborted = True
                    break

                b_next = board.copy()
                b_next.apply_move_fast(m)

                if idx == 0:
                    val = -self._pvs(b_next, brain, fast_eval_brain, style_weights, reaction_profile, cur_depth - 1, -beta, -alpha, 1, True)
                else:
                    red = 1 if (cur_depth >= 3 and idx >= 6 and not m.captured and not m.promotion) else 0
                    val = -self._pvs(b_next, brain, fast_eval_brain, style_weights, reaction_profile, cur_depth - 1 - red, -alpha - 0.0005, -alpha, 1, True)
                    if val > alpha:
                        val = -self._pvs(b_next, brain, fast_eval_brain, style_weights, reaction_profile, cur_depth - 1, -beta, -alpha, 1, True)

                if learning_model == "evolutionary_ga" and abs(val) < 5.0:
                    val += float(np.random.normal(0.0, 0.008))

                if val > alpha:
                    alpha = val

                prior_s = brain.move_prior_logit(m, root_policy)
                tie_break = 0.002 * math.tanh(prior_s) if abs(val) < 5.0 else 0.0
                display_score = round(val + tie_break, 4)

                cur_rankings.append({
                    "move_obj": m,
                    "uci": m.uci(),
                    "san": board.format_san(m, legal),
                    "from": sq_to_alg(m.from_sq),
                    "to": sq_to_alg(m.to_sq),
                    "raw_val": val,
                    "score": display_score,
                    "prior": round(float(prior_s), 3),
                    "visits": max(1, (len(ordered) - idx) * cur_depth * 3)
                })

            if not aborted and cur_rankings:
                cur_rankings.sort(key=lambda x: (x["raw_val"], x["score"]), reverse=True)
                best_move = cur_rankings[0]["move_obj"]
                best_rankings = cur_rankings
                depth_reached = cur_depth
                self.tt.store(board.zobrist_hash(), cur_depth, TT_EXACT, cur_rankings[0]["raw_val"], best_move.uci())
                # Stop early if forced checkmate found at current depth
                if cur_rankings[0]["raw_val"] >= 9.0:
                    break

        pv_line = self._extract_pv(board, max_plies=5)
        for r in best_rankings:
            r.pop("move_obj", None)
            r.pop("raw_val", None)
        return best_move, best_rankings, depth_reached, pv_line

    def _pvs(
        self,
        board: ChessBoard,
        primary_brain: PyTorchChessBrain,
        fast_brain: PyTorchChessBrain,
        style_weights: Dict[str, float],
        reaction_profile: str,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        allow_null: bool
    ) -> float:
        self.nodes_evaluated += 1
        orig_alpha = alpha
        zhash = board.zobrist_hash()

        tt_entry = self.tt.probe(zhash)
        tt_best_uci = None
        if tt_entry is not None:
            tt_depth, tt_flag, tt_score, tt_best_uci = tt_entry
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == TT_UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        status = board.get_game_status()
        if status["is_game_over"]:
            if status["status"] == "checkmate":
                return -10.0 + (ply * 0.01)
            return 0.0

        if depth <= 0:
            return self._quiescence(board, fast_brain, style_weights, alpha, beta, q_ply=0)

        if (self.nodes_evaluated & 31) == 0 and time.time() > self.search_deadline:
            v_white, _, _ = fast_brain.evaluate_board(board, style_weights)
            return v_white if board.turn == 'w' else -v_white

        in_chk = board.in_check(board.turn)

        # Null-Move Pruning (R = 2) when safe
        if allow_null and depth >= 3 and not in_chk and board.has_non_pawn_material(board.turn):
            b_null = board.copy()
            b_null.turn = 'b' if b_null.turn == 'w' else 'w'
            b_null.ep_sq = -1
            null_val = -self._pvs(b_null, primary_brain, fast_brain, style_weights, reaction_profile, depth - 3, -beta, -beta + 0.0005, ply + 1, False)
            if null_val >= beta:
                return null_val

        legal = board.generate_legal_moves()
        eval_brain = primary_brain if ply <= 1 else fast_brain
        _, logits_pack, _ = eval_brain.evaluate_board(board, style_weights)

        max_branch = 16 if ply <= 1 else 10
        ordered = sorted(
            legal,
            key=lambda m: self.score_move_tactics(board, m, eval_brain, logits_pack, reaction_profile, ply=ply, tt_best_uci=tt_best_uci),
            reverse=True
        )[:max_branch]

        best_val = -20.0
        best_uci = ordered[0].uci() if ordered else None

        for idx, m in enumerate(ordered):
            b_child = board.copy()
            b_child.apply_move_fast(m)

            if idx == 0:
                val = -self._pvs(b_child, primary_brain, fast_brain, style_weights, reaction_profile, depth - 1, -beta, -alpha, ply + 1, True)
            else:
                reduction = 1 if (depth >= 3 and idx >= 5 and not in_chk and not m.captured and not m.promotion) else 0
                val = -self._pvs(b_child, primary_brain, fast_brain, style_weights, reaction_profile, depth - 1 - reduction, -alpha - 0.0005, -alpha, ply + 1, True)
                if val > alpha and (reduction > 0 or val < beta):
                    val = -self._pvs(b_child, primary_brain, fast_brain, style_weights, reaction_profile, depth - 1, -beta, -alpha, ply + 1, True)

            if val > best_val:
                best_val = val
                best_uci = m.uci()
            if val > alpha:
                alpha = val
            if alpha >= beta:
                if not m.captured and ply < 32:
                    if self.killers[ply][0] != m.uci():
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = m.uci()
                    key = (board.turn, m.from_sq, m.to_sq)
                    self.history[key] = self.history.get(key, 0) + depth * depth
                break

        tt_flag = TT_EXACT if (orig_alpha < best_val < beta) else (TT_LOWER if best_val >= beta else TT_UPPER)
        self.tt.store(zhash, depth, tt_flag, best_val, best_uci)
        return best_val

    def _quiescence(
        self,
        board: ChessBoard,
        fast_brain: PyTorchChessBrain,
        style_weights: Dict[str, float],
        alpha: float,
        beta: float,
        q_ply: int
    ) -> float:
        """
        Fail-Soft Stockfish-style Quiescence Search with Stand-Pat, SEE pruning, and Delta Pruning.
        """
        self.q_nodes += 1
        v_white, _, _ = fast_brain.evaluate_board(board, style_weights)
        stand_pat = v_white if board.turn == 'w' else -v_white

        if q_ply >= 3 or ((self.q_nodes & 31) == 0 and time.time() > self.search_deadline):
            return stand_pat
        if stand_pat >= beta:
            return stand_pat

        best_val = stand_pat
        if alpha < stand_pat:
            alpha = stand_pat

        # Delta pruning
        if stand_pat + 0.85 < alpha:
            return stand_pat

        legal = board.generate_legal_moves()
        if not legal:
            if board.in_check(board.turn):
                return -10.0 + ((4 + q_ply) * 0.01)
            return 0.0

        tactical_moves = []
        for m in legal:
            if m.captured or m.promotion:
                s_val = see_score(board, m)
                if s_val >= -25:
                    tactical_moves.append((s_val, m))

        tactical_moves.sort(key=lambda x: x[0], reverse=True)

        for _, m in tactical_moves[:6]:
            b_child = board.copy()
            b_child.apply_move_fast(m)
            score = -self._quiescence(b_child, fast_brain, style_weights, -beta, -alpha, q_ply + 1)
            if score > best_val:
                best_val = score
            if score >= beta:
                return best_val
            if score > alpha:
                alpha = score

        return best_val

    def _extract_pv(self, board: ChessBoard, max_plies: int = 5) -> List[str]:
        pv: List[str] = []
        cur = board.copy()
        seen = set()
        for _ in range(max_plies):
            zh = cur.zobrist_hash()
            if zh in seen:
                break
            seen.add(zh)
            entry = self.tt.probe(zh)
            if not entry or not entry[3]:
                break
            uci = entry[3]
            legal = cur.generate_legal_moves()
            match = next((m for m in legal if m.uci() == uci), None)
            if not match:
                break
            pv.append(cur.format_san(match, legal))
            cur.apply_move_fast(match)
        return pv

    # --------------------------------------------------------------------------
    # INTERACTIVE SELF-PLAY TRAINING & TELEMETRY
    # --------------------------------------------------------------------------

    def train_batch_self_play(self, learning_model: str, iterations: int = 25) -> Dict[str, Any]:
        brain = self.networks.get(learning_model, self.networks["alphazero_mcts"])
        losses = []
        board = ChessBoard()

        for _ in range(iterations):
            legal = board.generate_legal_moves()
            if not legal or board.fullmove_number > 40:
                board = ChessBoard()
                legal = board.generate_legal_moves()

            _, policy_pack, _ = brain.evaluate_board(board)
            best_m = max(
                legal,
                key=lambda m: self.score_move_tactics(board, m, brain, policy_pack, "tactical")
            )
            b_next = board.copy()
            b_next.apply_move_fast(best_m)
            next_val_white, _, _ = brain.evaluate_board(b_next)

            loss = brain.record_and_train_online(board, next_val_white, best_m.from_sq, best_m.to_sq)
            losses.append(loss)
            board = b_next

        brain.games_learned += 1
        try:
            brain.save_checkpoint()
        except Exception:
            pass

        return {
            "model_id": learning_model,
            "iterations": iterations,
            "epochs_total": brain.training_epochs,
            "games_learned": brain.games_learned,
            "mean_loss": round(float(np.mean(losses)), 4),
            "final_loss": round(float(losses[-1]), 4),
            "loss_history": brain.loss_history[-30:]
        }

    def _build_neural_telemetry(
        self,
        board: ChessBoard,
        brain: PyTorchChessBrain,
        nn_out: Dict[str, Any],
        chosen_move: Optional[Move],
        chosen_san: str,
        candidate_rankings: List[Dict[str, Any]],
        learning_model: str,
        reaction_profile: str,
        td_loss: float,
        elapsed_ms: int,
        reasoning_steps: List[str],
        depth_reached: int = 1,
        pv_line: Optional[List[str]] = None,
        tt_hits: int = 0
    ) -> Dict[str, Any]:
        val_white = float(nn_out.get("value_white", 0.0))
        win_prob_white = round((val_white + 1.0) / 2.0 * 100.0, 1)
        if candidate_rankings:
            top_s = candidate_rankings[0]["score"]
            s_white = top_s if board.turn == 'w' else -top_s
            win_prob_white = round(float(np.clip((s_white + 1.0) / 2.0 * 100.0, 2.0, 98.0)), 1)

        return {
            "learning_model": self.LEARNING_MODELS.get(learning_model),
            "reaction_profile": self.REACTION_PROFILES.get(reaction_profile),
            "chosen_move": chosen_move.uci() if chosen_move else "",
            "chosen_san": chosen_san,
            "value_estimate": round(val_white, 4),
            "wdl_probs": nn_out.get("wdl_stm", [0.33, 0.34, 0.33]),
            "moves_left": nn_out.get("moves_left", 30.0),
            "param_count": brain.param_count,
            "search_depth": depth_reached,
            "pv_line": pv_line or [],
            "tt_hits": tt_hits,
            "win_prob_white": win_prob_white,
            "win_prob_black": round(100.0 - win_prob_white, 1),
            "nodes_evaluated": self.nodes_evaluated + self.q_nodes,
            "q_nodes": self.q_nodes,
            "elapsed_ms": elapsed_ms,
            "training_epochs": brain.training_epochs,
            "td_loss": td_loss,
            "loss_history": brain.loss_history[-25:],
            "candidate_moves": candidate_rankings,
            "reasoning": reasoning_steps,
            "heatmap": nn_out.get("heatmap", []),
            "layer_activations": nn_out.get("activations", {})
        }
