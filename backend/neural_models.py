"""
NeuroChess Deep PyTorch Neural Architectures
============================================
Implements four distinct, reverse-engineered modern chess neural networks in PyTorch:
1. StockfishNNUE_SFNNv9     : Dual-perspective HalfKAv2_hm sparse feature transformer (24,576 features),
                              PSQT phase-bucketed bypass shortcut, SCReLU (Squared Clipped ReLU),
                              and 8 phase-specialized Layer Stacks with WDL + Centipawn evaluation.
2. Lc0_SEResNet_Transformer : Leela Chess Zero / AlphaZero hybrid 20-plane canonical CNN stem,
                              Squeeze-and-Excitation Residual Blocks (SE-ResNet),
                              Board-Wide 64-token Multi-Head Self-Attention Transformer (BT4 style),
                              4096-move Query-Key Attention Policy Head + WDL & Moves-Left Head.
3. DeepQ_ActorCritic_RL     : Dual-stream Spatial+Strategic Dueling Double-DQN / Advantage Actor-Critic
                              with Prioritized Experience Replay (PER) & online AdamW TD(lambda) learning.
4. NeuroEvolutionaryCommittee: Ensemble of mutated PyTorch Residual Policy-Value Genomes with
                              tournament selection, uniform crossover, and adaptive Gaussian mutation.
"""

import os
import math
import random
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Limit intra-op threads cleanly for Kubernetes container CPU quotas
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))

PIECE_TO_PLANE_W = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
PIECE_TO_PLANE_B = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
}

MAT_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 0
}

# Positional Piece-Square Tables (White's perspective: rank 8 down to rank 1)
NN_PST = {
    'P': [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 27, 27, 10,  5,  5,
         0,  0,  0, 22, 22,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    'N': [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 12, 16, 16, 12,  0,-30,
        -30,  5, 16, 22, 22, 16,  5,-30,
        -30,  0, 16, 22, 22, 16,  0,-30,
        -30,  5, 12, 16, 16, 12,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    'B': [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  8, 12, 12,  8,  0,-10,
        -10,  5,  8, 14, 14,  8,  5,-10,
        -10,  0, 12, 14, 14, 12,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  8,  0,  0,  0,  0,  8,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    'R': [
          0,  0,  0,  0,  0,  0,  0,  0,
          8, 14, 14, 14, 14, 14, 14,  8,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  6,  6,  0,  0,  0
    ],
    'Q': [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  6,  6,  5,  0, -5,
          0,  0,  5,  6,  6,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    'K': [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

_KNIGHT_DIRS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_KING_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def compute_attack_maps(board) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-pass tactical attack map generator for White and Black.
    Returns two (8, 8) float32 arrays representing square control intensity.
    """
    atk_w = np.zeros((8, 8), dtype=np.float32)
    atk_b = np.zeros((8, 8), dtype=np.float32)
    squares = board.board

    for sq, p in enumerate(squares):
        if p is None:
            continue
        r, c = divmod(sq, 8)
        is_w = p.isupper()
        atk = atk_w if is_w else atk_b
        pt = p.upper()

        if pt == 'P':
            dr = -1 if is_w else 1
            nr = r + dr
            if 0 <= nr < 8:
                if c > 0:
                    atk[nr, c - 1] += 0.5
                if c < 7:
                    atk[nr, c + 1] += 0.5
        elif pt == 'N':
            for dr, dc in _KNIGHT_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    atk[nr, nc] += 0.35
        elif pt == 'K':
            for dr, dc in _KING_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    atk[nr, nc] += 0.25
        else:
            dirs = _ORTHO_DIRS if pt == 'R' else (_DIAG_DIRS if pt == 'B' else _ORTHO_DIRS + _DIAG_DIRS)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    atk[nr, nc] += 0.30
                    if squares[nr * 8 + nc] is not None:
                        break
                    nr += dr
                    nc += dc

    return np.clip(atk_w, 0.0, 1.0), np.clip(atk_b, 0.0, 1.0)


def board_to_spatial_planes(board) -> np.ndarray:
    """
    Canonical 20-plane (20, 8, 8) tensor representation from side-to-move's perspective (Lc0/AlphaZero style).
    If Black to move, board ranks are vertically mirrored (r -> 7 - r) and White/Black pieces swapped.
    """
    planes = np.zeros((20, 8, 8), dtype=np.float32)
    is_white_turn = (board.turn == 'w')
    p_map = PIECE_TO_PLANE_W if is_white_turn else PIECE_TO_PLANE_B
    v_flip = 0 if is_white_turn else 56

    for sq, p in enumerate(board.board):
        if p is not None:
            csq = sq ^ v_flip
            r, c = divmod(csq, 8)
            planes[p_map[p], r, c] = 1.0

    # Castling rights in canonical perspective
    cast = board.castling or ''
    if is_white_turn:
        if 'K' in cast: planes[12].fill(1.0)
        if 'Q' in cast: planes[13].fill(1.0)
        if 'k' in cast: planes[14].fill(1.0)
        if 'q' in cast: planes[15].fill(1.0)
    else:
        if 'k' in cast: planes[12].fill(1.0)
        if 'q' in cast: planes[13].fill(1.0)
        if 'K' in cast: planes[14].fill(1.0)
        if 'Q' in cast: planes[15].fill(1.0)

    if board.ep_sq != -1:
        ep_c = board.ep_sq ^ v_flip
        er, ec = divmod(ep_c, 8)
        planes[16, er, ec] = 1.0

    planes[17].fill(min(board.halfmove_clock / 100.0, 1.0))

    atk_w, atk_b = compute_attack_maps(board)
    if is_white_turn:
        planes[18] = atk_w
        planes[19] = atk_b
    else:
        planes[18] = atk_b[::-1, :].copy()
        planes[19] = atk_w[::-1, :].copy()

    return planes


def extract_halfka_hm_indices(board) -> Tuple[List[int], List[int], int]:
    """
    Stockfish SFNNv9 HalfKAv2_hm Sparse Feature Extractor:
    - 32 horizontally mirrored king buckets (files a-d canonicalized via XOR 7 if king on e-h)
    - 12 piece types x 64 squares per king bucket => 32 * 12 * 64 = 24,576 total sparse indices.
    Returns (us_indices, them_indices, layer_stack_bucket_0_to_7).
    """
    wk_sq = board.find_king('w')
    bk_sq = board.find_king('b')
    if wk_sq < 0:
        wk_sq = 60
    if bk_sq < 0:
        bk_sq = 4

    wk_hm = 7 if (wk_sq % 8) >= 4 else 0
    wk_c = wk_sq ^ wk_hm
    w_bucket = (wk_c // 8) * 4 + (wk_c % 8)  # 0..31

    bk_v = bk_sq ^ 56
    bk_hm = 7 if (bk_v % 8) >= 4 else 0
    bk_c = bk_v ^ bk_hm
    b_bucket = (bk_c // 8) * 4 + (bk_c % 8)  # 0..31

    w_feats: List[int] = []
    b_feats: List[int] = []
    w_base = w_bucket * 768
    b_base = b_bucket * 768

    for sq, p in enumerate(board.board):
        if p is None:
            continue
        w_feats.append(w_base + PIECE_TO_PLANE_W[p] * 64 + (sq ^ wk_hm))
        b_feats.append(b_base + PIECE_TO_PLANE_B[p] * 64 + ((sq ^ 56) ^ bk_hm))

    if not w_feats:
        w_feats = [0]
    if not b_feats:
        b_feats = [0]

    piece_count = len(w_feats)
    stack_bucket = min(7, max(0, (piece_count - 2) // 4))

    if board.turn == 'w':
        return w_feats, b_feats, stack_bucket
    return b_feats, w_feats, stack_bucket


def extract_tactical_vector(board) -> np.ndarray:
    """
    Canonical 160-dimensional rich tactical & positional vector from side-to-move's perspective:
    - [0:12]   Normalized piece counts (Us 0..5, Them 6..11)
    - [12:24]  Piece-Square Table positional scores per piece type
    - [24:40]  Center 4x4 spatial occupation & control
    - [40:64]  Pawn structure: doubled, isolated, and passed pawn metrics across 8 files
    - [64:80]  King safety (pawn shield, castling status), bishop pair, open files, phase
    - [80:96]  Tactical attack pressure differential across central/king zones
    - [96:160] 64-square canonical signed piece value presence map
    """
    feats = np.zeros(160, dtype=np.float32)
    is_white_turn = (board.turn == 'w')
    p_map = PIECE_TO_PLANE_W if is_white_turn else PIECE_TO_PLANE_B
    v_flip = 0 if is_white_turn else 56

    us_pawns_file = [0] * 8
    them_pawns_file = [0] * 8
    us_pawn_ranks = [[] for _ in range(8)]
    them_pawn_ranks = [[] for _ in range(8)]
    us_rooks_file = []
    them_rooks_file = []
    us_bishops = 0
    them_bishops = 0
    us_king_sq = 60
    them_king_sq = 4
    total_non_king_mat = 0

    for sq, p in enumerate(board.board):
        if p is None:
            continue
        csq = sq ^ v_flip
        r, c = divmod(csq, 8)
        idx = p_map[p]
        is_us = idx < 6
        pt = p.upper()

        feats[idx] += 1.0 / 8.0
        pst_sq = csq if is_us else (csq ^ 56)
        feats[12 + idx] += NN_PST[pt][pst_sq] / 50.0

        if 2 <= r <= 5 and 2 <= c <= 5:
            bonus = 0.28 if (3 <= r <= 4 and 3 <= c <= 4) else 0.12
            feats[24 + (r - 2) * 4 + (c - 2)] += bonus if is_us else -bonus

        val_norm = MAT_VALUES[p] / 900.0
        total_non_king_mat += MAT_VALUES[p]
        feats[96 + csq] = val_norm if is_us else -val_norm

        if pt == 'P':
            if is_us:
                us_pawns_file[c] += 1
                us_pawn_ranks[c].append(r)
            else:
                them_pawns_file[c] += 1
                them_pawn_ranks[c].append(r)
        elif pt == 'B':
            if is_us:
                us_bishops += 1
            else:
                them_bishops += 1
        elif pt == 'R':
            if is_us:
                us_rooks_file.append(c)
            else:
                them_rooks_file.append(c)
        elif pt == 'K':
            if is_us:
                us_king_sq = csq
            else:
                them_king_sq = csq

    # Pawn structure (doubled, isolated, passed)
    for f in range(8):
        if us_pawns_file[f] > 1:
            feats[40 + f] = -0.25 * (us_pawns_file[f] - 1)
        if them_pawns_file[f] > 1:
            feats[40 + f] += 0.25 * (them_pawns_file[f] - 1)

        adj_us = (us_pawns_file[f - 1] if f > 0 else 0) + (us_pawns_file[f + 1] if f < 7 else 0)
        adj_them = (them_pawns_file[f - 1] if f > 0 else 0) + (them_pawns_file[f + 1] if f < 7 else 0)
        if us_pawns_file[f] > 0 and adj_us == 0:
            feats[48 + f] -= 0.25
        if them_pawns_file[f] > 0 and adj_them == 0:
            feats[48 + f] += 0.25

        # Passed pawns
        for pr in us_pawn_ranks[f]:
            blocked = any(
                tr < pr
                for af in range(max(0, f - 1), min(8, f + 2))
                for tr in them_pawn_ranks[af]
            )
            if not blocked:
                feats[56 + f] += (7 - pr) * 0.12
        for tr in them_pawn_ranks[f]:
            blocked = any(
                pr > tr
                for af in range(max(0, f - 1), min(8, f + 2))
                for pr in us_pawn_ranks[af]
            )
            if not blocked:
                feats[56 + f] -= tr * 0.12

    # Bishop pair & Rooks on open/semi-open files
    feats[64] = 0.35 if us_bishops >= 2 else 0.0
    feats[65] = -0.35 if them_bishops >= 2 else 0.0
    for rf in us_rooks_file:
        if us_pawns_file[rf] == 0:
            feats[66] += 0.28 if them_pawns_file[rf] == 0 else 0.15
    for rf in them_rooks_file:
        if them_pawns_file[rf] == 0:
            feats[67] -= 0.28 if us_pawns_file[rf] == 0 else 0.15

    # Castling rights & Game Phase
    cast = board.castling or ''
    if is_white_turn:
        feats[68] = 0.4 if 'K' in cast else 0.0
        feats[69] = 0.4 if 'Q' in cast else 0.0
        feats[70] = -0.4 if 'k' in cast else 0.0
        feats[71] = -0.4 if 'q' in cast else 0.0
    else:
        feats[68] = 0.4 if 'k' in cast else 0.0
        feats[69] = 0.4 if 'q' in cast else 0.0
        feats[70] = -0.4 if 'K' in cast else 0.0
        feats[71] = -0.4 if 'Q' in cast else 0.0

    phase = min(total_non_king_mat / 6200.0, 1.0)
    feats[72] = phase
    feats[73] = 1.0 - phase
    feats[74] = min(board.fullmove_number / 60.0, 1.0)

    # King pawn shield
    ukr, ukc = divmod(us_king_sq, 8)
    tkr, tkc = divmod(them_king_sq, 8)
    shield_us = sum(
        0.18 for dc in (-1, 0, 1)
        if 0 <= ukc + dc < 8 and ukr > 0 and (ukr - 1) in us_pawn_ranks[ukc + dc]
    )
    shield_them = sum(
        0.18 for dc in (-1, 0, 1)
        if 0 <= tkc + dc < 8 and tkr < 7 and (tkr + 1) in them_pawn_ranks[tkc + dc]
    )
    feats[75] = shield_us
    feats[76] = -shield_them

    # Attack differential in central 4x4 zone
    atk_w, atk_b = compute_attack_maps(board)
    atk_us = atk_w if is_white_turn else atk_b[::-1, :]
    atk_them = atk_b if is_white_turn else atk_w[::-1, :]
    diff_center = (atk_us[2:6, 2:6] - atk_them[2:6, 2:6]).reshape(16)
    feats[80:96] = diff_center

    return feats


# ==============================================================================
# PYTORCH BUILDING BLOCKS: SCReLU, Squeeze-and-Excitation, Residual Blocks
# ==============================================================================

class SCReLU(nn.Module):
    """Stockfish SFNNv9 Squared Clipped ReLU: clamp(x, 0, 1)^2."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0).pow(2)


class SqueezeExcitation(nn.Module):
    """Leela Chess Zero (Lc0) Channel Squeeze-and-Excitation Block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels * 2)
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 8, 8)
        pooled = x.mean(dim=(2, 3))  # Global Avg Pool (B, C)
        out = self.fc(pooled)
        scale, bias = out[:, :self.channels], out[:, self.channels:]
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class SEResBlock(nn.Module):
    """Lc0 SE-ResNet Convolutional Residual Block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)
        self.se = SqueezeExcitation(channels, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + x)


class NNUELayerStack(nn.Module):
    """Stockfish Phase-Specialized Layer Stack (512 -> 32 -> 32 -> WDL + Eval)."""
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.wdl_head = nn.Linear(32, 3)
        self.scalar_head = nn.Linear(32, 1)
        self.act = SCReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        wdl = self.wdl_head(h2)
        scalar = self.scalar_head(h2).squeeze(-1)
        return wdl, scalar, h2


# ==============================================================================
# MODEL 1: STOCKFISH NNUE SFNNv9 (HalfKAv2_hm + PSQT + SCReLU + Layer Stacks)
# ==============================================================================

class StockfishNNUE_SFNNv9(nn.Module):
    """
    Reverse-Engineered Stockfish SFNNv9 Architecture in PyTorch:
    - HalfKAv2_hm Sparse Feature Transformer (24,576 -> 256 per king perspective)
    - 8-Bucket Direct PSQT Shortcut (24,576 -> 8)
    - Squared Clipped ReLU (SCReLU)
    - 8 Phase-Specialized Layer Stacks (512 -> 32 -> 32 -> WDL + Scalar)
    - Auxiliary Move Ordering Policy Head (512 -> 128)
    """
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(24576, 256, mode='sum')
        self.psqt = nn.EmbeddingBag(24576, 8, mode='sum')
        self.ft_bias = nn.Parameter(torch.zeros(256))
        self.act = SCReLU()

        self.stacks = nn.ModuleList([NNUELayerStack(512) for _ in range(8)])
        self.policy_head = nn.Sequential(
            nn.Linear(512, 128),
            SCReLU(),
            nn.Linear(128, 128)
        )

        # Careful sparse init scaled for <=32 active features
        nn.init.normal_(self.ft.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.psqt.weight, mean=0.0, std=0.02)

    def forward_batch(
        self,
        us_idx: torch.Tensor,
        us_offsets: torch.Tensor,
        them_idx: torch.Tensor,
        them_offsets: torch.Tensor,
        buckets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        acc_us = self.act(self.ft(us_idx, us_offsets) + self.ft_bias)
        acc_them = self.act(self.ft(them_idx, them_offsets) + self.ft_bias)
        dual_acc = torch.cat([acc_us, acc_them], dim=-1)  # (B, 512)

        psqt_us = self.psqt(us_idx, us_offsets)
        psqt_them = self.psqt(them_idx, them_offsets)
        b_idx = torch.arange(buckets.shape[0], device=buckets.device)
        psqt_diff = 0.5 * (psqt_us[b_idx, buckets] - psqt_them[b_idx, buckets])

        # Route each sample through its phase-specialized Layer Stack
        wdl_out = torch.zeros((buckets.shape[0], 3), device=buckets.device)
        scalar_out = torch.zeros((buckets.shape[0],), device=buckets.device)
        h_vis = torch.zeros((buckets.shape[0], 32), device=buckets.device)

        for s_id in range(8):
            mask = (buckets == s_id)
            if mask.any():
                w_s, sc_s, h_s = self.stacks[s_id](dual_acc[mask])
                wdl_out[mask] = w_s
                scalar_out[mask] = sc_s
                h_vis[mask] = h_s

        wdl_probs = F.softmax(wdl_out, dim=-1)
        wdl_value = wdl_probs[:, 0] - wdl_probs[:, 2]
        blended_val = torch.tanh(0.65 * (scalar_out + psqt_diff) + 1.25 * wdl_value)

        pol_logits = self.policy_head(dual_acc)  # (B, 128)
        return {
            "wdl_logits": wdl_out,
            "wdl_probs": wdl_probs,
            "value": blended_val,
            "pol_logits": pol_logits,
            "acc_us": acc_us,
            "acc_them": acc_them,
            "h_stack": h_vis
        }


# ==============================================================================
# MODEL 2: LEELA CHESS ZERO (SE-ResNet + BT4 Transformer + 4096 Attention Policy)
# ==============================================================================

class Lc0_SEResNet_Transformer(nn.Module):
    """
    Leela Chess Zero (Lc0) / AlphaZero Hybrid Architecture in PyTorch:
    - Stem Conv2d (20 -> 64) + GroupNorm + GELU
    - 3 x Squeeze-and-Excitation Residual Blocks (SE-ResNet)
    - 2-Layer 64-token Multi-Head Self-Attention Transformer (BT4 style)
    - Query-Key Attention Policy Head outputting (B, 64, 64) move pair logits + Promotion logits (B, 4)
    - WDL Value Head (Win/Draw/Loss) + Moves-Left Head (MLH)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.res1 = SEResBlock(64)
        self.res2 = SEResBlock(64)
        self.res3 = SEResBlock(64)

        self.pos_emb = nn.Parameter(torch.randn(1, 64, 64) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)

        # Lc0 Query-Key Attention Policy Head (64x64 move matrix)
        self.q_proj = nn.Linear(64, 32)
        self.k_proj = nn.Linear(64, 32)
        self.prom_head = nn.Linear(64, 4)

        # WDL Value Head & Moves-Left Head (MLH)
        self.val_conv = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.GELU()
        )
        self.val_fc = nn.Sequential(
            nn.Linear(16 * 64, 128),
            nn.GELU()
        )
        self.wdl_out = nn.Linear(128, 3)
        self.mlh_out = nn.Linear(128, 1)

    def forward(self, planes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # planes: (B, 20, 8, 8)
        h0 = self.stem(planes)
        h1 = self.res1(h0)
        h2 = self.res2(h1)
        h3 = self.res3(h2)  # (B, 64, 8, 8)

        # Flatten spatial grid to 64 square tokens for Board-Wide Self-Attention
        tokens = h3.flatten(2).transpose(1, 2)  # (B, 64, 64)
        attn_tokens = self.transformer(tokens + self.pos_emb)  # (B, 64, 64)

        # Query-Key Attention Policy Head: (B, 64, 64)
        q = self.q_proj(attn_tokens)
        k = self.k_proj(attn_tokens)
        policy_matrix = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(32.0)

        # Global token pooling for promotion bias
        prom_logits = self.prom_head(attn_tokens.mean(dim=1))  # (B, 4)

        # Reshape back to spatial for WDL + MLH Value Head
        attn_spatial = attn_tokens.transpose(1, 2).reshape(-1, 64, 8, 8)
        v_feat = self.val_conv(attn_spatial).flatten(1)
        v_hidden = self.val_fc(v_feat)
        wdl_logits = self.wdl_out(v_hidden)
        wdl_probs = F.softmax(wdl_logits, dim=-1)
        value = wdl_probs[:, 0] - wdl_probs[:, 2]
        moves_left = F.softplus(self.mlh_out(v_hidden).squeeze(-1)) * 20.0

        return {
            "wdl_logits": wdl_logits,
            "wdl_probs": wdl_probs,
            "value": value,
            "moves_left": moves_left,
            "policy_matrix": policy_matrix,
            "prom_logits": prom_logits,
            "h1": h1.mean(dim=(2, 3)),
            "h2": attn_tokens.mean(dim=1),
            "v_hidden": v_hidden
        }


# ==============================================================================
# MODEL 3: DUAL-STREAM DUELING DQN + ADVANTAGE ACTOR-CRITIC (TD-Lambda RL)
# ==============================================================================

class DeepQ_ActorCritic_RL(nn.Module):
    """
    Deep Reinforcement Learning Architecture combining:
    - Spatial Conv Branch (20x8x8 -> 128) + Tactical Vector Branch (160 -> 128)
    - Residual Fusion Trunk (256 -> 256)
    - Dueling Q-Streams: State-Value V(s) [WDL] + Action-Advantage A(s, a) [from/to 128]
    """
    def __init__(self):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 4, 4)
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.vec_branch = nn.Sequential(
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )
        self.val_stream = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, 3)  # WDL
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128)  # 64 from + 64 to
        )

    def forward(self, planes: torch.Tensor, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        c_feat = self.conv_branch(planes)
        v_feat = self.vec_branch(vec)
        fused = torch.cat([c_feat, v_feat], dim=-1)
        trunk = F.gelu(self.fusion(fused) + fused)

        wdl_logits = self.val_stream(trunk)
        wdl_probs = F.softmax(wdl_logits, dim=-1)
        value = wdl_probs[:, 0] - wdl_probs[:, 2]

        raw_adv = self.adv_stream(trunk)
        adv_from = raw_adv[:, :64] - raw_adv[:, :64].mean(dim=-1, keepdim=True)
        adv_to = raw_adv[:, 64:] - raw_adv[:, 64:].mean(dim=-1, keepdim=True)
        q_policy = torch.cat([adv_from, adv_to], dim=-1)

        return {
            "wdl_logits": wdl_logits,
            "wdl_probs": wdl_probs,
            "value": value,
            "pol_logits": q_policy,
            "h1": c_feat,
            "h2": trunk[:, :64],
            "v_hidden": trunk[:, 64:128]
        }


# ==============================================================================
# MODEL 4: NEURO-EVOLUTIONARY COMMITTEE SWARM (4 Mutating Policy-Value Genomes)
# ==============================================================================

class GenomeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(160, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU()
        )
        self.wdl = nn.Linear(192, 3)
        self.pol = nn.Linear(192, 128)

    def forward(self, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(vec)
        return self.wdl(h), self.pol(h), h


class NeuroEvolutionaryCommittee(nn.Module):
    """
    Population of 4 PyTorch neural genomes evaluated via fitness-weighted consensus,
    updated via Tournament Selection, Crossover, and Synaptic Gaussian Mutation + Gradient Polish.
    """
    def __init__(self, population_size: int = 4):
        super().__init__()
        self.genomes = nn.ModuleList([GenomeNet() for _ in range(population_size)])
        self.register_buffer("fitness", torch.ones(population_size, dtype=torch.float32))

    def forward(self, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        weights = F.softmax(self.fitness, dim=0)
        wdl_sum = None
        pol_sum = None
        h_best = None
        best_idx = int(torch.argmax(self.fitness).item())

        for i, g in enumerate(self.genomes):
            w_i, p_i, h_i = g(vec)
            wt = weights[i]
            wdl_sum = w_i * wt if wdl_sum is None else wdl_sum + w_i * wt
            pol_sum = p_i * wt if pol_sum is None else pol_sum + p_i * wt
            if i == best_idx:
                h_best = h_i

        wdl_probs = F.softmax(wdl_sum, dim=-1)
        value = wdl_probs[:, 0] - wdl_probs[:, 2]
        return {
            "wdl_logits": wdl_sum,
            "wdl_probs": wdl_probs,
            "value": value,
            "pol_logits": pol_sum,
            "h1": vec[:, :64],
            "h2": h_best[:, :64],
            "v_hidden": h_best[:, 64:128]
        }

    def evolve_generation(self, mutation_sigma: float = 0.015):
        """Crossover and Gaussian mutation keeping the #1 elite genome intact."""
        order = torch.argsort(self.fitness, descending=True).tolist()
        elite_a = self.genomes[order[0]]
        elite_b = self.genomes[order[1]]
        weakest = self.genomes[order[-1]]

        with torch.no_grad():
            for p_w, p_a, p_b in zip(weakest.parameters(), elite_a.parameters(), elite_b.parameters()):
                mask = (torch.rand_like(p_a) > 0.5)
                child = torch.where(mask, p_a, p_b) + torch.randn_like(p_a) * mutation_sigma
                p_w.copy_(child)
            self.fitness[order[-1]] = 0.5 * (self.fitness[order[0]] + self.fitness[order[1]])


# ==============================================================================
# UNIFIED PYTORCH CHESS MODEL ADAPTER (Inference, Online RL, Checkpointing)
# ==============================================================================

class PyTorchChessBrain:
    """
    Unified interface wrapping each PyTorch model architecture with:
    - Fast CPU inference (returning canonical & White-perspective WDL, Value, Move Priors, Telemetry)
    - Prioritized Experience Replay (PER) + AdamW online gradient updates
    - Checkpoint save/load (.pt) for Docker/Kubernetes pre-trained readiness
    """
    def __init__(self, model_id: str, weights_dir: Optional[str] = None):
        self.model_id = model_id
        self.weights_dir = weights_dir or os.environ.get(
            "NEUROCHESS_WEIGHTS_DIR",
            os.path.join(os.path.dirname(__file__), "weights")
        )
        os.makedirs(self.weights_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.weights_dir, f"{model_id}.pt")

        if model_id == "minimax_nnue":
            self.net = StockfishNNUE_SFNNv9()
            lr = 1.5e-3
        elif model_id == "alphazero_mcts":
            self.net = Lc0_SEResNet_Transformer()
            lr = 1.0e-3
        elif model_id == "dqn_td_rl":
            self.net = DeepQ_ActorCritic_RL()
            lr = 1.2e-3
        else:
            self.net = NeuroEvolutionaryCommittee()
            lr = 1.2e-3

        self.param_count = sum(p.numel() for p in self.net.parameters())
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.training_epochs = 0
        self.games_learned = 0
        self.loss_history: List[float] = []
        self.replay_buffer: List[Tuple[Dict[str, Any], float, int, int, float]] = []

        self.load_checkpoint_if_exists()

    def load_checkpoint_if_exists(self) -> bool:
        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                self.net.load_state_dict(ckpt["state_dict"])
                self.training_epochs = int(ckpt.get("training_epochs", 0))
                self.games_learned = int(ckpt.get("games_learned", 0))
                self.loss_history = list(ckpt.get("loss_history", []))[-100:]
                return True
            except Exception:
                return False
        return False

    def save_checkpoint(self):
        torch.save({
            "model_id": self.model_id,
            "state_dict": self.net.state_dict(),
            "training_epochs": self.training_epochs,
            "games_learned": self.games_learned,
            "loss_history": self.loss_history[-100:]
        }, self.checkpoint_path)

    def _encode_single(self, board) -> Dict[str, Any]:
        planes = board_to_spatial_planes(board)
        vec = extract_tactical_vector(board)
        us_idx, them_idx, bucket = extract_halfka_hm_indices(board)
        return {
            "planes": planes,
            "vec": vec,
            "us_idx": us_idx,
            "them_idx": them_idx,
            "bucket": bucket
        }

    @torch.no_grad()
    def infer(self, board) -> Dict[str, Any]:
        """
        Runs PyTorch inference on `board`.
        Returns canonical evaluation (from side-to-move's perspective) AND White-perspective value,
        WDL probabilities [Win, Draw, Loss] for side-to-move, policy move scorer, heatmap, and activations.
        """
        self.net.eval()
        enc = self._encode_single(board)
        is_w = (board.turn == 'w')
        v_flip = 0 if is_w else 56

        if self.model_id == "minimax_nnue":
            us_t = torch.tensor(enc["us_idx"], dtype=torch.long)
            th_t = torch.tensor(enc["them_idx"], dtype=torch.long)
            off = torch.tensor([0], dtype=torch.long)
            bkt = torch.tensor([enc["bucket"]], dtype=torch.long)
            out = self.net.forward_batch(us_t, off, th_t, off, bkt)
            pol_128 = out["pol_logits"][0].numpy()
            pol_matrix = pol_128[:64, None] + pol_128[None, 64:]
            prom_4 = np.array([0.8, 0.1, 0.05, 0.05], dtype=np.float32)
            h1 = out["acc_us"][0, :16].numpy()
            h2 = out["acc_them"][0, :16].numpy()
            vh = out["h_stack"][0, :12].numpy()
            mlh = 30.0
        elif self.model_id == "alphazero_mcts":
            pl_t = torch.from_numpy(enc["planes"]).unsqueeze(0)
            out = self.net(pl_t)
            pol_matrix = out["policy_matrix"][0].numpy()
            prom_4 = F.softmax(out["prom_logits"][0], dim=-1).numpy()
            h1 = out["h1"][0, :16].numpy()
            h2 = out["h2"][0, :16].numpy()
            vh = out["v_hidden"][0, :12].numpy()
            mlh = float(out["moves_left"][0].item())
        else:
            pl_t = torch.from_numpy(enc["planes"]).unsqueeze(0)
            vec_t = torch.from_numpy(enc["vec"]).unsqueeze(0)
            out = self.net(pl_t, vec_t) if self.model_id == "dqn_td_rl" else self.net(vec_t)
            pol_128 = out["pol_logits"][0].numpy()
            pol_matrix = pol_128[:64, None] + pol_128[None, 64:]
            prom_4 = np.array([0.8, 0.1, 0.05, 0.05], dtype=np.float32)
            h1 = out["h1"][0, :16].numpy()
            h2 = out["h2"][0, :16].numpy()
            vh = out["v_hidden"][0, :12].numpy()
            mlh = 28.0

        val_stm = float(out["value"][0].item())
        val_white = val_stm if is_w else -val_stm
        wdl_stm = out["wdl_probs"][0].numpy().tolist()

        # Build 8x8 attention heatmap in White's visual board orientation
        target_heat = pol_matrix.max(axis=0).reshape(8, 8)
        if not is_w:
            target_heat = target_heat[::-1, :]
        h_min, h_max = float(target_heat.min()), float(target_heat.max())
        heatmap = ((target_heat - h_min) / (h_max - h_min + 1e-6)).round(3).tolist()

        return {
            "value_stm": val_stm,
            "value_white": val_white,
            "wdl_stm": [round(float(x), 3) for x in wdl_stm],
            "moves_left": round(mlh, 1),
            "pol_matrix": pol_matrix,
            "prom_4": prom_4,
            "v_flip": v_flip,
            "heatmap": heatmap,
            "activations": {
                "input_layer": np.tanh(enc["vec"][:16]).round(3).tolist(),
                "residual_block_1": np.tanh(h1).round(3).tolist(),
                "residual_block_2": np.tanh(h2).round(3).tolist(),
                "value_head": np.tanh(vh).round(3).tolist(),
                "output_value": round(val_white, 3)
            }
        }

    def evaluate_board(
        self,
        board,
        style_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """
        Blends PyTorch Neural Value with Classical Tapered Material+Positional anchor
        for ultra-stable search & tactical awareness. Returns (val_white, nn_policy_info, nn_telemetry).
        """
        nn_out = self.infer(board)
        w = style_weights or {"material": 1.0, "positional": 1.0, "center": 1.0, "aggression": 1.0, "neural_blend": 0.55}

        mat_score = 0.0
        pos_score = 0.0
        center_score = 0.0
        aggression_score = 0.0
        non_king_mat = 0

        for sq, p in enumerate(board.board):
            if p is None:
                continue
            is_w = p.isupper()
            pt = p.upper()
            mv = MAT_VALUES[p]
            non_king_mat += mv
            mat_score += mv if is_w else -mv
            pst_sq = sq if is_w else (sq ^ 56)
            pst_val = NN_PST[pt][pst_sq]
            pos_score += pst_val if is_w else -pst_val

            r, c = divmod(sq, 8)
            if 2 <= r <= 5 and 2 <= c <= 5:
                center_score += 20.0 if is_w else -20.0
            if is_w and r <= 2:
                aggression_score += 14.0
            elif (not is_w) and r >= 5:
                aggression_score -= 14.0

        # Endgame mop-up bonus: drive enemy king to the edge when ahead by >= 300cp
        if non_king_mat < 2400 and abs(mat_score) >= 300:
            wk_sq = board.find_king('w')
            bk_sq = board.find_king('b')
            if wk_sq >= 0 and bk_sq >= 0:
                wkr, wkc = divmod(wk_sq, 8)
                bkr, bkc = divmod(bk_sq, 8)
                king_dist = abs(wkr - bkr) + abs(wkc - bkc)
                if mat_score > 0:
                    enemy_edge = max(abs(3.5 - bkr), abs(3.5 - bkc))
                    pos_score += enemy_edge * 16.0 + (14 - king_dist) * 8.0
                else:
                    enemy_edge = max(abs(3.5 - wkr), abs(3.5 - wkc))
                    pos_score -= enemy_edge * 16.0 + (14 - king_dist) * 8.0

        classical_cp = (
            mat_score * w.get("material", 1.0)
            + pos_score * w.get("positional", 1.0)
            + center_score * w.get("center", 1.0)
            + aggression_score * w.get("aggression", 1.0)
        )
        classical_tanh = math.tanh(classical_cp / 620.0)

        nn_blend = w.get("neural_blend", 0.55)
        blended_white = (1.0 - nn_blend) * classical_tanh + nn_blend * nn_out["value_white"]

        policy_pack = {
            "pol_matrix": nn_out["pol_matrix"],
            "prom_4": nn_out["prom_4"],
            "v_flip": nn_out["v_flip"]
        }
        return blended_white, policy_pack, nn_out

    def move_prior_logit(self, move, policy_pack: Dict[str, Any]) -> float:
        v_flip = policy_pack["v_flip"]
        cf = move.from_sq ^ v_flip
        ct = move.to_sq ^ v_flip
        logit = float(policy_pack["pol_matrix"][cf, ct])
        if move.promotion:
            p_idx = {'q': 0, 'r': 1, 'b': 2, 'n': 3}.get(move.promotion.lower(), 0)
            logit += float(policy_pack["prom_4"][p_idx]) * 1.5
        return logit

    def train_batch(
        self,
        samples: List[Tuple[Dict[str, Any], float, int, int]]
    ) -> float:
        """
        Executes a full PyTorch AdamW backpropagation step over a mini-batch of positions.
        Each sample: (encoded_board, target_stm_value in [-1, 1], canon_from_sq, canon_to_sq).
        """
        if not samples:
            return 0.0

        self.net.train()
        self.optimizer.zero_grad()

        target_vals = torch.tensor([float(np.clip(s[1], -0.99, 0.99)) for s in samples], dtype=torch.float32)
        # Convert scalar target in [-1, 1] into smooth WDL soft labels [Win, Draw, Loss]
        p_win = torch.clamp((target_vals + 1.0) * 0.5, 0.02, 0.96)
        p_loss = torch.clamp((1.0 - target_vals) * 0.5, 0.02, 0.96)
        p_draw = torch.clamp(0.28 * (1.0 - torch.abs(target_vals)), 0.02, 0.45)
        wdl_targets = torch.stack([p_win, p_draw, p_loss], dim=-1)
        wdl_targets = wdl_targets / wdl_targets.sum(dim=-1, keepdim=True)

        from_targets = torch.tensor([s[2] for s in samples], dtype=torch.long)
        to_targets = torch.tensor([s[3] for s in samples], dtype=torch.long)
        flat_move_targets = from_targets * 64 + to_targets

        if self.model_id == "minimax_nnue":
            us_flat, us_off, th_flat, th_off, bkts = [], [], [], [], []
            for s in samples:
                enc = s[0]
                us_off.append(len(us_flat))
                us_flat.extend(enc["us_idx"])
                th_off.append(len(th_flat))
                th_flat.extend(enc["them_idx"])
                bkts.append(enc["bucket"])

            out = self.net.forward_batch(
                torch.tensor(us_flat, dtype=torch.long),
                torch.tensor(us_off, dtype=torch.long),
                torch.tensor(th_flat, dtype=torch.long),
                torch.tensor(th_off, dtype=torch.long),
                torch.tensor(bkts, dtype=torch.long)
            )
            pol = out["pol_logits"]
            loss_p = 0.5 * (F.cross_entropy(pol[:, :64], from_targets) + F.cross_entropy(pol[:, 64:], to_targets))

        elif self.model_id == "alphazero_mcts":
            planes_b = torch.from_numpy(np.stack([s[0]["planes"] for s in samples]))
            out = self.net(planes_b)
            flat_pol = out["policy_matrix"].reshape(-1, 4096)
            loss_p = F.cross_entropy(flat_pol, flat_move_targets)

        elif self.model_id == "dqn_td_rl":
            planes_b = torch.from_numpy(np.stack([s[0]["planes"] for s in samples]))
            vecs_b = torch.from_numpy(np.stack([s[0]["vec"] for s in samples]))
            out = self.net(planes_b, vecs_b)
            pol = out["pol_logits"]
            loss_p = 0.5 * (F.cross_entropy(pol[:, :64], from_targets) + F.cross_entropy(pol[:, 64:], to_targets))

        else:
            vecs_b = torch.from_numpy(np.stack([s[0]["vec"] for s in samples]))
            out = self.net(vecs_b)
            pol = out["pol_logits"]
            loss_p = 0.5 * (F.cross_entropy(pol[:, :64], from_targets) + F.cross_entropy(pol[:, 64:], to_targets))

        loss_v = F.mse_loss(out["value"], target_vals)
        loss_wdl = -(wdl_targets * F.log_softmax(out["wdl_logits"], dim=-1)).sum(dim=-1).mean()
        total_loss = loss_v + 0.65 * loss_wdl + 0.18 * loss_p

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = round(float(total_loss.item()), 4)
        self.training_epochs += 1
        self.loss_history.append(loss_val)
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]

        if self.model_id == "evolutionary_ga":
            with torch.no_grad():
                best_idx = int(torch.argmax(self.net.fitness).item())
                self.net.fitness[best_idx] += 0.05
            if self.training_epochs % 12 == 0:
                self.net.evolve_generation(mutation_sigma=0.012)

        return loss_val

    def record_and_train_online(self, board, target_white_val: float, from_sq: int, to_sq: int) -> float:
        """
        Stores position in Prioritized Experience Replay (PER) and performs an AdamW step
        blending the latest move with high-error historical samples.
        """
        enc = self._encode_single(board)
        v_flip = 0 if board.turn == 'w' else 56
        target_stm = target_white_val if board.turn == 'w' else -target_white_val
        cf, ct = from_sq ^ v_flip, to_sq ^ v_flip

        with torch.no_grad():
            pred = self.infer(board)["value_stm"]
        priority = abs(target_stm - pred) + 0.05

        self.replay_buffer.append((enc, target_stm, cf, ct, priority))
        if len(self.replay_buffer) > 512:
            self.replay_buffer.pop(0)

        batch = [(enc, target_stm, cf, ct)]
        if len(self.replay_buffer) >= 8:
            weights = [x[4] for x in self.replay_buffer[:-1]]
            sampled = random.choices(self.replay_buffer[:-1], weights=weights, k=min(7, len(weights)))
            for item in sampled:
                batch.append((item[0], item[1], item[2], item[3]))

        return self.train_batch(batch)
