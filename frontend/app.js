// NeuroChess Frontend Interactive Controller, SVG Piece Renderer, WebAudio Synth & Live Neural Canvas

const PIECE_SVGS = {
  P: `<svg class="piece-svg" viewBox="0 0 45 45"><path d="M 22.5,9 C 19.5,9 17,11.5 17,14.5 C 17,16.8 18.4,18.7 20.4,19.5 C 17.5,21 15.5,24 15.5,28.5 L 29.5,28.5 C 29.5,24 27.5,21 24.6,19.5 C 26.6,18.7 28,16.8 28,14.5 C 28,11.5 25.5,9 22.5,9 z M 12.5,32 L 32.5,32 L 34,36.5 L 11,36.5 L 12.5,32 z" fill="#ffffff" stroke="#1e293b" stroke-width="2" stroke-linecap="round"/></svg>`,
  N: `<svg class="piece-svg" viewBox="0 0 45 45"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" fill="#ffffff" stroke="#1e293b" stroke-width="2"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" fill="#ffffff" stroke="#1e293b" stroke-width="2" stroke-linejoin="round"/><circle cx="15.5" cy="14.5" r="1.5" fill="#1e293b"/></svg>`,
  B: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1e293b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.06 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><circle cx="22.5" cy="8" r="2.2"/></g></svg>`,
  R: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1e293b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z"/><path d="M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z"/><path d="M 14,29.5 L 14,16.5 L 31,16.5 L 31,29.5 L 14,29.5 z"/><path d="M 14,16.5 L 11,14 L 34,14 L 31,16.5 L 14,16.5 z"/><path d="M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/></g></svg>`,
  Q: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1e293b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/><circle cx="6" cy="12" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="22.5" cy="8" r="1.8"/><circle cx="31" cy="9" r="1.8"/><circle cx="39" cy="12" r="1.8"/></g></svg>`,
  K: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1e293b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M 22.5,11.63 L 22.5,6"/><path d="M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 19,16 9.5,13 6.5,19.5 C 3.5,25.5 11.5,29.5 11.5,29.5 L 11.5,37 z"/></g></svg>`,
  p: `<svg class="piece-svg" viewBox="0 0 45 45"><path d="M 22.5,9 C 19.5,9 17,11.5 17,14.5 C 17,16.8 18.4,18.7 20.4,19.5 C 17.5,21 15.5,24 15.5,28.5 L 29.5,28.5 C 29.5,24 27.5,21 24.6,19.5 C 26.6,18.7 28,16.8 28,14.5 C 28,11.5 25.5,9 22.5,9 z M 12.5,32 L 32.5,32 L 34,36.5 L 11,36.5 L 12.5,32 z" fill="#1e293b" stroke="#f8fafc" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  n: `<svg class="piece-svg" viewBox="0 0 45 45"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" fill="#1e293b" stroke="#f8fafc" stroke-width="1.8"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" fill="#1e293b" stroke="#f8fafc" stroke-width="1.8" stroke-linejoin="round"/><circle cx="15.5" cy="14.5" r="1.5" fill="#f8fafc"/></svg>`,
  b: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#1e293b" stroke="#f8fafc" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.06 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><circle cx="22.5" cy="8" r="2.2"/></g></svg>`,
  r: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#1e293b" stroke="#f8fafc" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z"/><path d="M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z"/><path d="M 14,29.5 L 14,16.5 L 31,16.5 L 31,29.5 L 14,29.5 z"/><path d="M 14,16.5 L 11,14 L 34,14 L 31,16.5 L 14,16.5 z"/><path d="M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/></g></svg>`,
  q: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#1e293b" stroke="#f8fafc" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/><circle cx="6" cy="12" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="22.5" cy="8" r="1.8"/><circle cx="31" cy="9" r="1.8"/><circle cx="39" cy="12" r="1.8"/></g></svg>`,
  k: `<svg class="piece-svg" viewBox="0 0 45 45"><g fill="#1e293b" stroke="#f8fafc" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M 22.5,11.63 L 22.5,6"/><path d="M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 19,16 9.5,13 6.5,19.5 C 3.5,25.5 11.5,29.5 11.5,29.5 L 11.5,37 z"/></g></svg>`
};

const UNICODE_PIECES = {
  P: '♙', N: '♘', B: '♗', R: '♖', Q: '♕', K: '♔',
  p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚'
};

const MODEL_DESCRIPTIONS = {
  alphazero_mcts: "Red Convolucional Squeeze-and-Excitation + Transformer Multi-Cabeza (estilo Leela Chess Zero BT4) combinada con búsqueda Monte Carlo PUCT y cabezal WDL/MLH.",
  minimax_nnue: "Ingeniería inversa de Stockfish SFNNv9 en PyTorch: acumulador dual HalfKAv2_hm (24,576), SCReLU, 8 pilas de fase, Transposition Table, SEE, PVS y Quiescence Search.",
  dqn_td_rl: "Agente Deep RL de doble flujo (espacial + táctico) con optimizador AdamW, muestreo por error TD prioritario (PER) y aprendizaje continuo en vivo.",
  evolutionary_ga: "Comité de 4 genomas neuronales PyTorch evaluados por consenso ponderado según aptitud (fitness), con cruce y mutación genética adaptativa."
};

const REACTION_DESCRIPTIONS = {
  tactical: "Prioriza combinaciones forzadas, horquillas, clavadas y ganancia material exacta.",
  aggressive: "Busca sacrificios tácticos, columnas abiertas y ataques directos al Rey enemigo.",
  positional: "Domina casillas centrales, mejora la estructura de peones y restringe gradualmente las piezas rivales.",
  adaptive: "Analiza el estilo del jugador humano en tiempo real: juega agresivo cuando detecta imprecisiones y sólido bajo presión.",
  defensive: "Minimiza riesgos, construye fortalezas alrededor de su Rey y castiga cualquier sobreextensión."
};

// State
let sessionId = "session_" + Math.random().toString(36).substring(2, 9);
let gameState = null;
let selectedSquare = null;
let pendingPromotionMove = null;
let boardFlipped = false;
let showHeatmap = true;
let soundEnabled = true;
let isBusy = false;
let selfPlayInterval = null;

// WebAudio Synthesizer for Crisp Chess Sound Effects
const audioCtx = typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext)
  ? new (window.AudioContext || window.webkitAudioContext)()
  : null;

function playSound(type = 'move') {
  if (!soundEnabled || !audioCtx) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  const now = audioCtx.currentTime;

  if (type === 'capture') {
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(180, now);
    osc.frequency.exponentialRampToValueAtTime(65, now + 0.11);
    gain.gain.setValueAtTime(0.28, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.11);
    osc.start(now);
    osc.stop(now + 0.11);
  } else if (type === 'check') {
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, now);
    osc.frequency.setValueAtTime(880, now + 0.08);
    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.28);
    osc.start(now);
    osc.stop(now + 0.28);
  } else {
    osc.type = 'sine';
    osc.frequency.setValueAtTime(310, now);
    osc.frequency.exponentialRampToValueAtTime(140, now + 0.07);
    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.07);
    osc.start(now);
    osc.stop(now + 0.07);
  }
}

// DOM Elements
const boardEl = document.getElementById('chessboard');
const selectModel = document.getElementById('select-learning-model');
const selectReaction = document.getElementById('select-reaction-profile');
const selectColor = document.getElementById('select-player-color');
const selectDifficulty = document.getElementById('select-difficulty');
const modelDescEl = document.getElementById('model-desc');
const reactionDescEl = document.getElementById('reaction-desc');
const activeModelPill = document.getElementById('active-model-pill');
const statusTextEl = document.getElementById('status-text');
const turnDotEl = document.getElementById('turn-dot');
const thinkingBadgeEl = document.getElementById('thinking-badge');
const evalBarWhite = document.getElementById('eval-bar-white');
const evalBarBlack = document.getElementById('eval-bar-black');
const evalScoreLabel = document.getElementById('eval-score-label');
const fenInput = document.getElementById('fen-input');
const candidatesListEl = document.getElementById('candidates-list');
const reasoningListEl = document.getElementById('reasoning-list');
const moveHistoryEl = document.getElementById('move-history');
const statEpochs = document.getElementById('stat-epochs');
const statLoss = document.getElementById('stat-loss');
const statNodes = document.getElementById('stat-nodes');
const statTime = document.getElementById('stat-time');
const valueHeadBadge = document.getElementById('value-head-badge');
const nnCanvas = document.getElementById('nn-canvas');

function getSettings() {
  return {
    session_id: sessionId,
    learning_model: selectModel.value,
    reaction_profile: selectReaction.value,
    player_color: selectColor.value,
    difficulty: parseInt(selectDifficulty.value, 10)
  };
}

// Initialize Game
async function initNewGame() {
  stopSelfPlay();
  setBusy(true);
  const cfg = getSettings();
  boardFlipped = cfg.player_color === 'b';
  try {
    const res = await fetch('/api/game/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg)
    });
    gameState = await res.json();
    selectedSquare = null;
    renderAll();
  } catch (err) {
    console.error("Error starting game:", err);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  isBusy = busy;
  if (busy) {
    thinkingBadgeEl.classList.remove('hidden');
  } else {
    thinkingBadgeEl.classList.add('hidden');
  }
}

function renderAll() {
  if (!gameState) return;
  renderBoard();
  renderStatusAndEval();
  renderCapturedPieces();
  renderHistory();
  renderTelemetry();
  fenInput.value = gameState.fen;
}

function renderBoard() {
  boardEl.innerHTML = '';
  const files = 'abcdefgh';
  const ranks = '87654321';

  const lastMove = gameState.move_history.length > 0
    ? gameState.move_history[gameState.move_history.length - 1]
    : null;

  const legalMovesFromSelected = selectedSquare !== null
    ? gameState.legal_moves.filter(m => m.from_sq === selectedSquare)
    : [];

  const heatmap = gameState.telemetry && gameState.telemetry.heatmap
    ? gameState.telemetry.heatmap
    : null;

  for (let displayRow = 0; displayRow < 8; displayRow++) {
    for (let displayCol = 0; displayCol < 8; displayCol++) {
      const r = boardFlipped ? 7 - displayRow : displayRow;
      const c = boardFlipped ? 7 - displayCol : displayCol;
      const sq = r * 8 + c;
      const isLight = (r + c) % 2 === 0;

      const sqDiv = document.createElement('div');
      sqDiv.className = `square ${isLight ? 'light' : 'dark'}`;
      sqDiv.dataset.sq = sq;

      if (selectedSquare === sq) {
        sqDiv.classList.add('selected');
      } else if (lastMove && (lastMove.from_sq === sq || lastMove.to_sq === sq)) {
        sqDiv.classList.add('last-move');
      }

      if (gameState.in_check && gameState.king_sq === sq) {
        sqDiv.classList.add('in-check');
      }

      // Neural Attention Heatmap Overlay
      if (showHeatmap && heatmap && heatmap[r] && heatmap[r][c] > 0.38) {
        const intensity = (heatmap[r][c] - 0.35) * 0.52;
        const heatOverlay = document.createElement('div');
        heatOverlay.className = 'heatmap-overlay';
        heatOverlay.style.background = `radial-gradient(circle, rgba(168, 85, 247, ${intensity.toFixed(2)}) 0%, rgba(56, 189, 248, 0.05) 85%)`;
        sqDiv.appendChild(heatOverlay);
      }

      // Coordinates on edges
      if (displayCol === 0) {
        const rankSpan = document.createElement('span');
        rankSpan.className = 'coord-rank';
        rankSpan.textContent = ranks[r];
        sqDiv.appendChild(rankSpan);
      }
      if (displayRow === 7) {
        const fileSpan = document.createElement('span');
        fileSpan.className = 'coord-file';
        fileSpan.textContent = files[c];
        sqDiv.appendChild(fileSpan);
      }

      // Piece SVG
      const piece = gameState.board[sq];
      if (piece && PIECE_SVGS[piece]) {
        sqDiv.insertAdjacentHTML('beforeend', PIECE_SVGS[piece]);
      }

      // Legal move indicators
      const moveMatch = legalMovesFromSelected.find(m => m.to_sq === sq);
      if (moveMatch) {
        if (piece || moveMatch.is_en_passant) {
          const ring = document.createElement('div');
          ring.className = 'legal-capture-ring';
          sqDiv.appendChild(ring);
        } else {
          const dot = document.createElement('div');
          dot.className = 'legal-dot';
          sqDiv.appendChild(dot);
        }
      }

      sqDiv.addEventListener('click', () => onSquareClick(sq));
      boardEl.appendChild(sqDiv);
    }
  }
}

async function onSquareClick(sq) {
  if (isBusy || !gameState || gameState.status.is_game_over) return;

  // Check if clicking a legal destination from currently selected square
  if (selectedSquare !== null) {
    const candidateMoves = gameState.legal_moves.filter(
      m => m.from_sq === selectedSquare && m.to_sq === sq
    );

    if (candidateMoves.length > 0) {
      // Check if promotion move
      if (candidateMoves[0].promotion) {
        pendingPromotionMove = { from_sq: selectedSquare, to_sq: sq, candidates: candidateMoves };
        document.getElementById('promotion-modal').classList.remove('hidden');
        return;
      }
      await executePlayerMove(candidateMoves[0].uci);
      return;
    }
  }

  // Otherwise select own piece if it has legal moves
  const clickedPiece = gameState.board[sq];
  if (clickedPiece) {
    const isWhitePiece = clickedPiece === clickedPiece.toUpperCase();
    const isCurrentTurnPiece = (gameState.turn === 'w' && isWhitePiece) || (gameState.turn === 'b' && !isWhitePiece);
    if (isCurrentTurnPiece) {
      selectedSquare = selectedSquare === sq ? null : sq;
      renderBoard();
      return;
    }
  }

  selectedSquare = null;
  renderBoard();
}

async function executePlayerMove(uci) {
  selectedSquare = null;
  setBusy(true);
  const cfg = getSettings();
  try {
    const res = await fetch('/api/game/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        uci: uci,
        auto_reply: true,
        learning_model: cfg.learning_model,
        reaction_profile: cfg.reaction_profile,
        difficulty: cfg.difficulty
      })
    });
    if (!res.ok) {
      setBusy(false);
      return;
    }
    gameState = await res.json();
    if (gameState.in_check) {
      playSound('check');
    } else if (gameState.agent_move && gameState.agent_move.captured) {
      playSound('capture');
    } else {
      playSound('move');
    }
    renderAll();
  } catch (err) {
    console.error("Move error:", err);
  } finally {
    setBusy(false);
  }
}

async function triggerAutonomousAgentMove() {
  if (isBusy || !gameState || gameState.status.is_game_over) {
    if (gameState && gameState.status.is_game_over) stopSelfPlay();
    return;
  }
  setBusy(true);
  const cfg = getSettings();
  try {
    const res = await fetch('/api/game/agent-move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        learning_model: cfg.learning_model,
        reaction_profile: cfg.reaction_profile,
        difficulty: cfg.difficulty
      })
    });
    gameState = await res.json();
    playSound(gameState.in_check ? 'check' : 'move');
    renderAll();
    if (gameState.status.is_game_over) {
      stopSelfPlay();
    }
  } catch (err) {
    console.error("Agent move error:", err);
    stopSelfPlay();
  } finally {
    setBusy(false);
  }
}

function renderStatusAndEval() {
  const st = gameState.status;
  turnDotEl.className = `turn-dot ${gameState.turn === 'w' ? 'white' : 'black'}`;

  if (st.is_game_over) {
    statusTextEl.textContent = `🏁 ${st.reason}`;
  } else {
    const turnName = gameState.turn === 'w' ? 'Blancas' : 'Negras';
    const checkAlert = gameState.in_check ? ' — ¡JAQUE AL REY!' : '';
    statusTextEl.textContent = `Turno de las ${turnName}${checkAlert}`;
  }

  const tel = gameState.telemetry || {};
  const winWhite = typeof tel.win_prob_white === 'number' ? tel.win_prob_white : 50;
  const winBlack = 100 - winWhite;
  evalBarWhite.style.height = `${winWhite}%`;
  evalBarBlack.style.height = `${winBlack}%`;
  evalScoreLabel.textContent = `${Math.round(winWhite)}%`;
}

function renderCapturedPieces() {
  const capByWhite = [];
  const capByBlack = [];
  for (const m of gameState.move_history) {
    if (m.captured) {
      if (m.captured === m.captured.toLowerCase()) {
        capByWhite.push(UNICODE_PIECES[m.captured]);
      } else {
        capByBlack.push(UNICODE_PIECES[m.captured]);
      }
    }
  }
  document.getElementById('captured-by-white').textContent = capByWhite.join('') || '—';
  document.getElementById('captured-by-black').textContent = capByBlack.join('') || '—';
}

function renderHistory() {
  moveHistoryEl.innerHTML = '';
  const hist = gameState.move_history;
  for (let i = 0; i < hist.length; i += 2) {
    const moveNum = Math.floor(i / 2) + 1;
    const wMove = hist[i] ? hist[i].san : '';
    const bMove = hist[i + 1] ? hist[i + 1].san : '';

    const numSpan = document.createElement('span');
    numSpan.className = 'hist-num';
    numSpan.textContent = `${moveNum}.`;

    const wSpan = document.createElement('span');
    wSpan.className = 'hist-w';
    wSpan.textContent = wMove;

    const bSpan = document.createElement('span');
    bSpan.className = 'hist-b';
    bSpan.textContent = bMove;

    moveHistoryEl.appendChild(numSpan);
    moveHistoryEl.appendChild(wSpan);
    moveHistoryEl.appendChild(bSpan);
  }
  moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
}

function renderTelemetry() {
  const tel = gameState.telemetry;
  if (!tel) return;

  if (tel.learning_model) {
    activeModelPill.textContent = tel.learning_model.name;
  }

  const val = typeof tel.value_estimate === 'number' ? tel.value_estimate : 0;
  valueHeadBadge.textContent = `Valor: ${val >= 0 ? '+' : ''}${val.toFixed(3)}`;

  if (tel.training_epochs !== undefined) statEpochs.textContent = tel.training_epochs;
  if (tel.td_loss !== undefined) statLoss.textContent = tel.td_loss.toFixed(4);
  if (tel.nodes_evaluated !== undefined) statNodes.textContent = tel.nodes_evaluated;
  if (tel.elapsed_ms !== undefined) statTime.textContent = `${tel.elapsed_ms} ms`;

  // Update WDL bar, parameter count, depth & Principal Variation (PV)
  if (tel.wdl_probs && tel.wdl_probs.length === 3) {
    const wPct = Math.max(8, Math.round(tel.wdl_probs[0] * 100));
    const dPct = Math.max(8, Math.round(tel.wdl_probs[1] * 100));
    const lPct = Math.max(8, 100 - wPct - dPct);
    const wEl = document.getElementById('wdl-win');
    const dEl = document.getElementById('wdl-draw');
    const lEl = document.getElementById('wdl-loss');
    if (wEl && dEl && lEl) {
      wEl.style.width = `${wPct}%`;
      wEl.textContent = `${Math.round(tel.wdl_probs[0] * 100)}%`;
      dEl.style.width = `${dPct}%`;
      dEl.textContent = `${Math.round(tel.wdl_probs[1] * 100)}%`;
      lEl.style.width = `${lPct}%`;
      lEl.textContent = `${Math.round(tel.wdl_probs[2] * 100)}%`;
    }
  }

  const paramsLabel = document.getElementById('model-params-label');
  if (paramsLabel && tel.param_count) {
    const fmtParams = tel.param_count >= 1_000_000
      ? `${(tel.param_count / 1_000_000).toFixed(2)}M params`
      : `${Math.round(tel.param_count / 1000)}K params`;
    paramsLabel.textContent = `${fmtParams} • Depth ${tel.search_depth || 1} • TT ${tel.tt_hits || 0}`;
  }

  const pvEl = document.getElementById('pv-line-text');
  if (pvEl) {
    pvEl.textContent = (tel.pv_line && tel.pv_line.length > 0) ? tel.pv_line.join(' ') : '—';
  }

  // Render Top Candidate Moves
  const cands = tel.candidate_moves || [];
  if (cands.length > 0) {
    candidatesListEl.innerHTML = '';
    const scores = cands.map(c => c.score);
    const minS = Math.min(...scores);
    const maxS = Math.max(...scores);

    cands.slice(0, 5).forEach((c, idx) => {
      const normPct = maxS > minS ? ((c.score - minS) / (maxS - minS)) * 75 + 25 : 80 - idx * 12;
      const item = document.createElement('div');
      item.className = 'candidate-item';
      item.innerHTML = `
        <span class="cand-san">${idx === 0 ? '★ ' : ''}${c.san}</span>
        <div class="cand-bar-wrap">
          <div class="cand-bar-fill" style="width: ${normPct.toFixed(0)}%;"></div>
        </div>
        <span class="cand-score">${c.score >= 0 ? '+' : ''}${c.score.toFixed(2)}</span>
      `;
      candidatesListEl.appendChild(item);
    });
  }

  // Render Reasoning Log
  if (tel.reasoning && tel.reasoning.length > 0) {
    reasoningListEl.innerHTML = '';
    tel.reasoning.forEach(line => {
      const li = document.createElement('li');
      li.textContent = line;
      reasoningListEl.appendChild(li);
    });
  }

  // Draw Interactive Neural Network Topology & Synaptic Activations
  if (tel.layer_activations) {
    drawNeuralNetworkCanvas(tel.layer_activations);
  }
}

function drawNeuralNetworkCanvas(activations) {
  const ctx = nnCanvas.getContext('2d');
  const W = nnCanvas.width;
  const H = nnCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const layers = [
    (activations.input_layer || []).slice(0, 8),
    (activations.residual_block_1 || []).slice(0, 8),
    (activations.residual_block_2 || []).slice(0, 8),
    (activations.value_head || []).slice(0, 6),
    [activations.output_value || 0]
  ];

  const layerX = [36, 130, 225, 320, 392];
  const nodeCoords = [];

  layers.forEach((layer, lIdx) => {
    const count = layer.length;
    const coords = [];
    const spacing = (H - 28) / Math.max(count, 1);
    for (let i = 0; i < count; i++) {
      const y = 16 + spacing * (i + 0.5);
      coords.push({ x: layerX[lIdx], y, val: layer[i] });
    }
    nodeCoords.push(coords);
  });

  // Draw Synaptic Connections
  for (let l = 0; l < nodeCoords.length - 1; l++) {
    const srcLayer = nodeCoords[l];
    const dstLayer = nodeCoords[l + 1];
    for (const s of srcLayer) {
      for (const d of dstLayer) {
        const strength = Math.abs(s.val * d.val);
        if (strength < 0.02) continue;
        const alpha = Math.min(0.65, strength * 0.85 + 0.06);
        const isPos = (s.val * d.val) >= 0;
        ctx.strokeStyle = isPos
          ? `rgba(56, 189, 248, ${alpha.toFixed(2)})`
          : `rgba(244, 63, 94, ${alpha.toFixed(2)})`;
        ctx.lineWidth = strength > 0.25 ? 1.5 : 0.8;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(d.x, d.y);
        ctx.stroke();
      }
    }
  }

  // Draw Neurons
  nodeCoords.forEach((layer, lIdx) => {
    layer.forEach(node => {
      const radius = lIdx === 4 ? 9 : 5.2;
      const v = node.val;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      if (v >= 0) {
        const intensity = Math.min(1, Math.abs(v) + 0.25);
        ctx.fillStyle = `rgba(56, 189, 248, ${intensity})`;
      } else {
        const intensity = Math.min(1, Math.abs(v) + 0.25);
        ctx.fillStyle = `rgba(168, 85, 247, ${intensity})`;
      }
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = '#e2e8f0';
      ctx.stroke();
    });
  });
}

// Self-Play Autonomous Mode (AI vs AI)
function toggleSelfPlay() {
  const btn = document.getElementById('btn-self-play');
  if (selfPlayInterval) {
    stopSelfPlay();
  } else {
    btn.classList.add('active');
    btn.textContent = '⏹️ Detener Duelo Autónomo (IA vs IA)';
    triggerAutonomousAgentMove();
    selfPlayInterval = setInterval(() => {
      if (!isBusy) {
        triggerAutonomousAgentMove();
      }
    }, 950);
  }
}

function stopSelfPlay() {
  if (selfPlayInterval) {
    clearInterval(selfPlayInterval);
    selfPlayInterval = null;
  }
  const btn = document.getElementById('btn-self-play');
  btn.classList.remove('active');
  btn.textContent = '⚔️ Modo Duelo Autónomo (IA vs IA)';
}

// Event Listeners
selectModel.addEventListener('change', () => {
  modelDescEl.textContent = MODEL_DESCRIPTIONS[selectModel.value] || '';
});

selectReaction.addEventListener('change', () => {
  reactionDescEl.textContent = REACTION_DESCRIPTIONS[selectReaction.value] || '';
});

document.getElementById('btn-new-game').addEventListener('click', initNewGame);
document.getElementById('btn-ai-move').addEventListener('click', triggerAutonomousAgentMove);
document.getElementById('btn-self-play').addEventListener('click', toggleSelfPlay);

document.getElementById('btn-flip').addEventListener('click', () => {
  boardFlipped = !boardFlipped;
  renderBoard();
});

document.getElementById('btn-heatmap').addEventListener('click', () => {
  showHeatmap = !showHeatmap;
  document.getElementById('heatmap-state').textContent = showHeatmap ? 'ON' : 'OFF';
  renderBoard();
});

document.getElementById('btn-sound').addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  document.getElementById('sound-state').textContent = soundEnabled ? 'ON' : 'OFF';
});

document.getElementById('btn-train-net').addEventListener('click', async () => {
  if (isBusy) return;
  setBusy(true);
  try {
    const res = await fetch('/api/agent/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        learning_model: selectModel.value,
        iterations: 30
      })
    });
    const stats = await res.json();
    statEpochs.textContent = stats.epochs_total;
    statLoss.textContent = stats.final_loss.toFixed(4);

    const li = document.createElement('li');
    li.textContent = `Entrenamiento RL completado: +${stats.iterations} épocas (Loss final: ${stats.final_loss}).`;
    reasoningListEl.prepend(li);
  } catch (err) {
    console.error("Training error:", err);
  } finally {
    setBusy(false);
  }
});

// Promotion Modal Buttons
document.querySelectorAll('.prom-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const promChar = btn.dataset.prom;
    document.getElementById('promotion-modal').classList.add('hidden');
    if (pendingPromotionMove) {
      const match = pendingPromotionMove.candidates.find(m => m.promotion === promChar)
        || pendingPromotionMove.candidates[0];
      pendingPromotionMove = null;
      await executePlayerMove(match.uci);
    }
  });
});

// Boot application
window.addEventListener('DOMContentLoaded', () => {
  initNewGame();
});
