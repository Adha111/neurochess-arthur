# NeuroChess AI v3.0 — PyTorch Deep Neural Chess Engine & Cloud Kubernetes Platform

Aplicación web interactiva de ajedrez impulsada por **4 Arquitecturas Neuronales Profundas en PyTorch** inspiradas en ingeniería inversa de **Stockfish NNUE (`SFNNv9`)** y **Leela Chess Zero (`Lc0 SE-ResNet + BT4 Transformer`)**, combinadas con búsqueda avanzada (`PVS`, `Transposition Tables`, `SEE`, `Quiescence Search`, `Null-Move Pruning`, `LMR`, `PUCT MCTS`), empaquetada en **Docker** (PyTorch CPU optimizado) y lista para **Kubernetes en producción / nube**.

---

## Arquitecturas Neuronales (PyTorch)

1. **`Lc0_SEResNet_Transformer` (`alphazero_mcts`)**:
   - Tensor espacial canónico de **20 planos (20×8×8)** incluyendo mapas de ataque táctico en tiempo real (`us` / `them`).
   - Stem Convolucional + **3 bloques Residuales Squeeze-and-Excitation (`SE-ResNet`)**.
   - **Transformer Multi-Cabeza (`BT4` style)** sobre los 64 tokens del tablero para capturar relaciones de largo alcance (clavadas, rayos X, baterías).
   - **Cabezal de Política por Atención (`4096` pares origen-destino)** + **Cabezal WDL (`Win/Draw/Loss`)** + **Moves-Left Head (`MLH`)**.
   - Integrado con búsqueda **MCTS PUCT**, ruido Dirichlet y guardia táctica `SEE` de 1 ply.

2. **`StockfishNNUE_SFNNv9` (`minimax_nnue`)**:
   - Extractor disperso **`HalfKAv2_hm`** (32 cubetas de Rey reflejadas horizontalmente × 12 piezas × 64 casillas = **24,576 características**) mediante `nn.EmbeddingBag` dual (`us` / `them`).
   - Atajo lineal **`PSQT`** por 8 cubetas de fase de juego + activación **`SCReLU`** (`clamp(x, 0, 1)^2`).
   - **8 Layer Stacks** especializados por fase material + cabezal de ordenamiento de jugadas.
   - Integrado con búsqueda estilo Stockfish: **Iterative Deepening**, **Principal Variation Search (PVS)**, **Transposition Table (Zobrist 64-bit)**, **Null-Move Pruning**, **Late Move Reductions (LMR)**, **Killer/History Heuristics**, **Static Exchange Evaluation (`SEE`)** y **Quiescence Search con Delta Pruning**.

3. **`DeepQ_ActorCritic_RL` (`dqn_td_rl`)**:
   - Arquitectura híbrida de doble flujo (Convolucional + Vector Táctico de 160 dimensiones).
   - Flujos **Dueling Value/Advantage** con entrenamiento online **AdamW** y **Prioritized Experience Replay (PER)**.

4. **`NeuroEvolutionaryCommittee` (`evolutionary_ga`)**:
   - Comité de **4 genomas neuronales PyTorch** ponderados por aptitud (*fitness*) con selección por torneo, cruce uniforme y mutación gaussiana adaptativa.

---

## Pre-Entrenamiento y Benchmarks Tácticos

Los pesos pre-entrenados se guardan en `backend/weights/*.pt` y se empaquetan directamente dentro de la imagen Docker. Para re-entrenar y ejecutar los benchmarks tácticos:

```bash
python backend/pretrain_models.py
```

---

## Contenedor Docker (Local y Cloud Ready)

La imagen utiliza `python:3.11-slim` con **PyTorch CPU-only** (sin bloat de CUDA) y usuario `non-root` (`uid 1000`):

```bash
# Construir imagen versionada
docker build -t neuro-chess:v3.0.0 .

# Levantar contenedor
docker run -d --name neuro-chess -p 8000:8000 --restart unless-stopped neuro-chess:v3.0.0
```

O con **Docker Compose**:
```bash
docker compose up --build -d
```

---

## Despliegue en Kubernetes (Local o Nube: GKE / EKS / AKS)

Manifiestos Kustomize en `k8s/` con `RollingUpdate` sin downtime (`maxUnavailable: 0`), `startupProbe`, `livenessProbe`, `readinessProbe`, `HorizontalPodAutoscaler` (`HPA`) y `PodDisruptionBudget`:

```bash
# 1. Para nube, sube la imagen a tu registry (ej. Docker Hub / ECR / GCR / Artifact Registry):
#    docker tag neuro-chess:v3.0.0 <tu-registry>/neuro-chess:v3.0.0
#    docker push <tu-registry>/neuro-chess:v3.0.0

# 2. Aplicar o actualizar el clúster con Kustomize:
kubectl apply -k k8s/

# 3. Forzar rollout si reutilizas el mismo tag:
kubectl rollout restart deployment/neuro-chess-deployment -n neuro-chess
kubectl rollout status deployment/neuro-chess-deployment -n neuro-chess
```
