# choux

light and fluffy (like the pastry) way to play chess in your browser

Play at [rajeevatla.com/choux](https://rajeevatla.com/choux/) or run `bun install --frozen-lockfile` and `bun run dev` locally. Choose White or Black to start a game. The Svelte app uses `chess.js` for legal moves and a Wasm worker for the model. The worker scores every first move, then checks the opponent's best reply to up to eight candidates within a soft 200 ms search budget.

The theme follows your device by default. Use the **Theme** control to choose Light or Dark. The chess sprite is by Cburnett and Rfc1394, adapted by [cm-chessboard](https://github.com/shaack/cm-chessboard), under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Choux uses Azo Sans and Chassi L by [R-Typography](https://www.r-typography.com/fonts), bundled with the foundry's permission. The font files are third-party assets and are subject to R-Typography's terms.

## Inference benchmark

Open **Engine & model** below the game and choose **Run benchmark**. The worker loads a seeded dummy checkpoint and compares JAX-JS Wasm and WebGPU. The warm numbers cover browser round-trip inference on synthetic features. They exclude chess move generation and search.

Use Wasm as the initial engine backend. On Chrome 158 on Windows (Core i9-13905H, Intel Gen12LP WebGPU adapter), Wasm had lower startup and warm inference latency for both batch sizes. The production build was served under `/choux/` with cross-origin isolation disabled, matching the deployment path shape and header requirements.

| Backend | Cold start | 64 positions, warm P95 | 512 positions, warm P95 |
| ------- | ---------: | ---------------------: | ----------------------: |
| Wasm    |   320.3 ms |                2.40 ms |                13.10 ms |
| WebGPU  |  1301.5 ms |                9.40 ms |                55.40 ms |

`bun run biome:check` checks formatting and lint, `bun run biome:format` applies formatting, and `bun run check` runs Svelte and TypeScript diagnostics. `bun run build` produces `dist/` with relative asset paths for GitHub Pages. Repository Pages settings should use **GitHub Actions** as the publishing source.

## Network

Choux is a fully connected JAX-JS value network with shape **782 → 64 → 1**. For a batch of `B` positions it takes a `B × 782` Float32 tensor, applies a 64-unit ReLU layer, and returns a `B × 1` tensor of linear scores from the side-to-move's perspective. The chess search uses these scores to rank legal moves.

Each input row contains 768 piece-square flags (12 planes of 64 squares arranged by side to move and opponent), four castling flags, eight en passant file flags, the half-move clock clamped at 100 and divided by 100, and a flag for Black to move. The encoder flips the board ranks for Black so pieces on each side have a consistent orientation.

| Tensor         | Shape      | Trainable parameters |
| -------------- | ---------- | -------------------: |
| Hidden weights | `782 × 64` |               50,048 |
| Hidden bias    | `64`       |                   64 |
| Output weights | `64 × 1`   |                   64 |
| Output bias    | `1`        |                    1 |
| **Total**      |            |           **50,177** |

[`public/model.f32`](public/model.f32) stores those four tensors in order as raw little-endian Float32 values. The checkpoint is **200,708 bytes (about 196 KiB)**.

## Training

The first checkpoint trained on 6,405 positions and reached a mean absolute error of 0.331 on 707 held-out positions, down from 0.516 with random weights. Run `bun run train` to train from scratch and `bun run evaluate public/model.f32 --matches` to compare against random and material-only opponents.

The [teacher](scripts/teacher.ts) tries all legal moves one ply ahead and labels the position with its best score from the current player's point of view. Static scores use material (pawn 100, knight 320, bishop 330, rook 500, queen 900), placement, and check. A move that checkmates scores 1,200 and a move that draws scores zero. Dividing by 1,200 and clamping to `[-1, 1]` gives the network's training target. Game outcomes are recorded separately from these labels.

In four games against each baseline, Choux won one game against random moves and lost one against material-only search. The other six games reached the 160-ply cap. Choux led in material in all three unfinished games against random play and trailed in all three against material-only search.

After a completed game, **Share game for training** opens a prefilled public GitHub issue with the moves, side, result, and checkpoint SHA-256. Players decide whether to submit the issue on GitHub.

Training runs every six hours at 00:43, 06:43, 12:43, and 18:43 UTC. To run it on demand, choose **Check code** in the Actions tab and select **retrain**. The runner replays submitted games to verify legality and the result, deduplicates by move sequence, and records accepted games in `data/games.jsonl`. Training samples from up to 256 recent issue games and up to six self-play batches with available artifacts from the latest 30 successful self-play runs on `main`. This includes both scheduled and manually triggered runs. The training report counts the games and positions used from each source after deduplication.

The workflow checks the candidate against a fixed holdout set and runs a chess smoke test. It retains the incumbent if the candidate regresses. The build receives the selected checkpoint as an Actions artifact and deploys it with the static app. Git stores the accepted dataset and checkpoint so subsequent builds can reuse them.

### Stockfish comparison

Every retrain compares the candidate and incumbent against [Stockfish 18](https://github.com/official-stockfish/Stockfish/releases/tag/sf_18) on 22 fixed positions. At 20,000 nodes per search, Stockfish estimates win, draw, and loss probabilities for its best move and for each chosen move. The benchmark reports the average expected-score points lost (`win + draw / 2`) and agreement with Stockfish's best move. Lower loss is better. The promotion gate still uses held-out teacher error and a chess smoke test.

The `stockfish-benchmark` Actions artifact contains the per-position report and validation metrics. The compact `public/model-history.jsonl` file records each run's summary, including rejected candidates, and feeds the **Model history** table in **Engine & model**.

### Self-play dataset

At 27 minutes past each hour, **Self-play dataset** generates 16 games using the current checkpoint for both sides, with six seeded random opening plies and a cap of 96 plies per game. Each run uploads `games.jsonl` as the **choux-self-play** artifact. Choose **Self-play dataset → Run workflow** to generate a downloadable artifact on demand. Each row records the move sequence, checkpoint SHA-256, seed, and result. A `null` result means the game reached the cap. Those positions can still receive teacher scores.

To generate a smaller batch locally and use it for training, run:

```sh
bun run self-play self-play/games.jsonl public/model.f32 4 80 example
bun run train self-play/candidate.f32 data/games.jsonl public/model.f32 self-play/games.jsonl
```
