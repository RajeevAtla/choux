# choux

light and fluffy (like the pastry) way to play chess in your browser

Play at [rajeevatla.com/choux](https://rajeevatla.com/choux/) or run `bun install --frozen-lockfile` and `bun run dev` locally. Choose White or Black to start a game. The Svelte app uses `chess.js` for legal moves and a Wasm worker for the model. The worker scores every first move, then checks the opponent's best reply to up to eight candidates within a soft 200 ms search budget.

The theme follows your device by default. Use the **Theme** control to choose Light or Dark. The chess sprite is by Cburnett and Rfc1394, adapted by [cm-chessboard](https://github.com/shaack/cm-chessboard), under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Choux uses Azo Sans and Chassi L by [R-Typography](https://www.r-typography.com/fonts), bundled with the foundry's permission. The font files are third-party assets and are subject to R-Typography's terms.

## Inference benchmark

Open **Inference benchmark** from the game and choose **Run benchmark**. The worker loads a seeded dummy checkpoint and compares JAX-JS Wasm and WebGPU. The warm numbers cover browser round-trip inference on synthetic features. They exclude chess move generation and search.

Use Wasm as the initial engine backend. On Chrome 158 on Windows (Core i9-13905H, Intel Gen12LP WebGPU adapter), Wasm had lower startup and warm inference latency for both batch sizes. The production build was served under `/choux/` with cross-origin isolation disabled, matching the deployment path shape and header requirements.

| Backend | Cold start | 64 positions, warm P95 | 512 positions, warm P95 |
| ------- | ---------: | ---------------------: | ----------------------: |
| Wasm    |   320.3 ms |                2.40 ms |                13.10 ms |
| WebGPU  |  1301.5 ms |                9.40 ms |                55.40 ms |

`bun run biome:check` checks formatting and lint, `bun run biome:format` applies formatting, and `bun run check` runs Svelte and TypeScript diagnostics. `bun run build` produces `dist/` with relative asset paths for GitHub Pages. Repository Pages settings should use **GitHub Actions** as the publishing source.

## Training

The 782-input, 64-hidden-unit model scores positions from the side-to-move's perspective. The first checkpoint trained on 6,405 positions and reached a mean absolute error of 0.331 on 707 held-out positions, down from 0.516 with random weights. Run `bun run train` to train from scratch and `bun run evaluate public/model.f32 --matches` to compare against random and material-only opponents.

In four games against each baseline, Choux won one game against random moves and lost one against material-only search. The other six games reached the 160-ply cap. Choux led in material in all three unfinished games against random play and trailed in all three against material-only search.

After a completed game, **Share game for training** opens a prefilled public GitHub issue with the moves, side, result, and checkpoint SHA-256. Players decide whether to submit the issue on GitHub.

The Actions workflow runs daily at 04:23 UTC. To run it on demand, choose **Check code** in the Actions tab and select **retrain** when dispatching it. The runner replays submitted games to verify legality and the result, deduplicates by move sequence, and records accepted games in `data/games.jsonl`. Training samples positions from recent submissions and labels them with the same teacher as the generated games.

The workflow checks the candidate against a fixed holdout set and runs a chess smoke test. It retains the incumbent if the candidate regresses. The build receives the selected checkpoint as an Actions artifact and deploys it with the static app. Git stores the accepted dataset and checkpoint so subsequent builds can reuse them.
