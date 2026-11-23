# choux

light and fluffy (like the pastry) way to play chess in your browser

## Inference benchmark

Run `bun install --frozen-lockfile` and `bun run dev`, then choose **Run benchmark**. The worker loads a seeded dummy checkpoint and compares JAX-JS Wasm and WebGPU. The warm numbers are browser round-trip inference timings for synthetic features. They exclude chess move generation and search.

Use Wasm as the initial engine backend. On Chrome 158 on Windows (Core i9-13905H, Intel Gen12LP WebGPU adapter), Wasm had lower startup and warm inference latency for both batch sizes. The production build was served under `/choux/` with cross-origin isolation disabled, matching the deployment path shape and header requirements.

| Backend | Cold start | 64 positions, warm P95 | 512 positions, warm P95 |
| ------- | ---------: | ---------------------: | ----------------------: |
| Wasm    |   320.3 ms |                2.40 ms |                13.10 ms |
| WebGPU  |  1301.5 ms |                9.40 ms |                55.40 ms |

`bun run biome:check` checks formatting and lint, `bun run biome:format` applies formatting, and `bun run check` runs Svelte and TypeScript diagnostics. `bun run build` produces `dist/` with relative asset paths for GitHub Pages. Repository Pages settings should use **GitHub Actions** as the publishing source.

## Daily training plan

1. Offer an opt-in **Share game for training** action after play. Open a prefilled GitHub issue containing a compact game record, with moves, sides, result, and model version. Ask the player to submit it on GitHub and make clear that the issue is public. GitHub Issues are the durable inbox and require no repository token in the browser.
2. Use a daily and manually runnable Actions workflow to read the inbox. Replay moves to verify legality, cap game length, and deduplicate by a hash of the canonical game. Store accepted records as versioned data. Games contribute positions, which the local teacher labels for the evaluator.
3. Retrain with Bun in Actions and validate on a fixed held-out set. Pass the candidate checkpoint to the build job as an Actions artifact. Persist the last passing checkpoint so a later code-only deploy can reuse it. Package the versioned weights alongside the app and deploy them together to GitHub Pages.
