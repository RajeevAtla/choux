# choux

light and fluffy (like the pastry) way to play chess in your browser

## Inference benchmark

Run `bun install --frozen-lockfile` and `bun run dev`, then choose **Run benchmark**. The worker loads a seeded dummy checkpoint and compares JAX-JS Wasm and WebGPU. The warm numbers are browser round-trip inference timings for synthetic features. They exclude chess move generation and search.

`bun run biome:check` checks formatting and lint, `bun run biome:format` applies formatting, and `bun run check` runs Svelte and TypeScript diagnostics. `bun run build` produces `dist/` with relative asset paths for GitHub Pages. Repository Pages settings should use **GitHub Actions** as the publishing source.

## Daily training plan

1. Offer an opt-in **Share game for training** action after play. Open a prefilled GitHub issue containing a compact game record, with moves, sides, result, and model version. Ask the player to submit it on GitHub and make clear that the issue is public. GitHub Issues are the durable inbox and require no repository token in the browser.
2. Use a daily and manually runnable Actions workflow to read the inbox. Replay moves to verify legality, cap game length, and deduplicate by a hash of the canonical game. Store accepted records as versioned data. Games contribute positions, which the local teacher labels for the evaluator.
3. Retrain with Bun in Actions and validate on a fixed held-out set. Pass the candidate checkpoint to the build job as an Actions artifact. Promote a checkpoint only if it passes the model checks, then package the versioned weights alongside the app and deploy them together to GitHub Pages.
