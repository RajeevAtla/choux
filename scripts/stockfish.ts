import { createInterface } from 'node:readline';
import { Readable } from 'node:stream';

export const nodes = 20_000;

export async function startStockfish(executable: string) {
  const process = Bun.spawn([executable], {
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'inherit',
  });
  const lines = createInterface({ input: Readable.from(process.stdout) });
  const reader = lines[Symbol.asyncIterator]();
  const send = (command: string) => {
    process.stdin.write(`${command}\n`);
    process.stdin.flush();
  };

  async function readUntil(prefix: string, observe?: (line: string) => void) {
    for (;;) {
      const item = await reader.next();
      if (item.done)
        throw new Error(`Stockfish exited while waiting for ${prefix}.`);
      if (item.value.startsWith(prefix)) return item.value;
      observe?.(item.value);
    }
  }

  async function ready() {
    send('isready');
    await readUntil('readyok');
  }

  function close() {
    lines.close();
    process.kill();
  }

  let name = '';
  try {
    send('uci');
    await readUntil('uciok', (line) => {
      if (line.startsWith('id name ')) name = line.slice(8);
    });
    send('setoption name Threads value 1');
    send('setoption name Hash value 16');
    send('setoption name UCI_ShowWDL value true');
    await ready();
  } catch (error) {
    close();
    throw error;
  }

  async function analyze(
    fen: string,
    restriction?: string,
    options?: { nodes: number; moves: readonly string[] },
  ) {
    send('ucinewgame');
    await ready();
    send(
      options?.moves.length
        ? `position startpos moves ${options.moves.join(' ')}`
        : `position fen ${fen}`,
    );
    send(
      `go nodes ${options?.nodes ?? nodes}${restriction ? ` searchmoves ${restriction}` : ''}`,
    );
    let points: number | undefined;
    const end = await readUntil('bestmove ', (line) => {
      if (/\b(?:lowerbound|upperbound)\b/.test(line)) return;
      const wdl = /\bwdl (\d+) (\d+) (\d+)\b/.exec(line);
      if (wdl) points = (Number(wdl[1]) + Number(wdl[2]) / 2) / 1000;
    });
    const move = end.split(' ')[1];
    if (
      points === undefined ||
      !move ||
      move === '(none)' ||
      (restriction && move !== restriction)
    ) {
      throw new Error(
        `Stockfish did not score ${fen}${restriction ? ` after ${restriction}` : ''}.`,
      );
    }
    return { move, points };
  }

  return { analyze, name, close };
}
