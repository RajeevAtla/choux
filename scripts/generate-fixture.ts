import { WEIGHT_COUNT } from '../src/modelShape';

const bytes = new ArrayBuffer(WEIGHT_COUNT * Float32Array.BYTES_PER_ELEMENT);
const weights = new DataView(bytes);
let seed = 0x43484f55;

for (let index = 0; index < WEIGHT_COUNT; index++) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  weights.setFloat32(index * 4, (seed / 2 ** 32 - 0.5) * 0.1, true);
}

await Bun.write('public/benchmark-weights.f32', bytes);
