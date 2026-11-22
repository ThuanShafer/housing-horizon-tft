import fs from "fs";
import { loadModel, fromEncoded, classify } from "../web/tft.js";

const modelPath = process.argv[2] || new URL("../web/model/model.json", import.meta.url);
const parityPath = process.argv[3] || new URL("../web/model/parity.json", import.meta.url);
const examplesPath = process.argv[4] || new URL("../web/model/examples.json", import.meta.url);

const doc = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const parity = JSON.parse(fs.readFileSync(parityPath, "utf8"));
const model = loadModel(doc);
const pred = fromEncoded(model, parity.inputs);

function maxDiff(a, b) {
  let m = 0;
  const flatA = a.flat(3);
  const flatB = b.flat(3);
  for (let i = 0; i < flatA.length; i++) m = Math.max(m, Math.abs(flatA[i] - flatB[i]));
  return m;
}

const nowDiff = maxDiff(Array.from(pred.now.data), parity.now.flat());
const futDiff = maxDiff(Array.from(pred.future.data), parity.future.flat(2));
console.log("now logit max abs", nowDiff.toExponential(3));
console.log("future logit max abs", futDiff.toExponential(3));
if (nowDiff > 1e-3 || futDiff > 1e-3) {
  console.error("forward parity failed");
  process.exit(1);
}

if (fs.existsSync(examplesPath)) {
  const examples = JSON.parse(fs.readFileSync(examplesPath, "utf8"));
  let worst = 0;
  for (const example of examples) {
    const result = classify(model, { history: example.history });
    const probs = result.classification.probabilities.map((row) => row.probability);
    for (let i = 0; i < probs.length; i++) worst = Math.max(worst, Math.abs(probs[i] - example.torch_h6[i]));
  }
  console.log("example probability max abs", worst.toExponential(3));
  if (worst > 2e-2) {
    console.error("example parity failed");
    process.exit(1);
  }
}
console.log("parity ok");
