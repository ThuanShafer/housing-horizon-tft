// Browser and Node inference for the housing Temporal Fusion Transformer.
// The arithmetic follows housing_tft/model.py. Dropout is off, as at evaluation.

const ALIASES = {
  ethnicity: { maori: "Māori", pasifika: "Pacific", "nz european": "European", pakeha: "European", "pākehā": "European" },
  local_board: {
    kaipatiki: "Kaipātiki",
    "waitakere ranges": "Waitākere Ranges",
    waitemata: "Waitematā",
    puketapapa: "Puketāpapa",
    orakei: "Ōrākei",
    "maungakiekie-tamaki": "Maungakiekie-Tāmaki",
    "mangere-otahuhu": "Māngere-Ōtāhuhu",
    "otara-papatoetoe": "Ōtara-Papatoetoe",
    "aotea / great barrier": "Aotea/Great Barrier",
    "great barrier": "Aotea/Great Barrier",
  },
  disability: { true: "yes", false: "no", "1": "yes", "0": "no", y: "yes", n: "no" },
  prior_justice_history: { true: "yes", false: "no", "1": "yes", "0": "no", y: "yes", n: "no" },
  residency_status: { "permanent resident": "permanent_resident", "resident visa": "resident_visa", pr: "permanent_resident" },
  whanau_type: { single: "single_adult", "sole parent": "sole_parent", "two parent": "two_parent", "multi family": "multi_family" },
  deprivation_category: { "no tenure": "no_secure_tenure", "no secure tenure": "no_secure_tenure" },
  emergency_accommodation_type: { rough: "rough_or_vehicle", car: "rough_or_vehicle", vehicle: "rough_or_vehicle", "womens refuge": "refuge", "women's refuge": "refuge" },
  main_benefit: { "jobseeker support": "jobseeker", sps: "sole_parent_support", slp: "supported_living" },
};

const SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function numel(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function decodeTensor(entry) {
  const bin = atob(entry.b64);
  const raw = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) raw[i] = bin.charCodeAt(i);
  if (entry.dtype === "f32") return { shape: entry.shape.slice(), data: new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4) };
  if (entry.dtype === "i32") return { shape: entry.shape.slice(), data: new Int32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4) };
  return { shape: entry.shape.slice(), data: raw };
}

function sameShape(a, b) {
  return a.shape.length === b.shape.length && a.shape.every((v, i) => v === b.shape[i]);
}

function map1(t, fn) {
  const data = new Float32Array(t.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(t.data[i]);
  return { shape: t.shape.slice(), data };
}

function elu(t) { return map1(t, (x) => (x > 0 ? x : Math.exp(x) - 1)); }
function tanh(t) { return map1(t, Math.tanh); }
function sigmoid(t) {
  return map1(t, (x) => {
    const z = Math.max(-60, Math.min(60, x));
    return 1 / (1 + Math.exp(-z));
  });
}

function add(a, b) {
  if (!sameShape(a, b)) throw new Error(`add ${a.shape} vs ${b.shape}`);
  const data = new Float32Array(a.data.length);
  for (let i = 0; i < data.length; i++) data[i] = a.data[i] + b.data[i];
  return { shape: a.shape.slice(), data };
}

function mul(a, b) {
  if (!sameShape(a, b)) throw new Error(`mul ${a.shape} vs ${b.shape}`);
  const data = new Float32Array(a.data.length);
  for (let i = 0; i < data.length; i++) data[i] = a.data[i] * b.data[i];
  return { shape: a.shape.slice(), data };
}

function scale(t, k) { return map1(t, (x) => x * k); }

function reshape(t, shape) {
  if (numel(shape) !== t.data.length) throw new Error("reshape size");
  return { shape, data: t.data };
}

function unsqueeze(t, axis) {
  const ax = axis < 0 ? t.shape.length + 1 + axis : axis;
  const shape = t.shape.slice();
  shape.splice(ax, 0, 1);
  return { shape, data: t.data };
}

function squeeze(t, axis) {
  const ax = axis < 0 ? t.shape.length + axis : axis;
  if (t.shape[ax] !== 1) throw new Error("squeeze");
  const shape = t.shape.slice();
  shape.splice(ax, 1);
  return { shape, data: t.data };
}

function concat(tensors, axis) {
  const ax = axis < 0 ? tensors[0].shape.length + axis : axis;
  const shape = tensors[0].shape.slice();
  shape[ax] = tensors.reduce((sum, t) => sum + t.shape[ax], 0);
  const out = new Float32Array(numel(shape));
  const outer = numel(shape.slice(0, ax)) || 1;
  const inner = numel(shape.slice(ax + 1)) || 1;
  let offset = 0;
  for (const t of tensors) {
    const blockIn = t.shape[ax] * inner;
    const blockOut = shape[ax] * inner;
    for (let o = 0; o < outer; o++) {
      out.set(t.data.subarray(o * blockIn, o * blockIn + blockIn), o * blockOut + offset * inner);
    }
    offset += t.shape[ax];
  }
  return { shape, data: out };
}

function linear(x, w, b) {
  const inD = x.shape[x.shape.length - 1];
  const outD = w.shape[0];
  const rows = x.data.length / inD;
  const data = new Float32Array(rows * outD);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outD; o++) {
      let s = b ? b.data[o] : 0;
      const x0 = r * inD;
      const w0 = o * inD;
      for (let i = 0; i < inD; i++) s += x.data[x0 + i] * w.data[w0 + i];
      data[r * outD + o] = s;
    }
  }
  return { shape: x.shape.slice(0, -1).concat([outD]), data };
}

function batchedLinear(x, w, b) {
  const inD = x.shape[x.shape.length - 1];
  const n = x.shape[x.shape.length - 2];
  const outD = w.shape[2];
  const groups = x.data.length / (n * inD);
  const data = new Float32Array(groups * n * outD);
  for (let g = 0; g < groups; g++) {
    for (let ni = 0; ni < n; ni++) {
      const x0 = (g * n + ni) * inD;
      for (let o = 0; o < outD; o++) {
        let s = b.data[ni * outD + o];
        for (let i = 0; i < inD; i++) s += x.data[x0 + i] * w.data[(ni * inD + i) * outD + o];
        data[(g * n + ni) * outD + o] = s;
      }
    }
  }
  return { shape: x.shape.slice(0, -1).concat([outD]), data };
}

function layerNorm(x, weight, bias) {
  const d = x.shape[x.shape.length - 1];
  const block = weight.data.length;
  const innerRows = block / d;
  const groups = x.data.length / block;
  const data = new Float32Array(x.data.length);
  for (let g = 0; g < groups; g++) {
    for (let r = 0; r < innerRows; r++) {
      const base = g * block + r * d;
      let mean = 0;
      for (let i = 0; i < d; i++) mean += x.data[base + i];
      mean /= d;
      let variance = 0;
      for (let i = 0; i < d; i++) {
        const z = x.data[base + i] - mean;
        variance += z * z;
      }
      variance /= d;
      const inv = 1 / Math.sqrt(variance + 1e-5);
      const wbase = r * d;
      for (let i = 0; i < d; i++) data[base + i] = (x.data[base + i] - mean) * inv * weight.data[wbase + i] + bias.data[wbase + i];
    }
  }
  return { shape: x.shape.slice(), data };
}

function chunk2(t) {
  const d = t.shape[t.shape.length - 1];
  const half = d / 2;
  const rows = t.data.length / d;
  const a = new Float32Array(rows * half);
  const b = new Float32Array(rows * half);
  for (let r = 0; r < rows; r++) {
    a.set(t.data.subarray(r * d, r * d + half), r * half);
    b.set(t.data.subarray(r * d + half, (r + 1) * d), r * half);
  }
  const shape = t.shape.slice();
  shape[shape.length - 1] = half;
  return [{ shape: shape.slice(), data: a }, { shape: shape.slice(), data: b }];
}

function softmax(t) {
  const d = t.shape[t.shape.length - 1];
  const rows = t.data.length / d;
  const data = new Float32Array(t.data.length);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let i = 0; i < d; i++) max = Math.max(max, t.data[r * d + i]);
    let sum = 0;
    for (let i = 0; i < d; i++) {
      const e = Math.exp(t.data[r * d + i] - max);
      data[r * d + i] = e;
      sum += e;
    }
    for (let i = 0; i < d; i++) data[r * d + i] /= sum;
  }
  return { shape: t.shape.slice(), data };
}

function expand(t, target) {
  if (t.shape.every((v, i) => v === target[i])) return t;
  const inStride = new Array(t.shape.length);
  const outStride = new Array(target.length);
  let stride = 1;
  for (let i = t.shape.length - 1; i >= 0; i--) {
    inStride[i] = t.shape[i] === 1 ? 0 : stride;
    stride *= t.shape[i];
  }
  stride = 1;
  for (let i = target.length - 1; i >= 0; i--) {
    outStride[i] = stride;
    stride *= target[i];
  }
  const data = new Float32Array(numel(target));
  for (let i = 0; i < data.length; i++) {
    let rem = i;
    let src = 0;
    for (let d = 0; d < target.length; d++) {
      const coord = Math.floor(rem / outStride[d]);
      rem -= coord * outStride[d];
      if (t.shape[d] !== 1) src += coord * inStride[d];
    }
    data[i] = t.data[src];
  }
  return { shape: target.slice(), data };
}

function alignContext(context, x) {
  let c = context;
  while (c.shape.length < x.shape.length) c = unsqueeze(c, -2);
  const target = x.shape.slice(0, -1).concat([c.shape[c.shape.length - 1]]);
  return expand(c, target);
}

function grn(x, prefix, weights, context) {
  let hidden = linear(x, weights[`${prefix}.fc1.weight`], weights[`${prefix}.fc1.bias`]);
  if (weights[`${prefix}.fc_c.weight`]) {
    hidden = add(hidden, linear(alignContext(context, x), weights[`${prefix}.fc_c.weight`], null));
  }
  hidden = elu(hidden);
  hidden = linear(hidden, weights[`${prefix}.fc2.weight`], weights[`${prefix}.fc2.bias`]);
  const [a, b] = chunk2(linear(hidden, weights[`${prefix}.glu.weight`], weights[`${prefix}.glu.bias`]));
  let residual = x;
  if (weights[`${prefix}.skip.weight`]) residual = linear(x, weights[`${prefix}.skip.weight`], weights[`${prefix}.skip.bias`]);
  return layerNorm(add(residual, mul(a, sigmoid(b))), weights[`${prefix}.norm_w`], weights[`${prefix}.norm_b`]);
}

function batchedGrn(x, prefix, weights) {
  let hidden = elu(batchedLinear(x, weights[`${prefix}.w1`], weights[`${prefix}.b1`]));
  hidden = batchedLinear(hidden, weights[`${prefix}.w2`], weights[`${prefix}.b2`]);
  const [a, b] = chunk2(batchedLinear(hidden, weights[`${prefix}.wg`], weights[`${prefix}.bg`]));
  const residual = weights[`${prefix}.skip_w`]
    ? batchedLinear(x, weights[`${prefix}.skip_w`], weights[`${prefix}.skip_b`])
    : x;
  return layerNorm(add(residual, mul(a, sigmoid(b))), weights[`${prefix}.ln_g`], weights[`${prefix}.ln_b`]);
}

function sumSecondLast(t) {
  const d = t.shape[t.shape.length - 1];
  const n = t.shape[t.shape.length - 2];
  const groups = t.data.length / (n * d);
  const data = new Float32Array(groups * d);
  for (let g = 0; g < groups; g++) {
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d; k++) data[g * d + k] += t.data[(g * n + i) * d + k];
    }
  }
  const shape = t.shape.slice();
  shape.splice(shape.length - 2, 1);
  return { shape, data };
}

function vsn(values, prefix, weights, context) {
  let squeezed = false;
  if (values.shape.length === 3) {
    values = unsqueeze(values, 1);
    squeezed = true;
  }
  const [b, t, n, d] = values.shape;
  const flat = reshape(values, [b, t, n * d]);
  const weightsVar = softmax(grn(flat, `${prefix}.selector`, weights, context));
  const processed = batchedGrn(values, `${prefix}.var`, weights);
  const weighted = new Float32Array(processed.data.length);
  for (let i = 0; i < weightsVar.data.length; i++) {
    const wv = weightsVar.data[i];
    for (let k = 0; k < d; k++) weighted[i * d + k] = processed.data[i * d + k] * wv;
  }
  let selected = sumSecondLast({ shape: processed.shape, data: weighted });
  let w = weightsVar;
  if (squeezed) {
    selected = squeeze(selected, 1);
    w = squeeze(w, 1);
  }
  return [selected, w];
}

function numericEmbed(x, w, b) {
  const n = x.shape[x.shape.length - 1];
  const d = w.shape[1];
  const rows = x.data.length / n;
  const data = new Float32Array(rows * n * d);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < n; i++) {
      const xv = x.data[r * n + i];
      for (let k = 0; k < d; k++) data[(r * n + i) * d + k] = xv * w.data[i * d + k] + b.data[i * d + k];
    }
  }
  return { shape: x.shape.concat([d]), data };
}

function categorical(indices, prefix, weights) {
  const n = indices.shape[indices.shape.length - 1];
  const prefixShape = indices.shape.slice(0, -1);
  const count = numel(prefixShape);
  const cols = [];
  for (let i = 0; i < n; i++) {
    const table = weights[`${prefix}.embs.${i}.weight`];
    const d = table.shape[1];
    const col = new Float32Array(count * d);
    for (let r = 0; r < count; r++) {
      const row = indices.data[r * n + i];
      col.set(table.data.subarray(row * d, (row + 1) * d), r * d);
    }
    cols.push(unsqueeze({ shape: prefixShape.concat([d]), data: col }, -2));
  }
  return concat(cols, -2);
}

function lstm(x, h, c, weights) {
  const wih = weights["lstm.weight_ih_l0"];
  const whh = weights["lstm.weight_hh_l0"];
  const bih = weights["lstm.bias_ih_l0"];
  const bhh = weights["lstm.bias_hh_l0"];
  const tLen = x.shape[1];
  const steps = [];
  let hs = h;
  let cs = c;
  for (let t = 0; t < tLen; t++) {
    const xt = sliceTime(x, t);
    const gates = add(linear(xt, wih, bih), linear(hs, whh, bhh));
    const parts = chunk4(gates);
    const i = sigmoid(parts[0]);
    const f = sigmoid(parts[1]);
    const g = tanh(parts[2]);
    const o = sigmoid(parts[3]);
    cs = add(mul(f, cs), mul(i, g));
    hs = mul(o, tanh(cs));
    steps.push(hs);
  }
  return [stackTime(steps), hs, cs];
}

function sliceTime(t, index) {
  const [b, time, d] = t.shape;
  const data = new Float32Array(b * d);
  for (let row = 0; row < b; row++) data.set(t.data.subarray((row * time + index) * d, (row * time + index + 1) * d), row * d);
  return { shape: [b, d], data };
}

function stackTime(list) {
  const b = list[0].shape[0];
  const d = list[0].shape[1];
  const time = list.length;
  const data = new Float32Array(b * time * d);
  for (let row = 0; row < b; row++) {
    for (let t = 0; t < time; t++) data.set(list[t].data.subarray(row * d, (row + 1) * d), (row * time + t) * d);
  }
  return { shape: [b, time, d], data };
}

function chunk4(t) {
  const d = t.shape[t.shape.length - 1];
  const part = d / 4;
  const rows = t.data.length / d;
  return [0, 1, 2, 3].map((piece) => {
    const data = new Float32Array(rows * part);
    for (let r = 0; r < rows; r++) data.set(t.data.subarray(r * d + piece * part, r * d + (piece + 1) * part), r * part);
    const shape = t.shape.slice();
    shape[shape.length - 1] = part;
    return { shape, data };
  });
}

function einsumQ(x, w) {
  const [b, time, d] = x.shape;
  const heads = w.shape[0];
  const k = w.shape[2];
  const data = new Float32Array(b * heads * time * k);
  for (let bi = 0; bi < b; bi++) {
    for (let h = 0; h < heads; h++) {
      for (let t = 0; t < time; t++) {
        for (let ki = 0; ki < k; ki++) {
          let s = 0;
          const x0 = (bi * time + t) * d;
          for (let di = 0; di < d; di++) s += x.data[x0 + di] * w.data[(h * d + di) * k + ki];
          data[(((bi * heads + h) * time + t) * k) + ki] = s;
        }
      }
    }
  }
  return { shape: [b, heads, time, k], data };
}

function addHeadBias(q, bias) {
  const [b, h, time, k] = q.shape;
  const data = new Float32Array(q.data);
  for (let bi = 0; bi < b; bi++) {
    for (let hi = 0; hi < h; hi++) {
      for (let t = 0; t < time; t++) {
        for (let ki = 0; ki < k; ki++) data[(((bi * h + hi) * time + t) * k) + ki] += bias.data[hi * k + ki];
      }
    }
  }
  return { shape: q.shape.slice(), data };
}

function scoresOf(q, k) {
  const [b, h, time, kk] = q.shape;
  const sLen = k.shape[2];
  const data = new Float32Array(b * h * time * sLen);
  for (let bi = 0; bi < b; bi++) {
    for (let hi = 0; hi < h; hi++) {
      for (let t = 0; t < time; t++) {
        for (let s = 0; s < sLen; s++) {
          let acc = 0;
          const q0 = (((bi * h + hi) * time + t) * kk);
          const k0 = (((bi * h + hi) * sLen + s) * kk);
          for (let i = 0; i < kk; i++) acc += q.data[q0 + i] * k.data[k0 + i];
          data[(((bi * h + hi) * time + t) * sLen) + s] = acc;
        }
      }
    }
  }
  return { shape: [b, h, time, sLen], data };
}

function maskFill(scores, mask) {
  const tq = scores.shape[2];
  const ts = scores.shape[3];
  const heads = scores.shape[0] * scores.shape[1];
  const data = new Float32Array(scores.data);
  for (let h = 0; h < heads; h++) {
    for (let q = 0; q < tq; q++) {
      for (let s = 0; s < ts; s++) {
        if (!mask.data[q * ts + s]) data[(h * tq + q) * ts + s] = -1e9;
      }
    }
  }
  return { shape: scores.shape.slice(), data };
}

function meanHeads(t) {
  const [b, h, time, s] = t.shape;
  const data = new Float32Array(b * time * s);
  for (let bi = 0; bi < b; bi++) {
    for (let hi = 0; hi < h; hi++) {
      for (let i = 0; i < time * s; i++) data[bi * time * s + i] += t.data[((bi * h + hi) * time * s) + i];
    }
  }
  for (let i = 0; i < data.length; i++) data[i] /= h;
  return { shape: [b, time, s], data };
}

function matmul(a, b) {
  const [batch, m, k] = a.shape;
  const n = b.shape[2];
  const data = new Float32Array(batch * m * n);
  for (let bi = 0; bi < batch; bi++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let kk = 0; kk < k; kk++) s += a.data[(bi * m + i) * k + kk] * b.data[(bi * k + kk) * n + j];
        data[(bi * m + i) * n + j] = s;
      }
    }
  }
  return { shape: [batch, m, n], data };
}

function attention(query, context, mask, weights) {
  const q = addHeadBias(einsumQ(query, weights["attn.Wq"]), weights["attn.bq"]);
  const k = addHeadBias(einsumQ(context, weights["attn.Wk"]), weights["attn.bk"]);
  let scores = scale(scoresOf(q, k), 1 / Math.sqrt(weights["attn.Wq"].shape[2]));
  scores = maskFill(scores, mask);
  const attn = meanHeads(softmax(scores));
  const mixed = matmul(attn, linear(context, weights["attn.Wv.weight"], weights["attn.Wv.bias"]));
  return [linear(mixed, weights["attn.Wo.weight"], weights["attn.Wo.bias"]), attn];
}

function gatedResidual(update, skip, gatePrefix, normW, normB, weights) {
  const [a, b] = chunk2(linear(update, weights[`${gatePrefix}.weight`], weights[`${gatePrefix}.bias`]));
  return layerNorm(add(skip, mul(a, sigmoid(b))), weights[normW], weights[normB]);
}

export function loadModel(doc) {
  const weights = {};
  for (const [name, entry] of Object.entries(doc.weights)) weights[name] = decodeTensor(entry);
  const byName = Object.fromEntries(doc.features.map((feature) => [feature.name, feature]));
  const mean = {};
  const std = {};
  doc.scaler.names.forEach((name, i) => {
    mean[name] = doc.scaler.mean[i];
    std[name] = doc.scaler.std[i];
  });
  return { doc, weights, byName, mean, std };
}

export function forward(model, inputs) {
  const w = model.weights;
  const staticVars = concat([
    numericEmbed(inputs.static_num, w["static_num.w"], w["static_num.b"]),
    categorical(inputs.static_cat, "static_cat", w),
  ], -2);
  const [staticRep, wStatic] = vsn(staticVars, "vsn_static", w, null);
  const cS = grn(staticRep, "ctx_s", w, null);
  const cE = grn(staticRep, "ctx_e", w, null);
  let h = tanh(grn(staticRep, "ctx_h", w, null));
  let c = tanh(grn(staticRep, "ctx_c", w, null));
  const hist = concat([
    numericEmbed(inputs.hist_num, w["obs_num.w"], w["obs_num.b"]),
    categorical(inputs.hist_cat, "obs_cat", w),
    numericEmbed(inputs.known_num_hist, w["known_num.w"], w["known_num.b"]),
    categorical(inputs.known_cat_hist, "known_cat", w),
  ], -2);
  const future = concat([
    numericEmbed(inputs.known_num_fut, w["known_num.w"], w["known_num.b"]),
    categorical(inputs.known_cat_fut, "known_cat", w),
  ], -2);
  const [histSel, wHist] = vsn(hist, "vsn_hist", w, cS);
  const [futSel, wFut] = vsn(future, "vsn_future", w, cS);
  let enc;
  [enc, h, c] = lstm(histSel, h, c, w);
  const [dec] = lstm(futSel, h, c, w);
  const encE = grn(enc, "enrich", w, cE);
  const decE = grn(dec, "enrich", w, cE);
  const sequence = concat([encE, decE], 1);
  const [attnOut, attn] = attention(decE, sequence, w["attn_mask"], w);
  const skipped = gatedResidual(attnOut, decE, "gate1", "norm1_w", "norm1_b", w);
  const fed = grn(skipped, "ff", w, null);
  const fused = gatedResidual(fed, skipped, "gate2", "norm2_w", "norm2_b", w);
  const now = linear(sliceTime(encE, encE.shape[1] - 1), w["now_head.weight"], w["now_head.bias"]);
  const logits = linear(fused, w["head.weight"], w["head.bias"]);
  return { now, future: logits, attention: attn, wStatic, wHist, wFuture: wFut };
}

function normKey(key) {
  return String(key).trim().toLowerCase().replace(/ /g, "_").replace(/-/g, "_")
    .replace(/ā/g, "a").replace(/ē/g, "e").replace(/ī/g, "i").replace(/ō/g, "o").replace(/ū/g, "u");
}

function canonical(model, name, value) {
  let text;
  if (typeof value === "boolean") text = value ? "yes" : "no";
  else if (typeof value === "number" && Number.isInteger(value)) text = String(value);
  else text = String(value).trim();
  const alias = ALIASES[name] || {};
  const mapped = alias[text.toLowerCase()] || text;
  const vocab = model.doc.vocabs[name];
  if (vocab.includes(mapped)) return mapped;
  const found = vocab.find((option) => option.toLowerCase() === text.toLowerCase() || option.toLowerCase() === mapped.toLowerCase());
  return found || mapped;
}

function catIndex(model, name, value) {
  const key = canonical(model, name, value);
  const at = model.doc.vocabs[name].indexOf(key);
  return at < 0 ? 0 : at + 1;
}

function names(model, role, kind) {
  return model.doc.features.filter((feature) => feature.kind === kind && (feature.role === role || (role === "observed" && feature.role === "outcome"))).map((feature) => feature.name);
}

function numVec(model, row, keys) {
  const data = new Float32Array(keys.length);
  keys.forEach((name, i) => {
    data[i] = (Number(row[name]) - model.mean[name]) / model.std[name];
  });
  return data;
}

function catVec(model, row, keys) {
  const data = new Int32Array(keys.length);
  keys.forEach((name, i) => { data[i] = catIndex(model, name, row[name]); });
  return data;
}

function stack(rows, integer) {
  const n = rows[0].length;
  const data = integer ? new Int32Array(rows.length * n) : new Float32Array(rows.length * n);
  rows.forEach((row, i) => data.set(row, i * n));
  return { shape: [1, rows.length, n], data };
}

function labelOf(model, timeIdx) {
  const entry = model.doc.national && model.doc.national[timeIdx];
  if (entry && entry.month) return entry.month;
  const year = 2019 + Math.floor((timeIdx - 1) / 12);
  return `${SHORT[(timeIdx - 1) % 12]} ${year}`;
}

function coerce(model, raw) {
  const lowered = {};
  Object.entries(raw).forEach(([key, value]) => { lowered[normKey(key)] = value; });
  const missing = model.doc.features.map((feature) => feature.name).filter((name) => !(name in lowered));
  if (missing.length) {
    const shown = missing.slice(0, 8).join(", ");
    throw new Error(`missing features: ${shown}${missing.length > 8 ? ` and ${missing.length - 8} more` : ""}`);
  }
  const notes = [];
  const row = {};
  model.doc.features.forEach((feature) => {
    const value = lowered[feature.name];
    if (value === null || value === "") {
      if (feature.kind === "numeric") {
        row[feature.name] = model.mean[feature.name];
        notes.push(`${feature.name} was empty and set to the training average`);
      } else {
        row[feature.name] = "";
        notes.push(`${feature.name} was empty and treated as unknown`);
      }
      return;
    }
    if (feature.kind === "numeric") {
      const number = typeof value === "boolean" ? (value ? 1 : 0) : Number(String(value).replace(/,/g, "").replace("%", ""));
      if (!Number.isFinite(number)) throw new Error(`${feature.name} is not a number`);
      row[feature.name] = number;
    } else {
      row[feature.name] = canonical(model, feature.name, value);
      if (!model.doc.vocabs[feature.name].includes(row[feature.name])) {
        notes.push(`${feature.name} value ${JSON.stringify(value)} is outside the training vocabulary and was treated as unknown`);
      }
    }
  });
  ["whanau_id", "time_idx", "year", "month"].forEach((key) => {
    if (lowered[key] !== undefined && lowered[key] !== null && lowered[key] !== "") row[key] = lowered[key];
  });
  if ("time_idx" in row) row.time_idx = Number(row.time_idx);
  if ("year" in row) row.year = Number(row.year);
  if ("month" in row) row.month = Number(row.month);
  return { row, notes };
}

function originOf(row) {
  if (row.time_idx !== undefined && row.time_idx !== null && row.time_idx !== "") return Number(row.time_idx);
  if (row.year && row.month) return (Number(row.year) - 2021) * 12 + Number(row.month) - 1;
  return null;
}

function projectFuture(model, last, origin) {
  const rows = [];
  for (let step = 1; step <= 12; step++) {
    let month;
    let benefit;
    let places;
    const lease0 = Number(last.lease_months_remaining);
    const lease = lease0 >= 99 ? 99 : Math.max(0, lease0 - step);
    let school;
    let holidays;
    let quarter;
    if (origin !== null && model.doc.national && model.doc.national[origin + step]) {
      const macro = model.doc.national[origin + step];
      month = macro.month_of_year;
      benefit = macro.benefit_indexation;
      school = macro.school_holiday;
      holidays = macro.public_holidays;
      quarter = macro.quarter;
    } else {
      month = ((Number(last.month_of_year) - 1 + step) % 12) + 1;
      benefit = month === 4 ? 1 : 0;
      quarter = Math.floor((month - 1) / 3) + 1;
      school = [1, 4, 7, 9, 10, 12].includes(month) ? 1 : 0;
      holidays = { 1: 3, 2: 1, 4: 3, 6: 1, 10: 1, 12: 2 }[month] || 0;
    }
    rows.push({
      month_of_year: month,
      quarter,
      school_holiday: school,
      public_holidays: holidays,
      benefit_indexation: benefit,
      planned_support_places: Number(last.planned_support_places),
      lease_months_remaining: lease,
    });
  }
  return rows;
}

export function encodeCase(model, history, futureKnown) {
  const notes = [];
  let rows = history.map((raw) => {
    const out = coerce(model, raw);
    notes.push(...out.notes);
    return out.row;
  });
  if (rows.some((row) => row.time_idx !== undefined)) rows.sort((a, b) => Number(a.time_idx || 0) - Number(b.time_idx || 0));
  const provided = rows.length;
  const padded = provided < 12;
  if (provided > 12) {
    rows = rows.slice(-12);
    notes.push("used the latest 12 months and left the earlier ones out");
  } else if (padded) {
    rows = Array.from({ length: 12 - provided }, () => ({ ...rows[0] })).concat(rows);
    notes.push(`only ${provided} month${provided === 1 ? "" : "s"} were uploaded, so the earliest month was repeated to fill the 12-month encoder. Treat the forecast with less confidence.`);
  }
  const origin = originOf(rows[rows.length - 1]);
  let future;
  if (futureKnown) {
    if (futureKnown.length !== 12) throw new Error("future_known must contain 12 months");
    future = futureKnown.map((raw) => {
      const out = coerce(model, raw);
      notes.push(...out.notes);
      return out.row;
    });
  } else {
    future = projectFuture(model, rows[rows.length - 1], origin);
    if (origin === null) notes.push("no time index was supplied, so planned support places are held at the last value and the April indexation rate uses the generic 0.04");
    notes.push("future lease months count down from the last uploaded month and do not assume a renewal");
  }
  const staticNum = names(model, "static", "numeric");
  const staticCat = names(model, "static", "categorical");
  const obsNum = names(model, "observed", "numeric");
  const obsCat = names(model, "observed", "categorical");
  const knownNum = names(model, "known", "numeric");
  const knownCat = names(model, "known", "categorical");
  const inputs = {
    static_num: { shape: [1, staticNum.length], data: numVec(model, rows[rows.length - 1], staticNum) },
    static_cat: { shape: [1, staticCat.length], data: catVec(model, rows[rows.length - 1], staticCat) },
    hist_num: stack(rows.map((row) => numVec(model, row, obsNum)), false),
    hist_cat: stack(rows.map((row) => catVec(model, row, obsCat)), true),
    known_num_hist: stack(rows.map((row) => numVec(model, row, knownNum)), false),
    known_cat_hist: stack(rows.map((row) => catVec(model, row, knownCat)), true),
    known_num_fut: stack(future.map((row) => numVec(model, row, knownNum)), false),
    known_cat_fut: stack(future.map((row) => catVec(model, row, knownCat)), true),
  };
  const historyLabels = rows.map((row, index) => {
    if (padded && index < 12 - provided) return "padded";
    const originRow = originOf(row);
    return originRow === null ? `month ${index + 1}` : labelOf(model, originRow);
  });
  const futureLabels = future.map((_, step) => (origin === null ? `+${step + 1} months` : labelOf(model, origin + step + 1)));
  return {
    inputs,
    info: {
      notes,
      padded,
      months_provided: provided,
      origin,
      origin_label: origin === null ? null : labelOf(model, origin),
      history_labels: historyLabels,
      future_labels: futureLabels,
      whanau_id: rows[rows.length - 1].whanau_id || null,
    },
  };
}

function pack(probs, tiers) {
  let best = 0;
  probs.forEach((p, i) => { if (p > probs[best]) best = i; });
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  return {
    class_index: best,
    tier_id: tiers[best].id,
    name: tiers[best].name,
    short: tiers[best].short,
    probability: probs[best],
    margin: order[0][0] - order[1][0],
    low_confidence: probs[best] < 0.4,
    probabilities: tiers.map((tier, i) => ({ tier_id: tier.id, name: tier.name, short: tier.short, probability: probs[i] })),
  };
}

function softmaxRow(data, offset, width) {
  let max = -Infinity;
  for (let i = 0; i < width; i++) max = Math.max(max, data[offset + i]);
  const out = new Array(width);
  let sum = 0;
  for (let i = 0; i < width; i++) {
    out[i] = Math.exp(data[offset + i] - max);
    sum += out[i];
  }
  return out.map((v) => v / sum);
}

function topWeights(model, nameList, values, limit) {
  return values.map((weight, index) => ({ weight, index }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, limit)
    .map(({ weight, index }) => {
      const feature = model.byName[nameList[index]];
      return {
        name: feature.name,
        domain: feature.domain,
        role: feature.role,
        sensitive: feature.sensitive,
        description: feature.description,
        weight,
      };
    });
}

export function classify(model, payload) {
  let history;
  let future = null;
  if (Array.isArray(payload)) history = payload;
  else if (payload && Array.isArray(payload.history)) {
    history = payload.history;
    future = payload.future_known || null;
  } else if (payload && typeof payload === "object") history = [payload];
  else throw new Error("JSON must be one month, a list of months, or an object with a history list");
  if (!history.length) throw new Error("history is empty");
  const { inputs, info } = encodeCase(model, history, future);
  const pred = forward(model, inputs);
  const tiers = model.doc.tiers;
  const now = pack(softmaxRow(pred.now.data, 0, 5), tiers);
  const path = [];
  for (let step = 0; step < 12; step++) {
    const packed = pack(softmaxRow(pred.future.data, step * 5, 5), tiers);
    packed.step = step + 1;
    packed.label = info.future_labels[step];
    path.push(packed);
  }
  const histNames = model.doc.hist_names;
  const histMean = new Array(histNames.length).fill(0);
  const time = pred.wHist.shape[1];
  const nHist = pred.wHist.shape[2];
  for (let t = 0; t < time; t++) {
    for (let i = 0; i < nHist; i++) histMean[i] += pred.wHist.data[t * nHist + i];
  }
  for (let i = 0; i < nHist; i++) histMean[i] /= time;
  const futureStep = [];
  const nFut = pred.wFuture.shape[2];
  for (let i = 0; i < nFut; i++) futureStep.push(pred.wFuture.data[5 * nFut + i]);
  const attention = Array.from(pred.attention.data.subarray(5 * pred.attention.shape[2], 6 * pred.attention.shape[2]));
  const notes = info.notes.slice();
  const primary = path[5];
  if (primary.low_confidence) notes.push("the 6-month probability is under 0.40, so this is a weak call rather than a firm classification");
  else if (primary.margin < 0.08) notes.push("the top two tiers are close at the 6-month horizon");
  const sensitive = topWeights(model, histNames, histMean, 8).filter((row) => row.sensitive).map((row) => row.name);
  if (sensitive.length) {
    notes.push(`sensitive inputs (${sensitive.join(", ")}) are among the stronger variable-selection weights for this history. That is an association in the synthetic panel, not a reason to treat those inputs as causes.`);
  }
  return {
    primary_horizon_months: 6,
    classification: primary,
    nowcast: now,
    forecasts: { 3: path[2], 6: path[5], 12: path[11] },
    path,
    attention: { horizon_months: 6, labels: info.history_labels.concat(info.future_labels), weights: attention },
    variables: {
      history: topWeights(model, histNames, histMean, 8),
      static: topWeights(model, model.doc.static_names, Array.from(pred.wStatic.data), 5),
      future: topWeights(model, model.doc.future_names, futureStep, 5),
    },
    history_months_provided: info.months_provided,
    padded: info.padded,
    origin: info.origin,
    origin_label: info.origin_label,
    whanau_id: info.whanau_id,
    notes,
    disclaimer: "Synthetic-panel research prototype. This is not a determination of homelessness, eligibility, or who should receive support.",
  };
}

export function fromEncoded(model, encoded) {
  const inputs = {};
  for (const [key, entry] of Object.entries(encoded)) inputs[key] = decodeTensor(entry);
  return forward(model, inputs);
}
