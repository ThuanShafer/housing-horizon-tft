import { classify, loadModel } from "./tft.js";

const TIER_COLOR = {
  1: "var(--severe)",
  2: "var(--unstable)",
  3: "var(--risk)",
  4: "var(--subsidised)",
  5: "var(--stable)",
};

const DOMAIN_CLASS = {
  "Housing": "d-housing",
  "Temporal": "d-temporal",
  "Demographic": "d-demographic",
  "Area": "d-area",
  "Household": "d-household",
  "Socio-economic": "d-socio",
  "Service / Policy": "d-service",
  "Health": "d-health",
  "Justice": "d-justice",
  "Behavioural": "d-behaviour",
  "Macro & Events": "d-macro",
};

const scoreBody = document.querySelector("#score-body");
const matrixBody = document.querySelector("#matrix-body");
const driversBody = document.querySelector("#drivers-body");
const cohortBody = document.querySelector("#cohort-body");
const dictBody = document.querySelector("#dict-body");
const dictNote = document.querySelector("#dict-note");
const result = document.querySelector("#result");
const jsonBox = document.querySelector("#json");
const formError = document.querySelector("#form-error");
const exampleRow = document.querySelector("#example-row");

let model;
let metrics;
let cohort = [];
let examples = [];
let activeExample = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmt(value) {
  return Number(value).toFixed(3);
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function clear(node) {
  node.replaceChildren();
}

const seedPick = document.querySelector("#seed-pick");
const seedValue = document.querySelector("#seed-value");
const seedStatus = document.querySelector("#seed-status");

function bundleBase(seed) {
  return Number(seed) === 42 ? "model" : `model/seeds/${seed}`;
}

async function loadBundle(seed) {
  const base = bundleBase(seed);
  const stamp = Date.now();
  const [modelDoc, metricsDoc, cohortDoc, exampleDoc] = await Promise.all(
    ["model.json", "metrics.json", "cohort.json", "examples.json"].map((name) =>
      fetch(`${base}/${name}?t=${stamp}`).then((res) => {
        if (!res.ok) throw new Error(`Could not read the seed ${seed} bundle.`);
        return res.json();
      }),
    ),
  );
  model = loadModel(modelDoc);
  metrics = metricsDoc;
  cohort = cohortDoc;
  examples = exampleDoc;
  renderScore();
  renderDrivers();
  renderCohort();
  renderDictionary();
  renderExamples();
  showExample(examples[0]);
  const fit = metrics.families;
  seedStatus.textContent = `Fit ${fit.fit}, validation ${fit.validation}, test ${fit.test}.`;
}

let seedTicket = 0;

async function useSeed(seed) {
  const ticket = ++seedTicket;
  const previous = metrics ? String(metrics.seed) : "42";
  seedPick.disabled = true;
  seedStatus.textContent = Number(seed) === 42
    ? "Loading seed 42…"
    : `Training and testing seed ${seed}. The weights are fit again on a new 240 / 60 / 200 split.`;
  try {
    if (Number(seed) !== 42) {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: Number(seed) }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "Training failed.");
    }
    if (ticket !== seedTicket) return;
    await loadBundle(seed);
  } catch (error) {
    if (ticket !== seedTicket) return;
    seedStatus.textContent = error.message;
    seedPick.value = previous;
    seedValue.textContent = previous;
  } finally {
    if (ticket === seedTicket) seedPick.disabled = false;
  }
}

async function boot() {
  await loadBundle(42);
  seedPick.addEventListener("input", () => {
    seedValue.textContent = seedPick.value;
  });
  seedPick.addEventListener("change", () => {
    if (seedPick.disabled) return;
    if (metrics && Number(seedPick.value) === Number(metrics.seed)) return;
    useSeed(seedPick.value);
  });
}

function renderScore() {
  clear(scoreBody);
  const test = metrics.test_origin;
  const base = metrics.baselines_h6;
  const stats = el("div", "stats");
  [
    ["Nowcast macro-F1", test.now.macro_f1, `Latest month, ${metrics.headline_origin.label}`],
    ["Six-month macro-F1", test.h6.macro_f1, "Six months after that snapshot"],
    ["No-change macro-F1", base.persistence.macro_f1, "Carry the snapshot tier forward"],
    ["Linear with changes", base.last_month_plus_deltas_linear.macro_f1, "Last month plus 3- and 6-month deltas"],
  ].forEach(([label, value, note], index) => {
    const card = el("div", index === 1 ? "stat mark" : "stat");
    card.append(el("span", null, label), el("b", null, fmt(value)), el("span", null, note));
    stats.append(card);
  });
  scoreBody.append(stats);
  if (metrics.performance_h6) scoreBody.append(metricBlock(metrics.performance_h6));

  const lead = el("p");
  const correct = cohort.filter((row) => row.predicted.h6 === row.actual.h6).length;
  const unchanged = cohort.filter((row) => row.actual.now === row.actual.h6).length;
  const changers = cohort.length - unchanged;
  const caught = cohort.filter((row) => row.actual.now !== row.actual.h6 && row.predicted.h6 === row.actual.h6).length;
  lead.textContent = `On ${cohort.length} held-out whānau, looking six months past ${metrics.headline_origin.label}, the tier is right for ${correct}. ${unchanged} do not change tier, so carrying that month forward scores ${fmt(base.persistence.macro_f1)} macro-F1 (${pct(base.persistence.accuracy)} accuracy) against the transformer at ${fmt(test.h6.macro_f1)} (${pct(test.h6.accuracy)}). Of the ${changers} who change, the transformer calls ${caught}. A last-month linear model scores ${fmt(base.last_month_linear.macro_f1)}; with 3- and 6-month changes it scores ${fmt(base.last_month_plus_deltas_linear.macro_f1)}.`;
  scoreBody.append(lead);

  const table = el("table");
  const head = el("tr");
  ["Tier", "Support", "Recall", "F1"].forEach((label, i) => head.append(el("th", i ? "n" : null, label)));
  table.append(el("thead").appendChild(head).parentNode);
  const body = el("tbody");
  test.h6.labels.forEach((label, i) => {
    const row = el("tr");
    row.append(el("td", null, label), el("td", "n", String(test.h6.support[i])), el("td", "n", pct(test.h6.recall[i])), el("td", "n", fmt(test.h6.f1[i])));
    body.append(row);
  });
  table.append(body);
  scoreBody.append(el("h3", null, "Recall by tier at six months"), table);
  scoreBody.append(el("p", "section-note", `Same 200 whānau: nowcast macro-F1 ${fmt(test.now.macro_f1)}, +3 months ${fmt(test.h3.macro_f1)}, +12 months ${fmt(test.h12.macro_f1)}. Severe deprivation has single-digit support here, so that recall is a small count.`));
  clear(matrixBody);
  matrixBody.append(matrix(test.h6.confusion, ["Severe", "Unstable", "At risk", "Assisted", "Stable"]));

  const fit = metrics.families;
  const through = metrics.headline_origin.forecast_through || metrics.headline_origin.forecast_year;
  const seedText = `Seed ${metrics.seed} · ${fit.train_pool} train / ${fit.test} test · seed ${metrics.inner_seed} holds out ${fit.validation} for early stopping · fit on ${fit.fit}. Window ${metrics.headline_origin.history}, forecast through ${through}.`;
  scoreBody.append(el("p", "section-note", seedText));
}

function metricBlock(perf) {
  const wrap = el("div", "metric-block");
  const grid = el("div", "metrics");
  [
    ["Accuracy", perf.accuracy],
    ["Precision", perf.precision_macro],
    ["Recall", perf.recall_macro],
    ["Specificity", perf.specificity_macro],
    ["F1", perf.f1_macro],
    ["AUC-ROC", perf.auc_macro],
  ].forEach(([label, value]) => {
    const card = el("div", "metric");
    card.append(el("span", null, label), el("b", null, value === null || value === undefined ? "—" : fmt(value)));
    grid.append(card);
  });
  wrap.append(grid);
  wrap.append(el("p", "section-note", "Macro averages across the five tiers. Recall is sensitivity. Specificity and AUC are one-versus-rest. A class with no cases in this snapshot has no AUC."));
  wrap.append(rocSvg(perf.roc));
  return wrap;
}

function rocSvg(curves) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 220 220");
  svg.setAttribute("class", "roc");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "ROC curves for the six-month forecast");
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "path");
  axis.setAttribute("d", "M30 10 V190 H210");
  axis.setAttribute("class", "axis");
  svg.append(axis);
  const chance = document.createElementNS("http://www.w3.org/2000/svg", "path");
  chance.setAttribute("d", "M30 190 L210 10");
  chance.setAttribute("class", "chance");
  svg.append(chance);
  const tone = getComputedStyle(document.documentElement);
  const colors = ["--severe", "--unstable", "--risk", "--subsidised", "--stable"].map((name) => tone.getPropertyValue(name).trim());
  const legend = el("div", "roc-legend");
  curves.forEach((curve, index) => {
    if (!curve.points || !curve.points.length) return;
    const d = curve.points.map(([fpr, tpr], i) => `${i ? "L" : "M"}${30 + fpr * 180} ${190 - tpr * 180}`).join(" ");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", colors[index % colors.length]);
    path.setAttribute("stroke-width", "1.6");
    svg.append(path);
    legend.append(el("span", null, `${curve.name} ${curve.auc === null || curve.auc === undefined ? "n/a" : fmt(curve.auc)}`));
  });
  const x = document.createElementNS("http://www.w3.org/2000/svg", "text");
  x.setAttribute("x", "120");
  x.setAttribute("y", "208");
  x.setAttribute("class", "tick");
  x.textContent = "False positive rate";
  const y = document.createElementNS("http://www.w3.org/2000/svg", "text");
  y.setAttribute("x", "8");
  y.setAttribute("y", "16");
  y.setAttribute("class", "tick");
  y.textContent = "True positive rate";
  svg.append(x, y);
  const block = el("div");
  block.append(svg, legend);
  return block;
}

function matrix(confusion, labels) {
  const wrap = el("div", "matrix");
  const max = Math.max(...confusion.flat(), 1);
  wrap.append(el("div", "h", ""));
  labels.forEach((label) => wrap.append(el("div", "h", label)));
  confusion.forEach((row, i) => {
    wrap.append(el("div", "lab", labels[i]));
    row.forEach((count) => {
      const cell = el("div", null, String(count));
      const share = count / max;
      const tone = getComputedStyle(document.documentElement);
      const [r, g, b] = tone.getPropertyValue("--signal").trim().replace("#", "").match(/../g).map((part) => parseInt(part, 16));
      cell.style.background = `rgba(${r}, ${g}, ${b}, ${0.16 + 0.72 * share})`;
      cell.style.color = share > 0.62 ? tone.getPropertyValue("--bg").trim() : tone.getPropertyValue("--ink").trim();
      wrap.append(cell);
    });
  });
  return wrap;
}

function renderDrivers(output) {
  clear(driversBody);
  if (output) {
    driversBody.append(el("p", "section-note", "Weights for the whānau on the right. Amber marks a sensitive input. These are associations, not causes."));
    driversBody.append(barBlock("This history", output.variables.history.slice(0, 6)));
    driversBody.append(barBlock("Static inputs for this whānau", output.variables.static.slice(0, 4)));
  }
  const hist = metrics.importance.history;
  const lead = hist.slice(0, 3).map((row) => row.name.replaceAll("_", " ")).join(", ");
  driversBody.append(el("p", "section-note", `Across all 200 test histories the encoder leans on ${lead}.`));
  const eth = metrics.importance.static.find((row) => row.name === "ethnicity");
  if (eth) driversBody.append(el("p", "section-note", `Ethnicity’s average static weight is ${fmt(eth.weight)}. The generator does not use ethnicity as a cause of the tier.`));
  const ab = metrics.ablation_h6;
  driversBody.append(el("p", "section-note", `Clearing gambling and justice inputs, without retraining, moves six-month macro-F1 from ${fmt(ab.full.macro_f1)} to ${fmt(ab.without_gambling_and_justice.macro_f1)}.`));
}

function barBlock(title, rows) {
  const wrap = el("div");
  wrap.append(el("h3", null, title));
  const bars = el("div", "bars");
  const max = Math.max(...rows.map((row) => row.weight));
  rows.forEach((row) => {
    const line = el("div", row.sensitive ? "bar-row sensitive" : "bar-row");
    line.append(el("span", null, row.name));
    const track = el("div", "bar");
    const fill = document.createElement("i");
    fill.style.width = `${(row.weight / max) * 100}%`;
    fill.style.background = domainColor(row.domain);
    track.append(fill);
    line.append(track, el("span", "num", fmt(row.weight)));
    bars.append(line);
  });
  wrap.append(bars);
  return wrap;
}

function renderCohort() {
  clear(cohortBody);
  cohortBody.append(tierCompare());
  const filters = el("div", "filters");
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Filter test whānau");
  ["All whānau", "Tier changed", "Forecast missed", "Forecast matched"].forEach((label) => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    select.append(option);
  });
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search id or local board";
  search.setAttribute("aria-label", "Search test whānau");
  filters.append(select, search);
  const count = el("p", "section-note");
  const scroller = el("div", "scroll");
  const table = el("table");
  const head = el("tr");
  ["Whānau", "Board", "Ethnicity", "Dec 2024", "June actual", "June forecast", "P"].forEach((label, i) => {
    head.append(el("th", i === 6 ? "n" : null, label));
  });
  table.append(el("thead").appendChild(head).parentNode);
  const body = el("tbody");
  table.append(body);
  scroller.append(table);
  cohortBody.append(filters, count, scroller);

  function paint() {
    clear(body);
    const q = search.value.trim().toLowerCase();
    const mode = select.value;
    const rows = cohort.filter((row) => {
      if (q && !`${row.id} ${row.local_board}`.toLowerCase().includes(q)) return false;
      const changed = row.actual.now !== row.actual.h6;
      const hit = row.predicted.h6 === row.actual.h6;
      if (mode === "Tier changed") return changed;
      if (mode === "Forecast missed") return !hit;
      if (mode === "Forecast matched") return hit;
      return true;
    });
    count.textContent = `${rows.length} shown`;
    rows.forEach((row) => {
      const tr = el("tr");
      const hit = row.predicted.h6 === row.actual.h6;
      tr.append(
        el("td", null, row.id),
        el("td", null, row.local_board),
        el("td", null, row.ethnicity),
        el("td", null, tierShort(row.actual.now)),
        el("td", null, tierShort(row.actual.h6)),
        el("td", hit ? "hit" : "miss", tierShort(row.predicted.h6)),
        el("td", "n", pct(row.probability_h6[row.predicted.h6 - 1])),
      );
      body.append(tr);
    });
  }
  select.addEventListener("change", paint);
  search.addEventListener("input", paint);
  paint();
}

function tierCompare() {
  const actual = [0, 0, 0, 0, 0];
  const predicted = [0, 0, 0, 0, 0];
  cohort.forEach((row) => {
    actual[row.actual.h6 - 1] += 1;
    predicted[row.predicted.h6 - 1] += 1;
  });
  const max = Math.max(...actual, ...predicted, 1);
  const wrap = el("div", "tier-bars");
  model.doc.tiers.forEach((tier, index) => {
    const row = el("div", "row");
    row.append(el("span", null, tier.short));
    ["actual", "pred"].forEach((kind) => {
      const count = kind === "actual" ? actual[index] : predicted[index];
      const meter = el("div", `meter ${kind}`);
      const fill = document.createElement("i");
      fill.style.width = `${(count / max) * 100}%`;
      fill.style.background = kind === "actual" ? "var(--accent-2)" : TIER_COLOR[tier.id];
      meter.append(fill);
      meter.title = `${count}`;
      row.append(meter);
    });
    wrap.append(row);
  });
  const key = el("div", "chart-key");
  key.append(el("span", "a", "June actual"), el("span", "p", "June forecast"));
  const block = el("div");
  block.append(wrap, key);
  return block;
}

function tierShort(id) {
  return model.doc.tiers[id - 1].short;
}

function domainColor(name) {
  const variable = {
    "Housing": "--d-housing",
    "Temporal": "--d-temporal",
    "Demographic": "--d-demographic",
    "Area": "--d-area",
    "Household": "--d-household",
    "Socio-economic": "--d-socio",
    "Service / Policy": "--d-service",
    "Health": "--d-health",
    "Justice": "--d-justice",
    "Behavioural": "--d-behaviour",
    "Macro & Events": "--d-macro",
  }[name];
  return variable ? `var(${variable})` : "var(--signal)";
}

function renderDictionary() {
  dictNote.textContent = "Domains and roles are the workbook’s Column_Domains sheet. Outcome components define the tier. Ethnicity is audit-only. Major event context is not a predictor.";
  const legend = document.querySelector("#domain-legend");
  clear(legend);
  clear(dictBody);
  const domains = [];
  model.doc.features.forEach((feature) => {
    let group = domains.find((item) => item.name === feature.domain);
    if (!group) {
      group = { name: feature.domain, features: [] };
      domains.push(group);
    }
    group.features.push(feature);
  });
  const grid = el("div", "domain-grid");
  domains.forEach((group) => {
    const swatch = el("i", `swatch ${DOMAIN_CLASS[group.name] || ""}`);
    const label = el("span");
    label.append(swatch, document.createTextNode(`${group.name} · ${group.features.length}`));
    legend.append(label);
    const block = el("div");
    const title = el("h3");
    title.append(el("i", `swatch ${DOMAIN_CLASS[group.name] || ""}`), document.createTextNode(group.name));
    block.append(title);
    group.features.forEach((feature) => {
      const chip = el("span", `chip ${DOMAIN_CLASS[feature.domain] || ""}`);
      chip.textContent = feature.name;
      chip.title = `${feature.role} · ${feature.description}`;
      block.append(chip);
    });
    grid.append(block);
  });
  dictBody.append(grid);
}

function renderExamples() {
  clear(exampleRow);
  examples.forEach((example, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.id;
    button.addEventListener("click", () => showExample(example));
    exampleRow.append(button);
    example._button = button;
    if (index === 0) button.setAttribute("aria-pressed", "true");
  });
}

function showExample(example) {
  activeExample = example;
  exampleRow.querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", button.textContent === example.id ? "true" : "false"));
  const payload = { history: example.history };
  jsonBox.value = JSON.stringify(payload, null, 2);
  formError.textContent = "";
  paintResult(classify(model, payload), example);
}

function paintResult(output, example) {
  renderDrivers(output);
  const top = output.variables.history[0];
  document.querySelector("#flow-weights").textContent = output.variables.history.slice(0, 3).map((row) => row.name.replaceAll("_", " ")).join(" · ");
  document.querySelector("#flow-out").textContent = `${output.classification.short}, ${pct(output.classification.probability)}. Latest month: ${output.nowcast.short}.`;
  const peak = output.attention.weights.map((weight, index) => [weight, output.attention.labels[index]]).sort((a, b) => b[0] - a[0])[0];
  document.querySelector("#flow-attend").textContent = peak ? `Strongest month in this case: ${peak[1]}.` : "Attention over the 24 months.";
  document.querySelector("#flow-input").textContent = top ? `This case leans first on ${top.name.replaceAll("_", " ")}.` : "Static identity and place, the observed past, and a known future calendar.";
  clear(result);
  result.append(el("p", "kicker", example ? `Held-out example ${example.id}` : (output.whanau_id || "Uploaded history")));
  const pair = el("div", "pair");
  pair.append(tierCard("Six-month forecast", output.classification), tierCard("Latest month", output.nowcast));
  result.append(pair);
  if (output.origin_label) {
    result.append(el("p", null, `History ends ${output.origin_label}. The forecast months are ${output.forecasts["3"].label}, ${output.forecasts["6"].label} and ${output.forecasts["12"].label}.`));
  }
  if (example) {
    const actual = model.doc.tiers[example.actual.h6 - 1].name;
    const hit = example.actual.h6 === output.classification.tier_id;
    result.append(el("p", null, `In the workbook, the six-month tier is ${actual}. The forecast ${hit ? "matches" : "does not match"} that outcome.`));
  }

  result.append(el("h3", null, "Six-month probabilities"));
  const probs = el("div", "prob");
  output.classification.probabilities.forEach((row) => {
    const line = el("div", "prob-row");
    line.append(el("span", null, row.short));
    const track = el("div", "track");
    const fill = document.createElement("i");
    fill.style.width = `${row.probability * 100}%`;
    fill.style.background = TIER_COLOR[row.tier_id];
    track.append(fill);
    line.append(track, el("span", "num", pct(row.probability)));
    probs.append(line);
  });
  result.append(probs);

  result.append(el("h3", null, "Path of the next twelve months"));
  const path = el("div", "path");
  output.path.forEach((step) => {
    const cell = el("div", step.step === 3 || step.step === 6 || step.step === 12 ? "key" : "");
    cell.style.borderTop = `4px solid ${TIER_COLOR[step.tier_id]}`;
    cell.append(el("strong", null, step.label), document.createTextNode(step.short));
    path.append(cell);
  });
  result.append(path);

  result.append(el("h3", null, "Attention for the six-month query"));
  const attn = el("div", "attn");
  const max = Math.max(...output.attention.weights);
  output.attention.weights.forEach((weight, index) => {
    const bar = el("span");
    bar.style.height = `${Math.max(4, (weight / max) * 100)}%`;
    bar.style.background = index < 12 ? "var(--accent)" : "var(--risk)";
    bar.title = `${output.attention.labels[index]} ${weight.toFixed(3)}`;
    attn.append(bar);
  });
  result.append(attn);
  result.append(el("p", "section-note", "Teal bars are the twelve history months. Ochre bars are the twelve forecast months."));

  result.append(el("h3", null, "Strongest history inputs for this case"));
  const list = el("ul", "notes");
  output.variables.history.slice(0, 5).forEach((row) => {
    list.append(el("li", null, `${row.name} · ${row.domain} · weight ${fmt(row.weight)}`));
  });
  result.append(list);
  if (output.notes.length) {
    const notesScroll = el("div", "scroll notes-scroll");
    const notes = el("ul", "notes");
    output.notes.forEach((note) => notes.append(el("li", null, note)));
    notesScroll.append(notes);
    result.append(notesScroll);
  }
  result.append(el("p", "section-note", output.disclaimer));
}

function tierCard(label, packed) {
  const card = el("div", "card");
  card.style.borderTopColor = TIER_COLOR[packed.tier_id];
  card.append(el("p", "kicker", label), el("p", "tier-name", packed.name), el("p", null, `${pct(packed.probability)} · margin ${fmt(packed.margin)}`));
  if (packed.low_confidence) card.append(el("p", "section-note", "Low score. A person should look at this case."));
  return card;
}

function payloadFromText(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Paste JSON or CSV first.");
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) return JSON.parse(trimmed);
  return csvToPayload(trimmed);
}

function csvToPayload(text) {
  const rows = parseCsv(text);
  if (rows.length < 2) throw new Error("CSV needs a header row and at least one month.");
  const header = rows[0].map((name) => name.trim());
  const history = rows.slice(1).map((cells) => {
    const record = {};
    header.forEach((name, index) => { if (name) record[name] = cells[index] ?? ""; });
    return record;
  });
  return history.length === 1 ? history[0] : { history };
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === "\"" && text[i + 1] === "\"") { cell += "\""; i += 1; }
      else if (char === "\"") quoted = false;
      else cell += char;
    } else if (char === "\"") quoted = true;
    else if (char === ",") { row.push(cell); cell = ""; }
    else if (char === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (char !== "\r") cell += char;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((line) => line.some((value) => String(value).trim() !== ""));
}

async function fileToPayload(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".xlsx")) return xlsxToPayload(await file.arrayBuffer());
  const text = await file.text();
  jsonBox.value = text;
  if (name.endsWith(".json")) return JSON.parse(text);
  return csvToPayload(text);
}

async function xlsxToPayload(buffer) {
  const files = await unzip(buffer);
  const sheetName = Object.keys(files).find((key) => key.startsWith("xl/worksheets/sheet"));
  if (!sheetName) throw new Error("That workbook has no sheet.");
  const shared = parseSharedStrings(files["xl/sharedStrings.xml"] || "");
  const rows = parseSheet(files[sheetName], shared);
  if (rows.length < 2) throw new Error("The first sheet needs a header row and one month.");
  const header = rows[0];
  const history = rows.slice(1).filter((cells) => cells.some((value) => String(value).trim() !== "")).map((cells) => {
    const record = {};
    header.forEach((name, index) => { if (name) record[name] = cells[index] ?? ""; });
    return record;
  });
  jsonBox.value = JSON.stringify(history.length === 1 ? history[0] : { history }, null, 2);
  return history.length === 1 ? history[0] : { history };
}

function parseSharedStrings(xml) {
  const strings = [];
  for (const item of xml.matchAll(/<si[\s\S]*?<\/si>/g)) {
    strings.push([...item[0].matchAll(/<t[^>]*>([\s\S]*?)<\/t>/g)].map((match) => decodeXml(match[1])).join(""));
  }
  return strings;
}

function parseSheet(xml, shared) {
  const rows = [];
  for (const row of xml.matchAll(/<row[^>]*>([\s\S]*?)<\/row>/g)) {
    const cells = [];
    for (const cell of row[1].matchAll(/<c([^>]*)>([\s\S]*?)<\/c>/g)) {
      const ref = (cell[1].match(/r="([A-Z]+)\d+"/) || [])[1] || "";
      const type = (cell[1].match(/t="([^"]+)"/) || [])[1] || "";
      const column = columnIndex(ref);
      let value = (cell[2].match(/<v>([\s\S]*?)<\/v>/) || [])[1] || "";
      if (type === "s") value = shared[Number(value)] || "";
      else if (type === "inlineStr") value = [...cell[2].matchAll(/<t[^>]*>([\s\S]*?)<\/t>/g)].map((match) => decodeXml(match[1])).join("");
      else value = decodeXml(value);
      cells[column] = value;
    }
    rows.push(Array.from(cells, (value) => value ?? ""));
  }
  return rows;
}

function columnIndex(ref) {
  let index = 0;
  for (const char of ref) index = index * 26 + (char.charCodeAt(0) - 64);
  return Math.max(0, index - 1);
}

function decodeXml(value) {
  return value.replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&amp;", "&").replaceAll("&quot;", "\"").replaceAll("&apos;", "'");
}

async function unzip(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const files = {};
  let offset = 0;
  while (offset + 30 < bytes.length && view.getUint32(offset, true) === 0x04034b50) {
    const method = view.getUint16(offset + 8, true);
    const size = view.getUint32(offset + 18, true);
    const nameLen = view.getUint16(offset + 26, true);
    const extraLen = view.getUint16(offset + 28, true);
    const start = offset + 30 + nameLen + extraLen;
    const name = new TextDecoder().decode(bytes.subarray(offset + 30, offset + 30 + nameLen));
    const compressed = bytes.subarray(start, start + size);
    let raw = compressed;
    if (method === 8) {
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
      raw = new Uint8Array(await new Response(stream).arrayBuffer());
    } else if (method !== 0) throw new Error("This workbook uses a compression the page cannot read. Save it as CSV.");
    if (name.endsWith(".xml")) files[name] = new TextDecoder().decode(raw);
    offset = start + size;
  }
  return files;
}

function classifyPayload(payload) {
  activeExample = null;
  exampleRow.querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", "false"));
  paintResult(classify(model, payload), null);
}

function runText() {
  formError.textContent = "";
  try {
    classifyPayload(payloadFromText(jsonBox.value));
  } catch (error) {
    formError.textContent = error.message === "Unexpected end of JSON input" || error instanceof SyntaxError
      ? "That text is not valid JSON or CSV."
      : error.message;
  }
}

document.querySelector("#run").addEventListener("click", runText);
document.querySelector("#download-year").addEventListener("click", () => {
  const blob = new Blob([jsonBox.value], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = activeExample ? `${activeExample.id}.json` : "whanau.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

const file = document.querySelector("#file");
const drop = document.querySelector("#drop");
file.addEventListener("change", async () => {
  if (!file.files[0]) return;
  formError.textContent = "";
  try {
    classifyPayload(await fileToPayload(file.files[0]));
  } catch (error) {
    formError.textContent = error.message;
  }
});
["dragover", "dragenter"].forEach((name) => drop.addEventListener(name, (event) => {
  event.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => {
  event.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", async (event) => {
  const dropped = event.dataTransfer.files[0];
  if (!dropped) return;
  formError.textContent = "";
  try {
    classifyPayload(await fileToPayload(dropped));
  } catch (error) {
    formError.textContent = error.message;
  }
});

boot().catch((error) => {
  scoreBody.textContent = `The model file did not load. ${error.message}`;
});
