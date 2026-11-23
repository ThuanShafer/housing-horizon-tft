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

const scoreLead = document.querySelector("#score-lead");
const scoreStats = document.querySelector("#score-stats");
const scoreViz = document.querySelector("#score-viz");
const scoreRecall = document.querySelector("#score-recall");
const scoreFoot = document.querySelector("#score-foot");
const matrixBody = document.querySelector("#matrix-body");
const driversBody = document.querySelector("#drivers-body");
const cohortBody = document.querySelector("#cohort-body");
const dictBody = document.querySelector("#dict-body");
const dictNote = document.querySelector("#dict-note");
const result = document.querySelector("#result");
const jsonBox = document.querySelector("#json");
const formError = document.querySelector("#form-error");
const whanauSearch = document.querySelector("#whanau-search");
const whanauList = document.querySelector("#whanau-list");
const whanauMode = document.querySelector("#whanau-mode");
const whanauClear = document.querySelector("#whanau-clear");
const cohortCount = document.querySelector("#cohort-count");

let model;
let metrics;
let cohort = [];
let examples = [];
let histories = {};
let exampleById = new Map();
let activeExample = null;
let selectedId = null;
let listOpen = false;
let activeOption = 0;
let cohortTableBody = null;

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

function shiftMonth(value, steps) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})$/);
  if (!match) return String(value || "");
  const cursor = Number(match[1]) * 12 + (Number(match[2]) - 1) + steps;
  const year = Math.floor(cursor / 12);
  const month = (cursor % 12) + 1;
  return `${year}-${String(month).padStart(2, "0")}`;
}

function prettyMonth(value, long) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})$/);
  if (!match) return String(value || "");
  const names = long
    ? ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    : ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${names[Number(match[2]) - 1]} ${match[1]}`;
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
  const [modelDoc, metricsDoc, cohortDoc, exampleDoc, historyRes] = await Promise.all([
    ...["model.json", "metrics.json", "cohort.json", "examples.json"].map((name) =>
      fetch(`${base}/${name}?t=${stamp}`).then((res) => {
        if (!res.ok) throw new Error(`Could not read the seed ${seed} bundle.`);
        return res.json();
      }),
    ),
    fetch(`${base}/histories.json?t=${stamp}`),
  ]);
  model = loadModel(modelDoc);
  metrics = metricsDoc;
  cohort = cohortDoc;
  examples = exampleDoc;
  histories = historyRes.ok ? await historyRes.json() : {};
  exampleById = new Map(examples.map((example) => [example.id, example]));
  selectedId = null;
  listOpen = false;
  activeOption = 0;
  whanauSearch.value = "";
  whanauMode.value = "All whānau";
  renderScore();
  renderDrivers();
  renderCohort();
  renderDictionary();
  const first = cohort.find((row) => row.id === examples[0]?.id) || cohort[0];
  if (first) selectFamily(first);
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

function wireFinder() {
  whanauSearch.addEventListener("input", () => {
    listOpen = true;
    activeOption = 0;
    const exact = cohort.find((row) => row.id.toLowerCase() === whanauSearch.value.trim().toLowerCase());
    if (exact) {
      listOpen = false;
      selectFamily(exact);
      return;
    }
    paintCohortLists();
  });
  whanauSearch.addEventListener("focus", () => {
    listOpen = true;
    paintCohortLists();
  });
  whanauSearch.addEventListener("keydown", (event) => {
    const rows = matchingRows();
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (!listOpen) {
        listOpen = true;
        activeOption = 0;
      } else activeOption = Math.min(Math.max(rows.length - 1, 0), activeOption + 1);
      paintCohortLists();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      listOpen = true;
      activeOption = Math.max(0, activeOption - 1);
      paintCohortLists();
    } else if (event.key === "Enter") {
      event.preventDefault();
      const row = rows[activeOption] || rows[0];
      if (row) selectFamily(row, { fillSearch: true, scroll: true });
    } else if (event.key === "Escape") {
      listOpen = false;
      paintCohortLists();
    }
  });
  document.querySelector("#whanau-toggle").addEventListener("mousedown", (event) => event.preventDefault());
  document.querySelector("#whanau-toggle").addEventListener("click", () => {
    listOpen = !listOpen;
    activeOption = 0;
    paintCohortLists();
  });
  whanauClear.addEventListener("mousedown", (event) => event.preventDefault());
  whanauClear.addEventListener("click", () => {
    whanauSearch.value = "";
    listOpen = true;
    activeOption = 0;
    paintCohortLists();
  });
  whanauMode.addEventListener("change", () => {
    activeOption = 0;
    paintCohortLists();
  });
  document.addEventListener("click", (event) => {
    if (!listOpen || document.querySelector("#whanau-combo").contains(event.target)) return;
    listOpen = false;
    paintCohortLists();
  });
}

async function boot() {
  wireFinder();
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
  clear(scoreStats);
  clear(scoreViz);
  clear(scoreRecall);
  clear(scoreFoot);
  const test = metrics.test_origin;
  const base = metrics.baselines_h6;
  const stats = el("div", "stats score-headline-stats");
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
  scoreStats.append(stats);
  if (metrics.performance_h6) scoreViz.append(metricBlock(metrics.performance_h6));

  const correct = cohort.filter((row) => row.predicted.h6 === row.actual.h6).length;
  const unchanged = cohort.filter((row) => row.actual.now === row.actual.h6).length;
  const changers = cohort.length - unchanged;
  const caught = cohort.filter((row) => row.actual.now !== row.actual.h6 && row.predicted.h6 === row.actual.h6).length;
  scoreLead.textContent = `On ${cohort.length} held-out whānau, looking six months past ${metrics.headline_origin.label}, the tier is right for ${correct}. ${unchanged} do not change tier, so carrying that month forward scores ${fmt(base.persistence.macro_f1)} macro-F1 (${pct(base.persistence.accuracy)} accuracy) against the transformer at ${fmt(test.h6.macro_f1)} (${pct(test.h6.accuracy)}). Of the ${changers} who change, the transformer calls ${caught}. A last-month linear model scores ${fmt(base.last_month_linear.macro_f1)}; with 3- and 6-month changes it scores ${fmt(base.last_month_plus_deltas_linear.macro_f1)}.`;

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
  scoreRecall.append(el("h3", null, "Recall by tier at six months"), table);
  scoreRecall.append(el("p", "section-note", `Same 200 whānau: nowcast macro-F1 ${fmt(test.now.macro_f1)}, +3 months ${fmt(test.h3.macro_f1)}, +12 months ${fmt(test.h12.macro_f1)}. Severe deprivation has single-digit support here, so that recall is a small count.`));
  clear(matrixBody);
  matrixBody.append(matrix(test.h6.confusion, ["Severe", "Unstable", "At risk", "Assisted", "Stable"]));

  const fit = metrics.families;
  const through = metrics.headline_origin.forecast_through || metrics.headline_origin.forecast_year;
  const seedText = `Seed ${metrics.seed} · ${fit.train_pool} train / ${fit.test} test · seed ${metrics.inner_seed} holds out ${fit.validation} for early stopping · fit on ${fit.fit}. Window ${metrics.headline_origin.history}, forecast through ${through}.`;
  scoreFoot.append(el("p", "section-note", seedText));
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
  svg.setAttribute("viewBox", "0 0 240 240");
  svg.setAttribute("class", "roc");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "ROC curves for the six-month forecast");
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "path");
  axis.setAttribute("d", "M30 10 V205 H215");
  axis.setAttribute("class", "axis");
  svg.append(axis);
  const chance = document.createElementNS("http://www.w3.org/2000/svg", "path");
  chance.setAttribute("d", "M30 205 L215 10");
  chance.setAttribute("class", "chance");
  svg.append(chance);
  const tone = getComputedStyle(document.documentElement);
  const colors = ["--severe", "--unstable", "--risk", "--subsidised", "--stable"].map((name) => tone.getPropertyValue(name).trim());
  const legend = el("div", "roc-legend");
  curves.forEach((curve, index) => {
    if (!curve.points || !curve.points.length) return;
    const d = curve.points.map(([fpr, tpr], i) => `${i ? "L" : "M"}${30 + fpr * 185} ${205 - tpr * 195}`).join(" ");
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
  x.setAttribute("y", "222");
  x.setAttribute("class", "tick");
  x.textContent = "False positive rate";
  const y = document.createElementNS("http://www.w3.org/2000/svg", "text");
  y.setAttribute("x", "8");
  y.setAttribute("y", "16");
  y.setAttribute("class", "tick");
  y.textContent = "True positive rate";
  svg.append(x, y);
  const block = el("div", "roc-block");
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
  const scroller = el("div", "scroll");
  const table = el("table");
  const head = el("tr");
  const originLabel = prettyMonth(metrics.headline_origin.label);
  const sixLabel = prettyMonth(shiftMonth(metrics.headline_origin.label, 6));
  ["Whānau", "Board", "Ethnicity", originLabel, `${sixLabel} actual`, `${sixLabel} forecast`, "P"].forEach((label, i) => {
    head.append(el("th", i === 6 ? "n" : null, label));
  });
  table.append(el("thead").appendChild(head).parentNode);
  cohortTableBody = el("tbody");
  table.append(cohortTableBody);
  scroller.append(table);
  cohortBody.append(scroller);
  paintCohortLists();
}

function passesMode(row, mode) {
  const changed = row.actual.now !== row.actual.h6;
  const hit = row.predicted.h6 === row.actual.h6;
  if (mode === "Tier changed") return changed;
  if (mode === "Forecast missed") return !hit;
  if (mode === "Forecast matched") return hit;
  return true;
}

function matchingRows() {
  const q = whanauSearch.value.trim().toLowerCase();
  return cohort.filter((row) => {
    if (q && !`${row.id} ${row.local_board}`.toLowerCase().includes(q)) return false;
    return passesMode(row, whanauMode.value);
  });
}

function familyHistory(id) {
  if (histories && histories[id]) return histories[id];
  return exampleById.get(id)?.history || null;
}

function tierMark(id, tone) {
  const pill = el("span", tone ? `pill ${tone}` : "pill fact");
  const dot = document.createElement("i");
  dot.style.background = TIER_COLOR[id];
  pill.append(dot, document.createTextNode(tierShort(id)));
  return pill;
}

function chanceCell(row) {
  const probability = row.probability_h6[row.predicted.h6 - 1];
  const wrap = el("span", "chance");
  const track = document.createElement("i");
  const fill = document.createElement("b");
  fill.style.width = `${probability * 100}%`;
  fill.style.background = row.predicted.h6 === row.actual.h6 ? "var(--stable)" : "var(--severe)";
  track.append(fill);
  wrap.append(track, el("span", null, pct(probability)));
  return wrap;
}

function paintCohortLists() {
  const rows = matchingRows();
  if (activeOption > rows.length - 1) activeOption = 0;
  cohortCount.textContent = rows.length === 1 && rows[0].id === selectedId
    ? `${rows[0].id} loaded`
    : `${rows.length} shown`;
  whanauClear.hidden = whanauSearch.value.trim() === "";
  whanauSearch.setAttribute("aria-expanded", listOpen ? "true" : "false");
  whanauList.hidden = !listOpen;
  clear(whanauList);
  if (!rows.length) {
    whanauSearch.removeAttribute("aria-activedescendant");
    const empty = el("li", "empty", "No whānau matches that id.");
    empty.setAttribute("role", "presentation");
    whanauList.append(empty);
  } else {
    if (listOpen && rows[activeOption]) whanauSearch.setAttribute("aria-activedescendant", `whanau-opt-${activeOption}`);
    else whanauSearch.removeAttribute("aria-activedescendant");
    rows.forEach((row, index) => {
      const option = el("li");
      option.id = `whanau-opt-${index}`;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", index === activeOption ? "true" : "false");
      const board = el("span", "board", row.local_board);
      option.append(el("b", null, row.id), board, tierMark(row.predicted.h6, row.predicted.h6 === row.actual.h6 ? "hit" : "miss"));
      option.addEventListener("mousedown", (event) => {
        event.preventDefault();
        selectFamily(row, { fillSearch: true, scroll: true });
      });
      whanauList.append(option);
    });
    const active = listOpen ? whanauList.querySelector("[aria-selected='true']") : null;
    if (active) {
      const top = active.offsetTop;
      const bottom = top + active.offsetHeight;
      if (top < whanauList.scrollTop) whanauList.scrollTop = top;
      else if (bottom > whanauList.scrollTop + whanauList.clientHeight) {
        whanauList.scrollTop = bottom - whanauList.clientHeight;
      }
    }
  }
  if (!cohortTableBody) return;
  clear(cohortTableBody);
  rows.forEach((row) => {
    const tr = el("tr");
    const hit = row.predicted.h6 === row.actual.h6;
    if (row.id === selectedId) tr.classList.add("picked");
    tr.setAttribute("aria-selected", row.id === selectedId ? "true" : "false");
    const forecastCell = el("td");
    forecastCell.append(tierMark(row.predicted.h6, hit ? "hit" : "miss"));
    const probabilityCell = el("td", "n");
    probabilityCell.append(chanceCell(row));
    const nowCell = el("td");
    nowCell.append(tierMark(row.actual.now));
    const juneCell = el("td");
    juneCell.append(tierMark(row.actual.h6));
    tr.append(el("td", "wid", row.id), el("td", null, row.local_board), el("td", null, row.ethnicity), nowCell, juneCell, forecastCell, probabilityCell);
    tr.addEventListener("click", () => selectFamily(row, { fillSearch: true, scroll: true }));
    cohortTableBody.append(tr);
  });
}

function selectFamily(row, opts = {}) {
  if (!row) return;
  selectedId = row.id;
  if (opts.fillSearch) {
    whanauSearch.value = row.id;
    listOpen = false;
    activeOption = 0;
  }
  if (!passesMode(row, whanauMode.value)) whanauMode.value = "All whānau";
  const history = familyHistory(row.id);
  const example = exampleById.get(row.id);
  formError.textContent = "";
  if (history) {
    activeExample = example || { id: row.id, history, actual: row.actual };
    jsonBox.value = JSON.stringify({ history }, null, 2);
    paintResult(classify(model, { history }), activeExample);
  } else {
    activeExample = { id: row.id, actual: row.actual };
    jsonBox.value = JSON.stringify(row, null, 2);
    paintPublished(row);
  }
  paintCohortLists();
  if (opts.scroll) document.querySelector("#call").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function paintPublished(row) {
  renderDrivers();
  const forecast = model.doc.tiers[row.predicted.h6 - 1];
  const snapshot = model.doc.tiers[row.actual.now - 1];
  const june = model.doc.tiers[row.actual.h6 - 1];
  const probability = row.probability_h6[row.predicted.h6 - 1];
  const runnerUp = Math.max(...row.probability_h6.filter((_, index) => index !== forecast.id - 1));
  const hit = row.predicted.h6 === row.actual.h6;
  document.querySelector("#flow-input").textContent = `${row.id}, ${row.local_board}. ${row.ethnicity}.`;
  document.querySelector("#flow-weights").textContent = "Weights appear once this history is on the page.";
  document.querySelector("#flow-attend").textContent = "Attention is drawn from the loaded twelve months.";
  document.querySelector("#flow-out").textContent = `${forecast.short}, ${pct(probability)}.`;
  clear(result);
  result.append(el("p", "kicker", `${row.id} · ${row.local_board}`));
  const pair = el("div", "pair");
  pair.append(
    tierCard("Six-month forecast", {
      tier_id: forecast.id,
      name: forecast.name,
      probability,
      margin: probability - runnerUp,
      low_confidence: probability < 0.4,
    }),
    factCard(prettyMonth(metrics.headline_origin.label, true), snapshot),
  );
  result.append(pair);
  const sixName = prettyMonth(shiftMonth(metrics.headline_origin.label, 6), true);
  result.append(el("p", null, `In the workbook, ${sixName} is ${june.name}. The forecast ${hit ? "matches" : "does not match"} that outcome.`));
  result.append(el("h3", null, "Six-month probabilities"));
  const probs = el("div", "prob");
  row.probability_h6.forEach((value, index) => {
    const tier = model.doc.tiers[index];
    const line = el("div", "prob-row");
    line.append(el("span", null, tier.short));
    const track = el("div", "track");
    const fill = document.createElement("i");
    fill.style.width = `${value * 100}%`;
    fill.style.background = TIER_COLOR[tier.id];
    track.append(fill);
    line.append(track, el("span", "num", pct(value)));
    probs.append(line);
  });
  result.append(probs);
  result.append(el("p", "section-note", "Published test forecast for this whānau. The month path and attention appear when the twelve-month history is in the bundle."));
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
  const sixLabel = prettyMonth(shiftMonth(metrics.headline_origin.label, 6));
  key.append(el("span", "a", `${sixLabel} actual`), el("span", "p", `${sixLabel} forecast`));
  const block = el("div", "viz-block");
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

function factCard(label, tier) {
  const card = el("div", "card");
  card.style.borderTopColor = TIER_COLOR[tier.id];
  card.append(el("p", "kicker", label), el("p", "tier-name", tier.name));
  return card;
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
  result.append(el("p", "kicker", example ? `Test whānau ${example.id}` : (output.whanau_id || "Uploaded history")));
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
  const attnWrap = el("div", "attn-wrap");
  const attn = el("div", "attn");
  const max = Math.max(...output.attention.weights);
  output.attention.weights.forEach((weight, index) => {
    const bar = el("span");
    bar.style.height = `${Math.max(4, (weight / max) * 100)}%`;
    bar.style.background = index < 12 ? "var(--accent)" : "var(--risk)";
    bar.title = `${output.attention.labels[index]} ${weight.toFixed(3)}`;
    attn.append(bar);
  });
  attnWrap.append(attn);
  result.append(attnWrap);
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
  selectedId = null;
  whanauSearch.value = "";
  listOpen = false;
  paintResult(classify(model, payload), null);
  paintCohortLists();
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
  scoreLead.textContent = `The model file did not load. ${error.message}`;
  [scoreStats, scoreViz, scoreRecall, scoreFoot].forEach(clear);
  clear(matrixBody);
});
