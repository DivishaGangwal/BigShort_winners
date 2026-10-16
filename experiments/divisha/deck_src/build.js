// KESTREL deck, dark financial-portfolio theme.
// NODE_PATH=<dir with pptxgenjs>/node_modules node experiments/divisha/deck_src/build.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require("/Users/divishagangwal/.claude/skills/synced/143aa10f-e77c-4eec-bc5d-3610d2863ff1_9d3d8cfe-3947-42df-81bb-26e36a4b9d7b/pptx/scripts/apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const OUT = path.resolve(__dirname, "../KESTREL_architecture.pptx");

// Palette (hex for options that only take hex)
const BG = "0A1222", CARD = "14213A", CARD_G = "0F3329", CARD_R = "3A1820", WHITE = "FFFFFF",
  MUTED = "A9B6CC", GREEN = "21C17A", RED = "EF4B4B", GOLD = "D9B44A", ICE = "8FB3E8", SLATE = "6B7A99", GRID = "24324F";
const THEME = { name: "KESTREL Portfolio", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: BG, lt1: WHITE, dk2: CARD, lt2: "DCE4F0", accent1: GREEN, accent2: RED, accent3: GOLD, accent4: SLATE, accent5: ICE, accent6: MUTED, hlink: GOLD, folHlink: ICE } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "KESTREL: Crash-Aware Trading Signal Engine";
pres.author = "BigShort Winners";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor; // text1 = BG, background1 = white, text2 = CARD

pres.defineSlideMaster({
  title: "HERO",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.3, w: 12.1, h: 1.4, fontSize: 44, bold: true, color: C.background1, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.75, w: 12.1, h: 0.9, fontSize: 20, color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.text1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.8, fontSize: 32, bold: true, color: C.background1, valign: "middle" }, text: "" } },
    { text: { text: "KESTREL · BigShort Winners", options: { x: 0.5, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: C.accent6 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6 },
});

const title = (s, t) => s.addText(t, { placeholder: "title" });
const txt = (s, t, o) => s.addText(t, Object.assign({ isTextBox: true, fontSize: 14, color: C.background1, valign: "top", margin: 0.05 }, o));
function card(s, x, y, w, h, fill, name, border) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: border || fill, width: 1 }, objectName: name });
}
function badge(s, x, y, label, fill, name) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill }, line: { color: fill }, objectName: name });
  s.addText(label, { isTextBox: true, x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.text1, margin: 0 });
}
const CT = () => ({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 11,
  titleFontSize: 13, titleColor: WHITE, legendColor: MUTED, dataLabelColor: WHITE, catAxisLineColor: GRID, valAxisLineColor: GRID,
  valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
  plotArea: { fill: { color: BG } }, chartArea: { fill: { color: BG } },
});

// ================= 1. Intro =================
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
title(s, "Markets are uncertain, and crashes are where models break");
const why = [
  ["Regimes shift without warning", "Volatility jumped from 0.012 to 0.074 (6.4× normal) within days"],
  ["Models learn the calm", "850 days of labelled history contain no crash at all"],
  ["Mistakes compound", "Wrong-way positions, flip-flopping costs and drawdowns wipe out returns"],
];
why.forEach(([h, d], i) => {
  const y = 1.35 + i * 1.1;
  badge(s, 0.5, y + 0.05, String(i + 1), i === 0 ? RED : GOLD, `intro-badge-${i}`);
  txt(s, h, { x: 1.15, y, w: 5.6, h: 0.42, fontSize: 18, bold: true });
  txt(s, d, { x: 1.15, y: y + 0.42, w: 5.6, h: 0.6, fontSize: 14, color: C.accent6 });
});
const V = D.viz;
s.addChart(pres.charts.LINE, [
  { name: "volatility_20", labels: V.labels, values: V.vol },
  { name: "Highest in training", labels: V.labels, values: V.labels.map(() => V.train_max_vol) },
], Object.assign(CT(), {
  x: 0.5, y: 4.75, w: 6.3, h: 2.15, showTitle: true, title: "Volatility, days 831–900: the crash leaves the training range",
  chartColors: [RED, GOLD], lineSize: 2, lineDataSymbol: "none", showLegend: true, legendPos: "r", valAxisLabelFormatCode: "0.00", catAxisLabelFrequency: 10,
}));
card(s, 7.2, 1.35, 5.6, 5.55, CARD, "building-card", GOLD);
txt(s, "WHAT WE ARE BUILDING", { x: 7.5, y: 1.55, w: 5.0, h: 0.4, fontSize: 13, bold: true, color: C.accent3, charSpacing: 2 });
txt(s, "A crash-aware trading signal engine", { x: 7.5, y: 1.95, w: 5.0, h: 0.9, fontSize: 24, bold: true, fontFace: "Cambria" });
txt(s, [
  { text: "Forecasts next-day direction from leak-free, volatility-normalised features", options: { bullet: true, breakLine: true } },
  { text: "Detects when the market leaves every regime it has seen", options: { bullet: true, breakLine: true } },
  { text: "Trains on, and stress-tests against, thousands of simulated crashes", options: { bullet: true, breakLine: true } },
  { text: "Outputs one daily position in [−1, +1], sized for risk, cost and drawdown", options: { bullet: true } },
], { x: 7.5, y: 2.95, w: 5.0, h: 3.7, fontSize: 16, paraSpaceAfter: 10 });
s.addNotes("Presenter: Divisha (25 s). Markets are uncertain, and the moments that matter most are regime breaks. In this competition the scored period is a crash: volatility rose to 6.4 times its normal level and far above anything in 850 days of labelled history (gold line). Most models are trained on calm data and fail exactly there. What we built: a crash-aware trading signal engine that forecasts, detects regime breaks, trains on simulated crashes and sizes a daily position for risk, cost and drawdown.");

// ================= 2. Headline =================
s = pres.addSlide({ masterName: "HERO", sectionTitle: "Introduction" });
s.addText("Surviving a 6× volatility crash", { placeholder: "title" });
s.addText("Leak-free features · Monte Carlo crashes · worst-regime losses · risk-sized positions", { placeholder: "body" });
[["53.0", "Round-3 leaderboard score"], ["+12.7%", "Our return while the market fell 9.2%"], ["1.8%", "Maximum drawdown through the crash"]].forEach(([v, l], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 4.1, 3.7, 2.1, CARD, `hero-stat-${i}`, i === 0 ? GOLD : GRID);
  s.addText(v, { isTextBox: true, x: x + 0.25, y: 4.25, w: 3.2, h: 1.0, fontSize: 48, bold: true, color: i === 0 ? C.accent3 : C.accent1, fontFace: "Cambria", margin: 0 });
  s.addText(l, { isTextBox: true, x: x + 0.25, y: 5.3, w: 3.2, h: 0.7, fontSize: 15, color: C.background1, margin: 0 });
});
s.addNotes("Presenter: Divisha (15 s). Headline: 53.0 on the leaderboard; +12.7% while buy-and-hold lost 9.2%; 1.8% maximum drawdown. Return and drawdown come from a post-scoring analysis; nothing from the scored period was used to build the submission.");

// ================= 3. Architecture =================
pres.addSection({ title: "Architecture" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
title(s, "One question drives every layer: what happens at 6× volatility?");
const steps = [
  ["Data contract", "Hash-checked; rows ≤ t only; 76 leakage tests"],
  ["Causal features", "Scale-free, vol-normalised, clipped at ±5σ"],
  ["Regimes + MC crashes", "Vol/RSI regimes; 10,000 simulated crash rows per fold"],
  ["Robust losses", "Worst-regime (exact max) + MC augmentation"],
  ["Crash-aware hybrid", "LR + boosting + attention LSTM"],
  ["Risk-sized signal", "Regime switch, vol scaling, cost-aware"],
];
const bw = 1.85, gap = 0.24;
steps.forEach(([h, d], i) => {
  const x = 0.5 + i * (bw + gap);
  const core = i === 2 || i === 3;
  card(s, x, 1.55, bw, 1.25, core ? CARD_G : CARD, `arch-box-${i}`, core ? GREEN : GRID);
  s.addText(h, { isTextBox: true, x: x + 0.1, y: 1.6, w: bw - 0.2, h: 1.15, fontSize: 16, bold: true, color: core ? C.accent1 : C.background1, align: "center", valign: "middle", margin: 0 });
  txt(s, d, { x, y: 2.95, w: bw, h: 1.5, color: C.accent6 });
  if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + bw + 0.02, y: 2.175, w: gap - 0.04, h: 0, line: { color: GOLD, width: 1.5, endArrowType: "triangle" }, objectName: `arch-arrow-${i}` });
});
card(s, 0.5, 4.75, 12.3, 1.85, CARD, "arch-callout", GRID);
txt(s, [
  { text: "Rounds 1–2: ", options: { bold: true, color: C.accent3 } }, { text: "next-day direction. No model beat the prior on real data, but MC-trained models stopped blowing up on crash inputs.", options: { breakLine: true } },
  { text: "Round 3: ", options: { bold: true, color: C.accent3 } }, { text: "daily signal in [−1, 1] scored as a portfolio (Sharpe, money made, drawdown, cost). Robustness became the edge.", options: {} },
], { x: 0.8, y: 4.95, w: 11.7, h: 1.5, fontSize: 16, paraSpaceAfter: 8 });
s.addNotes("Presenter: Divisha (25 s). Left to right. The green layers (regimes, Monte Carlo crashes, robust losses) are our core contribution: the crash never appears in labelled history, so we manufactured crashes and optimised the worst regime instead of the average.");

// ================= 4. Features =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
title(s, "Features: volatility ratios carry the signal, with zero lookahead");
const fRows = [
  ["Group", "Examples", "Why"],
  ["Vol ratios", "return ÷ σ, momentum ÷ σ√5, range ÷ σ", "Same scale in calm and crash"],
  ["Momentum / trend", "open/SMA10, SMA10/SMA30, momentum_5", "Trend state, scale-free"],
  ["Lag returns", "open-to-open 1, 3, 5 days; flow and volume lags", "Short memory"],
  ["Bar shape", "(high−low)/open, wicks, close location", "Intraday pressure"],
  ["Same-row close", "close[t] = close[t−5]·(1+mom5[t])", "Rows ≤ t only; 0.013% error"],
];
s.addTable(fRows.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? BG : WHITE, fill: { color: i === 0 ? GOLD : (i % 2 ? CARD : "101B30") } } }))),
  { x: 0.5, y: 1.35, w: 6.6, colW: [1.55, 3.0, 2.05], fontSize: 13, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: GRID }, rowH: 0.62 });
txt(s, [{ text: "No lookahead: ", options: { bold: true, color: C.accent3 } }, { text: "scaler and cutoffs fit on training rows only; tests prove perturbing any later row never changes a feature at t." }],
  { x: 0.5, y: 5.4, w: 6.6, h: 1.1 });
const NICE = { osma10_vol: "open/SMA10 ÷ σ", mom_vol: "momentum ÷ σ√5", rsi_14: "RSI-14", ret1_vol: "1-day return ÷ σ", smagap_vol: "SMA10/SMA30 ÷ σ",
  order_flow_lag2: "order flow, lag 2", sma10_sma30: "SMA10/SMA30", ret_1d_c: "1-day return", volume_z_20_lag2: "volume z, lag 2", momentum_5: "momentum, 5-day" };
s.addChart(pres.charts.BAR, [{ name: "|standardised coefficient|", labels: D.importance.names.map(n => NICE[n] || n).reverse(), values: D.importance.values.slice().reverse() }], Object.assign(CT(), {
  x: 7.4, y: 1.3, w: 5.4, h: 5.5, barDir: "bar", showTitle: true, title: "Feature importance (final model, |std. coefficient|)",
  chartColors: [GREEN], showValue: true, dataLabelPosition: "outEnd", showLegend: false, valAxisHidden: true, dataLabelFormatCode: "0.00",
}));
s.addNotes("Presenter: Friday (30 s). Rubric 1. The strongest inputs are volatility ratios: open-to-SMA gap and momentum divided by volatility. That normalisation keeps inputs in the training range at 6x volatility; without it standardised inputs reached |z| = 24. Standardise on training rows, clip at ±5. The close is rebuilt from the same row only; tomorrow's close is never reconstructed.");

// ================= 5. Validation =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
title(s, "Validation: chronological, purged, rules fixed before results");
const x0 = 0.9, x1 = 7.6, tmin = 25, tmax = 900;
const sx = t => x0 + (t - tmin) / (tmax - tmin) * (x1 - x0);
[[449, 451, 500], [499, 501, 550], [549, 551, 600], [599, 601, 650], [649, 651, 700], [699, 701, 750], [749, 751, 800], [799, 801, 850]].forEach(([te, v0, v1], i) => {
  const y = 1.45 + i * 0.42;
  s.addShape(pres.shapes.RECTANGLE, { x: sx(25), y, w: sx(te) - sx(25), h: 0.28, fill: { color: SLATE }, line: { color: SLATE }, objectName: `fold-train-${i}` });
  s.addShape(pres.shapes.RECTANGLE, { x: sx(v0), y, w: sx(v1) - sx(v0), h: 0.28, fill: { color: i === 7 ? RED : GOLD }, line: { color: i === 7 ? RED : GOLD }, objectName: `fold-val-${i}` });
});
s.addShape(pres.shapes.RECTANGLE, { x: sx(851), y: 1.45 + 8 * 0.42, w: sx(900) - sx(851), h: 0.28, fill: { color: GREEN }, line: { color: GREEN }, objectName: "fold-predict" });
txt(s, "Train", { x: 0.9, y: 5.05, w: 1.0, h: 0.35, color: C.accent4 });
txt(s, "Validate", { x: 1.9, y: 5.05, w: 1.2, h: 0.35, color: C.accent3 });
txt(s, "Reserved (once)", { x: 3.1, y: 5.05, w: 1.9, h: 0.35, color: C.accent2 });
txt(s, "Predict 851–900", { x: 5.0, y: 5.05, w: 2.4, h: 0.35, color: C.accent1 });
txt(s, "One-row purge before every block (label t uses close t+1). Scaler, regime cutoffs and the crash simulator are refit inside each fold on training rows only.",
  { x: 0.5, y: 5.55, w: 7.1, h: 1.2, color: C.accent6 });
[["19", "round-2 configurations: LR and LSTM × 5 loss arms, plus the prior baseline"], ["Fixed", "hyperparameters; every selection rule written before running"], ["Stop", "round 2: prior baseline won the rule, so we did not force a model"]].forEach(([v, l], i) => {
  const y = 1.4 + i * 1.8;
  card(s, 8.1, y, 4.7, 1.55, CARD, `val-card-${i}`, GRID);
  s.addText(v, { isTextBox: true, x: 8.3, y: y + 0.15, w: 1.5, h: 1.25, fontSize: 32, bold: true, color: C.accent3, fontFace: "Cambria", valign: "middle", margin: 0 });
  txt(s, l, { x: 9.8, y: y + 0.2, w: 2.85, h: 1.2, valign: "middle" });
});
s.addNotes("Presenter: Navjyot (30 s). Rubric 2. Seven expanding folds plus a reserved block, never shuffled, one-row purge. Logistic regression, gradient boosting, Extra Trees and an attention LSTM compared under identical folds and metrics. Every selection rule was written before results, which guards against leaderboard overfitting; in round 2 the prior won, so we stopped.");

// ================= 6. Loss regimes =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
title(s, "Loss regimes: optimise the worst market state, not the average");
[["A", "Standard", "Mean log loss: the reference"], ["B", "Worst-regime", "(1−λ)·mean + λ·max over regimes; λ ∈ {0.25, 0.5, 0.75}"],
 ["C", "Regime-weighted", "Weight 1.5–3× on high-vol and high-RSI rows"], ["D", "MC augmented", "Real rows + simulated crashes at 25% weight"],
 ["E", "MC + worst-regime", "D's data, B's objective, sim_crash as a regime"]].forEach(([l, h, d], i) => {
  const x = 0.5 + i * 2.5, hi = l === "D" || l === "E";
  card(s, x, 1.35, 2.3, 2.75, hi ? CARD_G : CARD, `arm-card-${l}`, hi ? GREEN : GRID);
  badge(s, x + 0.2, 1.55, l, hi ? GREEN : GOLD, `arm-badge-${l}`);
  txt(s, h, { x: x + 0.2, y: 2.15, w: 1.95, h: 0.45, fontSize: 16, bold: true, color: hi ? C.accent1 : C.background1 });
  txt(s, d, { x: x + 0.2, y: 2.6, w: 1.95, h: 1.4, color: C.accent6 });
});
card(s, 0.5, 4.4, 12.3, 2.35, CARD, "epigraph-box", GOLD);
s.addText("min (1−λ)·L̄ + λ·t + η·‖w‖²/2    s.t.   L_g ≤ t  for every regime g",
  { isTextBox: true, x: 0.8, y: 4.55, w: 11.7, h: 0.6, fontSize: 18, fontFace: "Cambria", color: C.accent3, margin: 0 });
txt(s, [
  { text: "Exact max, not a soft approximation: SLSQP with analytic gradients; all 15 fits converged (violation ≤ 1e-12)", options: { bullet: true, breakLine: true } },
  { text: "η = 1/(n·C), so λ = 0 equals sklearn logistic regression (tested to 1e-10); LSTM uses the exact full-batch group max", options: { bullet: true, breakLine: true } },
  { text: "Regimes: high/low vol, high/low RSI (cutoffs per fold) + sim_crash", options: { bullet: true } },
], { x: 0.8, y: 5.2, w: 11.7, h: 1.45, paraSpaceAfter: 4 });
s.addNotes("Presenter: Divisha (30 s). Five arms on identical folds, so differences come from the loss alone. B penalises the worst regime (Group-DRO style), solved exactly with an epigraph constraint. D adds Monte Carlo crashes; E combines both with the simulated crash as its own regime. Arm E was the best real-data model in round 2 and the LSTM we submitted in round 3.");

// ================= 7. Monte Carlo =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
title(s, "Monte Carlo crash regime: train on crashes history never showed us");
["Start from a random training day; 60-day real burn-in",
 "Block-bootstrap training returns (5-day blocks), rescaled to 3× or 5× median volatility",
 "Inject 1–2 single-day drops of −8% to −15%",
 "Rebuild OHLC; recompute SMA, momentum, vol and RSI with verified dataset formulas",
 "Resample volume and order flow from high-vol training days",
 "200 windows × 50 days = 10,000 rows per fold; separate seeds for training and stress test"].forEach((t, i) => {
  const y = 1.35 + i * 0.88;
  badge(s, 0.5, y + 0.05, String(i + 1), i === 5 ? RED : GREEN, `mc-step-${i}`);
  txt(s, t, { x: 1.15, y, w: 5.6, h: 0.8, valign: "middle" });
});
const keys = ["LR_A", "LR_D_mc", "LR_E_mc_lam0.5", "LSTM_A", "LSTM_E_mc_lam0.5"];
const lab = ["LR standard", "LR + MC", "LR MC+worst", "LSTM standard", "LSTM MC+worst"];
s.addChart(pres.charts.BAR, [
  { name: "Real validation", labels: lab, values: keys.map(k => D.r2_real[k]) },
  { name: "Simulated crash (synthetic)", labels: lab, values: keys.map(k => D.r2_stress[k]) },
], Object.assign(CT(), {
  x: 7.0, y: 1.3, w: 5.8, h: 4.6, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Round 2 log loss (lower is better)",
  chartColors: [ICE, RED], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "b", valAxisMinVal: 0.6, valAxisMaxVal: 1.3, valAxisLabelFormatCode: "0.0",
}));
card(s, 7.0, 6.05, 5.8, 0.75, CARD_R, "mc-callout", RED);
txt(s, [{ text: "MC training cut crash-case loss 1.22 → 0.70", options: { bold: true } }, { text: " (synthetic labels: robustness, not skill)", options: { color: C.accent6 } }],
  { x: 7.2, y: 6.12, w: 5.5, h: 0.62, valign: "middle" });
s.addNotes("Presenter: Navjyot (30 s). Labelled volatility never exceeds 0.0204; the scored block reaches 0.074. We simulate crashes from each fold's training rows only, with indicator formulas verified against the dataset. One seed trains, an independent seed stress-tests. The standard logistic regression blew up on crash inputs (1.22); MC-trained versions stayed near 0.69 to 0.70.");

// ================= 8. Flash crash & risk =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
title(s, "Flash-crash adaptation: detect, size, control");
[["Detect", "volatility_20 above anything in training → crash regime; switch from long-bias blend to the crash-trained ensemble"],
 ["Size", "signal = clip(5·(2p−1)) × min(1, σ_median/σ_t): exposure shrinks 3–6× automatically in the crash"],
 ["Control", "Backtest net of 10 bps per unit change; drawdown caps 15% real / 25% MC; turnover tracked"]].forEach(([h, d], i) => {
  const y = 1.4 + i * 1.75;
  badge(s, 0.5, y, String(i + 1), i === 0 ? RED : GREEN, `risk-badge-${i}`);
  txt(s, h, { x: 1.15, y: y - 0.02, w: 5.2, h: 0.45, fontSize: 18, bold: true, color: C.accent3 });
  txt(s, d, { x: 1.15, y: y + 0.45, w: 5.4, h: 1.15 });
});
const pol = D.policies;
const pk = ["P0_long_full", "P1_long_volscaled", "P3_blend", "P2_model_volscaled", "P6_trend_volscaled", "P5_regime_switch"];
s.addChart(pres.charts.BAR, [
  { name: "Real walk-forward Sharpe", labels: ["Full long", "Vol-scaled long", "Blend", "Model only", "Trend", "Regime switch ✓"], values: pk.map(k => pol[k][0]) },
  { name: "MC crash Sharpe", labels: ["Full long", "Vol-scaled long", "Blend", "Model only", "Trend", "Regime switch ✓"], values: pk.map(k => pol[k][1]) },
], Object.assign(CT(), {
  x: 6.9, y: 1.3, w: 5.9, h: 4.7, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Policy backtest, net of 10 bps (8 real blocks, 1,600 crash paths)",
  chartColors: [ICE, GREEN], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", valAxisLabelFormatCode: "0.0",
}));
txt(s, "Rule: maximise ½ real + ½ crash Sharpe within the drawdown caps. Only the regime switch is positive on both.", { x: 6.9, y: 6.15, w: 5.9, h: 0.65, italic: true, color: C.accent6 });
s.addNotes("Presenter: Friday (30 s). Rubric 3. When volatility leaves the training range we hand control to the crash-trained ensemble; positions scale inversely with volatility; every policy is judged net of costs with drawdown caps on real blocks and simulated crashes. Full long has the best calm-market Sharpe but loses in crashes; the regime switch is the only policy positive on both.");

// ================= 9. Predictions in action =================
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
title(s, "Our predictions in action: positions shrink and flip as the crash hits");
s.addChart(pres.charts.LINE, [{ name: "Open price", labels: V.labels, values: V.open }], Object.assign(CT(), {
  x: 0.5, y: 1.3, w: 8.6, h: 2.75, showTitle: true, title: "KESTREL open price, days 831–900", chartColors: [GOLD], lineSize: 2, lineDataSymbol: "none",
  showLegend: false, valAxisLabelFormatCode: "0", catAxisLabelFrequency: 10, valAxisMinVal: 250,
}));
s.addChart(pres.charts.BAR, [
  { name: "Long", labels: V.labels, values: V.long },
  { name: "Short", labels: V.labels, values: V.short },
], Object.assign(CT(), {
  x: 0.5, y: 4.15, w: 8.6, h: 2.75, barDir: "col", barGrouping: "stacked", barGapWidthPct: 40, showTitle: true, title: "Our final signal (v5), from day 851",
  chartColors: [GREEN, RED], showLegend: true, legendPos: "r", valAxisLabelFormatCode: "0.0", catAxisLabelFrequency: 10, valAxisMinVal: -0.4, valAxisMaxVal: 0.4,
}));
card(s, 9.4, 1.3, 3.4, 5.6, CARD, "viz-notes", GRID);
txt(s, [
  { text: "Day 851", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "Volatility still normal: long blend (0.22)", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Day 852", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "Volatility leaves the training range: regime switch engages (0.86, then 0.43)", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Days 854–900", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "Positions within ±0.33 at 4–6× volatility; 14 days exactly flat; direction flips with daily reversals", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "Sanity", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "Model probabilities stay in 0.42–0.58: no garbage outputs", options: {} },
], { x: 9.6, y: 1.45, w: 3.05, h: 5.35 });
s.addNotes("Presenter: Divisha (25 s). Top: the price path; the crash starts at 851 with swings of 10% a day. Bottom: our final signal, v5: the v3 calls rescaled to a 15% risk target, with positions of 0.05 or less set exactly flat. Positions shrink as volatility explodes and flip direction with the daily reversals; probabilities stay in a sane 0.42 to 0.58 band. Prices shown are the visible open prices from the dataset.");

// ================= 10. Results =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
title(s, "Results: +12.7% with 1.8% drawdown while the market fell 9.2%");
const ser = ["v3 (53.0)", "v1 (35.8)", "v4 (36.4)", "Buy & hold"];
s.addChart(pres.charts.LINE, ser.map(k => ({ name: k, labels: D.eq_labels, values: D.equity[k] })), Object.assign(CT(), {
  x: 0.5, y: 1.3, w: 8.4, h: 5.5, showTitle: true, title: "Cumulative return, days 852–900 (%), net of 10 bps",
  chartColors: [GREEN, GOLD, ICE, RED], lineSize: 2, lineDataSymbol: "none", showLegend: true, legendPos: "b", valAxisLabelFormatCode: "0", catAxisLabelFrequency: 6,
}));
[["53.0", "Leaderboard: v3 (v1 35.8, v4 36.4)"], ["4.75", "Sharpe in the crash (v1 1.65)"], ["73%", "Days the v3 call was right"]].forEach(([v, l], i) => {
  const y = 1.3 + i * 1.85;
  card(s, 9.2, y, 3.6, 1.6, CARD, `res-card-${i}`, i === 0 ? GOLD : GRID);
  s.addText(v, { isTextBox: true, x: 9.4, y: y + 0.1, w: 3.2, h: 0.85, fontSize: 40, bold: true, fontFace: "Cambria", color: i === 0 ? C.accent3 : C.accent1, margin: 0 });
  txt(s, l, { x: 9.4, y: y + 0.95, w: 3.2, h: 0.55 });
});
s.addNotes("Presenter: Divisha (25 s). v3 scored 53.0. In the post-scoring analysis it made 12.7% while buy-and-hold lost 9.2%, with a 1.8% maximum drawdown and the right direction on 73% of days. These curves use closes rebuilt after scoring; nothing from the scored period was used to build any submission.");

// ================= 11. Failure & limitations =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
title(s, "Failure case and limitations we can defend");
card(s, 0.5, 1.35, 6.0, 5.4, CARD_R, "fail-card", RED);
txt(s, "Why smoothing (v4) dropped to 36.4", { x: 0.75, y: 1.5, w: 5.5, h: 0.5, fontSize: 18, bold: true, color: C.accent2 });
txt(s, [
  { text: "The crash reversed daily (autocorrelation −0.16 vs +0.13 in history); yesterday's call was right only 29% of days, and smoothing mixes it in", options: { bullet: true, breakLine: true } },
  { text: "Smoothing shrank positions under the organizer's 0.05 flat threshold: 38 of 50 days flat", options: { bullet: true, breakLine: true } },
  { text: "The score rewards money made, not only Sharpe", options: { bullet: true, breakLine: true } },
  { text: "Our history and simulator both trend, so they favoured smoothing", options: { bullet: true } },
], { x: 0.75, y: 2.1, w: 5.5, h: 4.5, paraSpaceAfter: 8 });
card(s, 6.8, 1.35, 6.0, 5.4, CARD, "limits-card", GRID);
txt(s, "Assumptions and limitations", { x: 7.05, y: 1.5, w: 5.5, h: 0.5, fontSize: 18, bold: true, color: C.accent3 });
txt(s, [
  { text: "Target inferred: next-close direction (849/849 rows)", options: { bullet: true, breakLine: true } },
  { text: "Same-row timing: indicators at t are known when trading t", options: { bullet: true, breakLine: true } },
  { text: "No real directional edge in calm markets (rounds 1–2)", options: { bullet: true, breakLine: true } },
  { text: "Simulator choices (3×/5× vol, −8% to −15% shocks) are designed, not estimated", options: { bullet: true, breakLine: true } },
  { text: "50-day blocks: Sharpe standard error ≈ 2.2", options: { bullet: true } },
], { x: 7.05, y: 2.1, w: 5.5, h: 4.5, paraSpaceAfter: 8 });
s.addNotes("Presenter: Navjyot (25 s). Rubric 4: assumptions, limitations, failure cases. Our second submission smoothed v3 and dropped to 36.4: the crash reversed daily, smoothing pushed positions below the organizer's flat threshold, and the score rewards money made. Our backtest favoured smoothing because history and the simulator trend; fix: reversal scenarios and no smoothing at high volatility.");

// ================= 12. Analyst use =================
pres.addSection({ title: "Impact" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Impact" });
title(s, "How a real financial analyst can use this");
[["Risk desk: stress testing", "Run the Monte Carlo crash engine on any book to see drawdown, VaR and CVaR before a shock arrives"],
 ["Regime early warning", "The out-of-range volatility detector flags when every model is outside its experience: cut leverage, escalate review"],
 ["Position-sizing overlay", "Apply volatility scaling and the crash switch on top of an existing strategy to cap drawdowns"],
 ["Model validation template", "Reuse leakage tests, purged chronological CV and pre-registered selection rules for model-risk sign-off"],
 ["Hedge timing input", "Use the crash-regime flag as one input to when to add or roll protective hedges"],
 ["Strategy post-mortems", "Attribute losses to timing, turnover or thresholds, as in our v4 analysis"]].forEach(([h, d], i) => {
  const cx = 0.5 + (i % 3) * 4.15, cy = 1.35 + Math.floor(i / 3) * 2.6;
  card(s, cx, cy, 3.95, 2.35, CARD, `analyst-card-${i}`, i < 3 ? GOLD : GRID);
  badge(s, cx + 0.2, cy + 0.2, String(i + 1), i < 3 ? GOLD : GREEN, `analyst-badge-${i}`);
  txt(s, h, { x: cx + 0.85, y: cy + 0.2, w: 2.95, h: 0.55, fontSize: 16, bold: true, valign: "middle" });
  txt(s, d, { x: cx + 0.2, y: cy + 0.85, w: 3.6, h: 1.4, color: C.accent6 });
});
txt(s, "Decision support, not autopilot: a single-asset prototype; validate on your own universe before use.", { x: 0.5, y: 6.6, w: 12.3, h: 0.4, italic: true, color: C.accent3 });
s.addNotes("Presenter: Friday (25 s). Three ways a desk could use this today: stress-test a book against simulated crashes, use the out-of-range volatility detector as an early warning, and put the volatility-scaling and crash-switch overlay on an existing strategy to cap drawdowns. The validation framework doubles as a model-risk template. It is a single-asset prototype: decision support, not autopilot.");

// ================= 13. Future scope =================
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Impact" });
title(s, "Future scope: technically sound extensions of this architecture");
[["MC risk engine", "Regime-switching and jump-diffusion paths, including reversal crashes; report VaR/CVaR, drawdown and ruin probability; CVaR-constrained sizing", GREEN],
 ["RL agent in the MC crash simulator", "Not on the 50 real crash rows (too few, would fail in calm markets). Train in the simulator with calm, crash, trend and reversal regimes; reward = differential Sharpe − cost − drawdown", GREEN],
 ["Regime detection", "Hidden Markov or change-point models to detect volatility spikes and concept drift, replacing the fixed volatility threshold", GOLD],
 ["Generative crash scenarios", "TimeGAN or diffusion scenarios to widen the simulator; validate tail fidelity, a known weakness", GOLD],
 ["Overfitting-aware validation", "Combinatorial purged CV and probability of backtest overfitting before any policy is trusted", GOLD],
 ["Autocorrelation-aware smoothing", "Smooth only when recent return autocorrelation (rows ≤ t) is positive: the v4 lesson", GOLD]].forEach(([h, d, col], i) => {
  const cx = 0.5 + (i % 3) * 4.15, cy = 1.35 + Math.floor(i / 3) * 2.75;
  card(s, cx, cy, 3.95, 2.55, i < 2 ? CARD_G : CARD, `future-card-${i}`, i < 2 ? GREEN : GRID);
  badge(s, cx + 0.2, cy + 0.2, String(i + 1), col, `future-badge-${i}`);
  txt(s, h, { x: cx + 0.85, y: cy + 0.2, w: 2.95, h: 0.55, fontSize: 16, bold: true, valign: "middle" });
  txt(s, d, { x: cx + 0.2, y: cy + 0.85, w: 3.6, h: 1.6, color: C.accent6 });
});
s.addNotes("Presenter: Friday (25 s). Monte Carlo for risk: extend our simulator with regime-switching, jump and reversal paths and report VaR, CVaR and drawdown distributions. RL: training only on the 50 real crash rows would overfit and fail in normal markets, so the sound version uses our simulator as the environment with domain randomisation across regimes, with costs and drawdown in the reward and an overfitting test before use. 3 to 6 come from current research.");

// ================= 14. Appendix =================
pres.addSection({ title: "Appendix" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
title(s, "Appendix: reproducibility and sources");
txt(s, [
  { text: "Reproduce", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "python -m src.run_round3 --lstm   (v3 submission, byte-identical rerun)", options: { bullet: true, breakLine: true } },
  { text: "python -m src.run_round2   ·   python -m src.postmortem_round3", options: { bullet: true, breakLine: true } },
  { text: "76 automated leakage and correctness tests run before every experiment", options: { bullet: true, breakLine: true } },
  { text: "Pinned requirements-lock.txt; deterministic CPU PyTorch; dataset, manifest, simulator and submission hashes recorded", options: { bullet: true } },
], { x: 0.5, y: 1.35, w: 12.3, h: 2.4, paraSpaceAfter: 4 });
card(s, 0.5, 3.95, 12.3, 2.85, CARD, "sources-card", GRID);
txt(s, [
  { text: "Future-scope sources (web search; no Reddit threads surfaced)", options: { bold: true, color: C.accent3, breakLine: true } },
  { text: "Gort et al. 2022, DRL backtest overfitting: arxiv.org/abs/2209.05559", options: { bullet: true, breakLine: true } },
  { text: "Regime-conditioned diffusion, CVaR portfolios (MARCD): arxiv.org/pdf/2510.10807", options: { bullet: true, breakLine: true } },
  { text: "Tail risk under jump-diffusion, CVaR: arxiv.org/abs/2607.00883", options: { bullet: true, breakLine: true } },
  { text: "Generative synthetic financial data (TimeGAN, VAE): arxiv.org/abs/2512.21798", options: { bullet: true, breakLine: true } },
  { text: "HMM regime-adaptive trading: blog.quantinsti.com/regime-adaptive-trading-python", options: { bullet: true } },
], { x: 0.75, y: 4.1, w: 11.8, h: 2.6, fontSize: 12, color: C.accent6, paraSpaceAfter: 2 });
s.addNotes("Backup slide for Q&A; not presented in the 3–5 minute slot.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
