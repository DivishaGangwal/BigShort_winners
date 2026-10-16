// Build KESTREL architecture deck (rubric-aligned). node build.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require("/Users/divishagangwal/.claude/skills/synced/143aa10f-e77c-4eec-bc5d-3610d2863ff1_9d3d8cfe-3947-42df-81bb-26e36a4b9d7b/pptx/scripts/apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const OUT = path.resolve(__dirname, "../../../experiments/divisha/KESTREL_architecture.pptx");

const HEX = {
  dk1: "161B26", lt1: "FFFFFF", dk2: "1F2B44", lt2: "EEF2F7",
  accent1: "0E9F9A", accent2: "E4572E", accent3: "F2A541", accent4: "5B6C8F", accent5: "9DB4D3", accent6: "A3A9B5",
  hlink: "0E9F9A", folHlink: "5B6C8F",
};
const THEME = { name: "KESTREL Crash", headFontFace: "Cambria", bodyFontFace: "Calibri", colors: HEX };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "KESTREL: Crash-Aware Regime Modelling";
pres.author = "BigShort Winners";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.5, w: 12.1, h: 1.4, fontSize: 44, bold: true, color: C.background1, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.95, w: 12.1, h: 0.8, fontSize: 20, color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.8, fontSize: 32, bold: true, color: C.text2, valign: "middle" }, text: "" } },
    { text: { text: "KESTREL · BigShort Winners", options: { x: 0.5, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: C.accent4 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4 },
});

function title(slide, t) { slide.addText(t, { placeholder: "title" }); }
function txt(slide, t, o) { slide.addText(t, Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0.05 }, o)); }
function card(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name,
    shadow: { type: "outer", color: "1F2B44", opacity: 0.18, blur: 6, offset: 2, angle: 90 } });
}
function badge(slide, x, y, label, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill }, line: { color: fill }, objectName: name });
  slide.addText(label, { isTextBox: true, x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1, margin: 0 });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: "1F2B44", valAxisLabelColor: "5B6C8F", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 11, titleFontSize: 13,
  titleColor: "1F2B44", valGridLine: { color: "DDE3EC", size: 0.5 }, catGridLine: { style: "none" } };

// ---------- 1. Title ----------
pres.addSection({ title: "Story" });
let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Story" });
s.addText("Surviving a 6× Volatility Crash", { placeholder: "title" });
s.addText("Crash-aware regime modelling: leak-free features, Monte Carlo crashes, robust losses, risk-sized positions", { placeholder: "body" });
[["53.0", "Round-3 leaderboard score"], ["+12.7%", "Return while the market fell 9.2%"], ["1.8%", "Maximum drawdown in the crash"]].forEach(([v, l], i) => {
  const x = 0.6 + i * 4.1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.35, w: 3.7, h: 2.0, rectRadius: 0.08, fill: { color: C.text1 }, line: { color: C.accent1, width: 1 }, objectName: `title-stat-${i}` });
  s.addText(v, { isTextBox: true, x: x + 0.25, y: 4.5, w: 3.2, h: 1.0, fontSize: 48, bold: true, color: C.accent1, fontFace: "Cambria", margin: 0 });
  s.addText(l, { isTextBox: true, x: x + 0.25, y: 5.5, w: 3.2, h: 0.6, fontSize: 15, color: C.background1, margin: 0 });
});
s.addNotes("Presenter: Divisha (about 20 s). Single asset, three rounds; the scored period is a crash with volatility up to 6.4 times the labelled median and no labelled precedent. Headline: 53.0 on the leaderboard, +12.7% while the market fell 9.2%, 1.8% maximum drawdown. Return and drawdown figures come from a post-scoring analysis; nothing from the scored period was used to build the submission.");

// ---------- 2. Architecture ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Story" });
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
  const fill = i === 2 || i === 3 ? C.accent1 : C.text2;
  card(s, x, 1.55, bw, 1.25, fill, `arch-box-${i}`);
  s.addText(h, { isTextBox: true, x: x + 0.1, y: 1.6, w: bw - 0.2, h: 1.15, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0 });
  txt(s, d, { x, y: 2.95, w: bw, h: 1.5, fontSize: 14, color: C.text2 });
  if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + bw + 0.02, y: 2.175, w: gap - 0.04, h: 0, line: { color: C.accent4, width: 1.5, endArrowType: "triangle" }, objectName: `arch-arrow-${i}` });
});
card(s, 0.5, 4.75, 12.3, 1.85, C.background2, "arch-callout");
txt(s, [
  { text: "Rounds 1–2: ", options: { bold: true } }, { text: "next-day direction. No model beat the prior on real data, but MC-trained models stopped blowing up on crash inputs.", options: { breakLine: true } },
  { text: "Round 3: ", options: { bold: true } }, { text: "signal in [−1, 1] scored as a portfolio (Sharpe, money made, drawdown, cost). Robustness became the edge.", options: {} },
], { x: 0.8, y: 4.95, w: 11.7, h: 1.5, fontSize: 16, paraSpaceAfter: 8 });
s.addNotes("Presenter: Divisha (30 s). Walk left to right. The teal layers (regimes, Monte Carlo crashes, robust losses) are our core contribution: the scored period is a crash that never appears in labelled history, so we manufactured crashes and optimised the worst regime instead of the average.");

// ---------- 3. Feature engineering ----------
pres.addSection({ title: "Rubric" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Features: volatility ratios carry the signal, with zero lookahead");
const fRows = [
  ["Group", "Examples", "Why"],
  ["Vol ratios", "return ÷ σ, momentum ÷ σ√5, range ÷ σ", "Same scale in calm and crash"],
  ["Momentum / trend", "open/SMA10, SMA10/SMA30, momentum_5", "Trend state, scale-free"],
  ["Lag returns", "open-to-open 1, 3, 5 days; flow and volume lags", "Short memory"],
  ["Bar shape", "(high−low)/open, wicks, close location", "Intraday pressure"],
  ["Same-row close", "close[t] = close[t−5]·(1+mom5[t])", "Rows ≤ t only; 0.013% error"],
];
s.addTable(fRows.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : "161B26", fill: { color: i === 0 ? "1F2B44" : (i % 2 ? "FFFFFF" : "EEF2F7") } } }))),
  { x: 0.5, y: 1.35, w: 6.6, colW: [1.55, 3.0, 2.05], fontSize: 13, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "DDE3EC" }, rowH: 0.62 });
txt(s, [{ text: "No lookahead: ", options: { bold: true } }, { text: "scaler and cutoffs fit on training rows only; tests prove perturbing any later row never changes a feature at t." }],
  { x: 0.5, y: 5.4, w: 6.6, h: 1.1, fontSize: 14 });
const NICE = { osma10_vol: "open/SMA10 ÷ σ", mom_vol: "momentum ÷ σ√5", rsi_14: "RSI-14", ret1_vol: "1-day return ÷ σ", smagap_vol: "SMA10/SMA30 ÷ σ",
  order_flow_lag2: "order flow, lag 2", sma10_sma30: "SMA10/SMA30", ret_1d_c: "1-day return", volume_z_20_lag2: "volume z, lag 2", momentum_5: "momentum, 5-day" };
const imp = { names: D.importance.names.map(n => NICE[n] || n), values: D.importance.values };
s.addChart(pres.charts.BAR, [{ name: "|standardised coefficient|", labels: imp.names.slice().reverse(), values: imp.values.slice().reverse() }], Object.assign({
  x: 7.4, y: 1.3, w: 5.4, h: 5.5, barDir: "bar", showTitle: true, title: "Feature importance (final model, |std. coefficient|)",
  chartColors: ["0E9F9A"], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: "1F2B44", showLegend: false, valAxisHidden: true, dataLabelFormatCode: "0.00",
}, chartText));
s.addNotes("Presenter: Friday (40 s). Rubric 1. Our strongest inputs are volatility ratios: open-to-SMA gap and momentum divided by volatility. Dividing by volatility is what keeps inputs inside the training range when volatility is 6 times higher; without it standardised inputs reached |z| = 24. Standardise on training rows, clip at ±5. The close is rebuilt from the same row only; tomorrow's close is never reconstructed. Importance is the absolute standardised logistic-regression coefficient on the final MC-augmented fit.");

// ---------- 4. Validation ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Validation: chronological, purged, rules fixed before results");
const x0 = 0.9, x1 = 7.6, tmin = 25, tmax = 900;
const sx = t => x0 + (t - tmin) / (tmax - tmin) * (x1 - x0);
const folds = [[449, 451, 500], [499, 501, 550], [549, 551, 600], [599, 601, 650], [649, 651, 700], [699, 701, 750], [749, 751, 800], [799, 801, 850]];
folds.forEach(([te, v0, v1], i) => {
  const y = 1.45 + i * 0.42;
  s.addShape(pres.shapes.RECTANGLE, { x: sx(25), y, w: sx(te) - sx(25), h: 0.28, fill: { color: C.accent5 }, line: { color: C.accent5 }, objectName: `fold-train-${i}` });
  s.addShape(pres.shapes.RECTANGLE, { x: sx(v0), y, w: sx(v1) - sx(v0), h: 0.28, fill: { color: i === 7 ? C.accent2 : C.accent3 }, line: { color: i === 7 ? C.accent2 : C.accent3 }, objectName: `fold-val-${i}` });
});
s.addShape(pres.shapes.RECTANGLE, { x: sx(851), y: 1.45 + 8 * 0.42, w: sx(900) - sx(851), h: 0.28, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "fold-predict" });
txt(s, "Train", { x: 0.9, y: 5.05, w: 1.0, h: 0.35, color: C.accent4 });
txt(s, "Validate", { x: 1.9, y: 5.05, w: 1.2, h: 0.35, color: "B07A1E" });
txt(s, "Reserved (once)", { x: 3.1, y: 5.05, w: 1.9, h: 0.35, color: C.accent2 });
txt(s, "Predict 851–900", { x: 5.0, y: 5.05, w: 2.4, h: 0.35, color: C.text2 });
txt(s, "One-row purge before every block (label t uses close t+1). Scaler, regime cutoffs and the crash simulator are refit inside each fold on training rows only.",
  { x: 0.5, y: 5.55, w: 7.1, h: 1.2, fontSize: 14 });
[["19", "round-2 configurations: LR and LSTM × 5 loss arms, plus the prior baseline"], ["Fixed", "hyperparameters; every selection rule written before running"], ["Stop", "round 2: prior baseline won the rule, so we did not force a model"]].forEach(([v, l], i) => {
  const y = 1.4 + i * 1.8;
  card(s, 8.1, y, 4.7, 1.55, C.background2, `val-card-${i}`);
  s.addText(v, { isTextBox: true, x: 8.3, y: y + 0.15, w: 1.5, h: 1.25, fontSize: 36, bold: true, color: C.accent1, fontFace: "Cambria", valign: "middle", margin: 0 });
  txt(s, l, { x: 9.8, y: y + 0.2, w: 2.85, h: 1.2, fontSize: 14, valign: "middle" });
});
s.addNotes("Presenter: Navjyot (40 s). Rubric 2. Seven expanding folds plus a reserved block, never shuffled, with a one-row purge. We compared logistic regression, gradient boosting, Extra Trees and an attention LSTM under identical folds and metrics (accuracy, balanced accuracy, AUC, log loss, Brier, calibration). Every selection rule was written before results, which guards against leaderboard overfitting. In round 2 the training prior won, so we stopped instead of picking a model.");

// ---------- 5. Loss regimes ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Loss regimes: optimise the worst market state, not the average");
const arms = [
  ["A", "Standard", "Mean log loss: the reference"],
  ["B", "Worst-regime", "(1−λ)·mean + λ·max over regimes; λ ∈ {0.25, 0.5, 0.75}"],
  ["C", "Regime-weighted", "Weight 1.5–3× on high-vol and high-RSI rows"],
  ["D", "MC augmented", "Real rows + simulated crashes at 25% weight"],
  ["E", "MC + worst-regime", "D's data, B's objective, sim_crash as a regime"],
];
arms.forEach(([l, h, d], i) => {
  const x = 0.5 + i * 2.5;
  const hi = l === "D" || l === "E";
  card(s, x, 1.35, 2.3, 2.75, hi ? "E3F4F3" : C.background2, `arm-card-${l}`);
  badge(s, x + 0.2, 1.55, l, hi ? C.accent1 : C.accent4, `arm-badge-${l}`);
  txt(s, h, { x: x + 0.2, y: 2.15, w: 1.95, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  txt(s, d, { x: x + 0.2, y: 2.6, w: 1.95, h: 1.4, fontSize: 14 });
});
card(s, 0.5, 4.4, 12.3, 2.35, C.text2, "epigraph-box");
s.addText("min (1−λ)·L̄ + λ·t + η·‖w‖²/2    s.t.   L_g ≤ t  for every regime g",
  { isTextBox: true, x: 0.8, y: 4.55, w: 11.7, h: 0.6, fontSize: 18, fontFace: "Cambria", color: C.background1, margin: 0 });
txt(s, [
  { text: "Exact max, not a soft approximation: SLSQP with analytic gradients; all 15 fits converged (violation ≤ 1e-12)", options: { bullet: true, breakLine: true } },
  { text: "η = 1/(n·C), so λ = 0 equals sklearn logistic regression (tested to 1e-10); LSTM uses the exact full-batch group max", options: { bullet: true, breakLine: true } },
  { text: "Regimes: high/low vol, high/low RSI (cutoffs per fold) + sim_crash", options: { bullet: true } },
], { x: 0.8, y: 5.2, w: 11.7, h: 1.45, fontSize: 14, color: C.background1, paraSpaceAfter: 4 });
s.addNotes("Presenter: Divisha (40 s). Five training arms on identical folds, so differences come from the loss alone. B is Group-DRO style: penalise the worst regime's loss. We solve the exact max with an epigraph constraint rather than a softmax approximation, so we know which regime is genuinely worst. D adds Monte Carlo crashes; E combines both and treats the simulated crash as its own regime. Arm E was the best real-data model in round 2 and the LSTM we submitted in round 3.");

// ---------- 6. Monte Carlo ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Monte Carlo crash regime: train on crashes history never showed us");
const mc = [
  "Start from a random training day; 60-day real burn-in",
  "Block-bootstrap training returns (5-day blocks), rescaled to 3× or 5× median volatility",
  "Inject 1–2 single-day drops of −8% to −15%",
  "Rebuild OHLC; recompute SMA, momentum, vol and RSI with verified dataset formulas",
  "Resample volume and order flow from high-vol training days",
  "200 windows × 50 days = 10,000 rows per fold; separate seeds for training and stress test",
];
mc.forEach((t, i) => {
  const y = 1.35 + i * 0.88;
  badge(s, 0.5, y + 0.05, String(i + 1), i === 5 ? C.accent2 : C.accent1, `mc-step-${i}`);
  txt(s, t, { x: 1.15, y, w: 5.6, h: 0.8, fontSize: 14, valign: "middle" });
});
const keys = ["LR_A", "LR_D_mc", "LR_E_mc_lam0.5", "LSTM_A", "LSTM_E_mc_lam0.5"];
const lab = ["LR standard", "LR + MC", "LR MC+worst", "LSTM standard", "LSTM MC+worst"];
s.addChart(pres.charts.BAR, [
  { name: "Real validation", labels: lab, values: keys.map(k => D.r2_real[k]) },
  { name: "Simulated crash (synthetic)", labels: lab, values: keys.map(k => D.r2_stress[k]) },
], Object.assign({
  x: 7.0, y: 1.3, w: 5.8, h: 4.6, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Round 2 log loss (lower is better)",
  chartColors: ["9DB4D3", "E4572E"], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: "1F2B44", dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "b", valAxisMinVal: 0.6, valAxisMaxVal: 1.3, valAxisLabelFormatCode: "0.0",
}, chartText));
card(s, 7.0, 6.05, 5.8, 0.75, "FBE7E1", "mc-callout");
txt(s, [{ text: "MC training cut crash-case loss 1.22 → 0.70", options: { bold: true } }, { text: " (synthetic labels: robustness, not skill)" }],
  { x: 7.2, y: 6.12, w: 5.5, h: 0.62, fontSize: 14, valign: "middle" });
s.addNotes("Presenter: Navjyot (40 s). Labelled volatility never exceeds 0.0204; the scored block reaches 0.074. So we simulate crashes from each fold's training rows only: real starting state, bootstrapped returns scaled to 3x or 5x volatility, injected shocks, indicators recomputed with formulas we verified against the dataset. One seed trains, an independent seed stress-tests. The standard logistic regression blew up on crash inputs (log loss 1.22); MC-trained versions stayed near 0.69 to 0.70. Stress labels are synthetic, so this measures robustness, not prediction skill.");

// ---------- 7. Flash crash + risk ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Flash-crash adaptation: detect, size, control");
const rows = [
  ["Detect", "volatility_20 above anything in training → crash regime; switch from long-bias blend to the crash-trained ensemble"],
  ["Size", "signal = clip(5·(2p−1)) × min(1, σ_median/σ_t): exposure shrinks 3–6× automatically in the crash"],
  ["Control", "Backtest net of 10 bps per unit change; drawdown caps 15% real / 25% MC; turnover tracked"],
];
rows.forEach(([h, d], i) => {
  const y = 1.4 + i * 1.75;
  badge(s, 0.5, y, String(i + 1), i === 0 ? C.accent2 : C.accent1, `risk-badge-${i}`);
  txt(s, h, { x: 1.15, y: y - 0.02, w: 5.2, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  txt(s, d, { x: 1.15, y: y + 0.45, w: 5.4, h: 1.15, fontSize: 14 });
});
const pol = D.policies;
const pk = ["P0_long_full", "P1_long_volscaled", "P3_blend", "P2_model_volscaled", "P6_trend_volscaled", "P5_regime_switch"];
const pl = ["Full long", "Vol-scaled long", "Blend", "Model only", "Trend", "Regime switch ✓"];
s.addChart(pres.charts.BAR, [
  { name: "Real walk-forward Sharpe", labels: pl, values: pk.map(k => pol[k][0]) },
  { name: "MC crash Sharpe", labels: pl, values: pk.map(k => pol[k][1]) },
], Object.assign({
  x: 6.9, y: 1.3, w: 5.9, h: 4.7, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Policy backtest, net of 10 bps (8 real blocks, 1,600 crash paths)",
  chartColors: ["1F2B44", "0E9F9A"], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: "1F2B44", dataLabelFormatCode: "0.0",
  showLegend: true, legendPos: "b", valAxisLabelFormatCode: "0.0",
}, chartText));
txt(s, "Rule: maximise ½ real + ½ crash Sharpe within the drawdown caps. Only the regime switch is positive on both.", { x: 6.9, y: 6.15, w: 5.9, h: 0.65, fontSize: 14, italic: true, color: C.text2 });
s.addNotes("Presenter: Friday (40 s). Rubric 3. Detection: when volatility leaves the training range we stop trusting the long bias and hand control to the crash-trained ensemble. Sizing: positions scale inversely with volatility, so risk per day stays roughly constant. Control: every policy is judged net of costs, with drawdown caps, on real walk-forward blocks and on 1,600 simulated crashes. Full long has the best calm-market Sharpe but loses in crashes; the regime switch is the only policy positive on both. This is Sharpe versus raw PnL in practice.");

// ---------- 8. Results ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Results: +12.7% with 1.8% drawdown while the market fell 9.2%");
const eq = D.equity;
const ser = ["v3 (53.0)", "v1 (35.8)", "v4 (36.4)", "Buy & hold"];
s.addChart(pres.charts.LINE, ser.map(k => ({ name: k, labels: D.eq_labels, values: eq[k] })), Object.assign({
  x: 0.5, y: 1.3, w: 8.4, h: 5.5, showTitle: true, title: "Cumulative return, days 852–900 (%), net of 10 bps",
  chartColors: ["0E9F9A", "F2A541", "5B6C8F", "E4572E"], lineSize: 2, lineDataSymbol: "none", showLegend: true, legendPos: "b",
  valAxisLabelFormatCode: "0", catAxisLabelFrequency: 6,
}, chartText));
[["53.0", "Leaderboard: v3 (v1 35.8, v4 36.4)"], ["4.75", "Sharpe in the crash (v1 1.65)"], ["73%", "Days the v3 call was right"]].forEach(([v, l], i) => {
  const y = 1.3 + i * 1.85;
  card(s, 9.2, y, 3.6, 1.6, i === 0 ? C.text2 : C.background2, `res-card-${i}`);
  s.addText(v, { isTextBox: true, x: 9.4, y: y + 0.1, w: 3.2, h: 0.85, fontSize: 40, bold: true, fontFace: "Cambria", color: i === 0 ? C.accent1 : C.text2, margin: 0 });
  txt(s, l, { x: 9.4, y: y + 0.95, w: 3.2, h: 0.55, fontSize: 14, color: i === 0 ? C.background1 : C.text1 });
});
s.addNotes("Presenter: Divisha (30 s). v3, the regime-switching hybrid with the MC-trained worst-group LSTM, scored 53.0. In the post-scoring analysis it made 12.7% while buy-and-hold lost 9.2%, with a 1.8% maximum drawdown and the right direction on 73% of days. Be explicit: these curves use closes rebuilt after scoring; nothing from the scored period was used to build any submission.");

// ---------- 9. Failure case & limitations ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Rubric" });
title(s, "Failure case and limitations we can defend");
card(s, 0.5, 1.35, 6.0, 5.4, "FBE7E1", "fail-card");
txt(s, "Why smoothing (v4) dropped to 36.4", { x: 0.75, y: 1.5, w: 5.5, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
txt(s, [
  { text: "The crash reversed daily (autocorrelation −0.16 vs +0.13 in history); yesterday's call was right only 29% of days, and smoothing mixes it in", options: { bullet: true, breakLine: true } },
  { text: "Smoothing shrank positions under the organizer's 0.05 flat threshold: 38 of 50 days flat", options: { bullet: true, breakLine: true } },
  { text: "The score rewards money made, not only Sharpe", options: { bullet: true, breakLine: true } },
  { text: "Our history and simulator both trend, so they favoured smoothing", options: { bullet: true } },
], { x: 0.75, y: 2.1, w: 5.5, h: 4.5, fontSize: 14, paraSpaceAfter: 8 });
card(s, 6.8, 1.35, 6.0, 5.4, C.background2, "limits-card");
txt(s, "Assumptions and limitations", { x: 7.05, y: 1.5, w: 5.5, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
txt(s, [
  { text: "Target inferred: next-close direction (849/849 rows)", options: { bullet: true, breakLine: true } },
  { text: "Same-row timing: indicators at t are known when trading t", options: { bullet: true, breakLine: true } },
  { text: "No real directional edge in calm markets (rounds 1–2)", options: { bullet: true, breakLine: true } },
  { text: "Simulator choices (3×/5× vol, −8% to −15% shocks) are designed, not estimated", options: { bullet: true, breakLine: true } },
  { text: "50-day blocks: Sharpe standard error ≈ 2.2", options: { bullet: true } },
], { x: 7.05, y: 2.1, w: 5.5, h: 4.5, fontSize: 14, paraSpaceAfter: 8 });
s.addNotes("Presenter: Navjyot (30 s). Rubric 4: state assumptions, limitations and failure cases. Our second submission smoothed v3 and dropped to 36.4. The post-mortem found three causes: the crash reversed daily, smoothing pushed positions below the organizer's flat threshold, and the score rewards money made. Our backtest favoured smoothing because history and the simulator trend. Fixes: reversal scenarios in the simulator and no smoothing at high volatility.");

// ---------- 10. Future scope ----------
pres.addSection({ title: "Future" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Future" });
title(s, "Future scope: technically sound extensions of this architecture");
const fut = [
  ["MC risk engine", "Regime-switching and jump-diffusion paths, including reversal crashes; report VaR/CVaR, drawdown and ruin probability; CVaR-constrained sizing", C.accent1],
  ["RL agent in the MC crash simulator", "Not on the 50 real crash rows (too few, would fail in calm markets). Train in the simulator with calm, crash, trend and reversal regimes; reward = differential Sharpe − cost − drawdown", C.accent1],
  ["Regime detection", "Hidden Markov or change-point models to detect volatility spikes and concept drift, replacing the fixed volatility threshold", C.accent4],
  ["Generative crash scenarios", "TimeGAN or diffusion scenarios to widen the simulator; validate tail fidelity, a known weakness", C.accent4],
  ["Overfitting-aware validation", "Combinatorial purged CV and probability of backtest overfitting before any policy is trusted", C.accent4],
  ["Autocorrelation-aware smoothing", "Smooth only when recent return autocorrelation (rows ≤ t) is positive: the v4 lesson", C.accent4],
];
fut.forEach(([h, d, col], i) => {
  const cx = 0.5 + (i % 3) * 4.15, cy = 1.35 + Math.floor(i / 3) * 2.75;
  card(s, cx, cy, 3.95, 2.55, i < 2 ? "E3F4F3" : C.background2, `future-card-${i}`);
  badge(s, cx + 0.2, cy + 0.2, String(i + 1), col, `future-badge-${i}`);
  txt(s, h, { x: cx + 0.85, y: cy + 0.2, w: 2.95, h: 0.55, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
  txt(s, d, { x: cx + 0.2, y: cy + 0.85, w: 3.6, h: 1.6, fontSize: 14 });
});
s.addNotes("Presenter: Friday (30 s). 1 and 2 are the team's proposals, reframed to be technically sound. Monte Carlo for risk: extend our simulator with regime-switching, jump and reversal paths and report VaR, CVaR and drawdown distributions. RL: training only on the 50 real crash rows would overfit and fail in normal markets, so the sound version uses our Monte Carlo simulator as the environment with domain randomisation across regimes, so crashes are normal for the agent without forgetting calm markets, with costs and drawdown in the reward and an overfitting test before use. 3 to 6 come from current research: HMM regime detection, generative scenarios with tail checks, purged combinatorial validation, and autocorrelation-aware smoothing.");

// ---------- 11. Appendix ----------
pres.addSection({ title: "Appendix" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
title(s, "Appendix: reproducibility and sources");
txt(s, [
  { text: "Reproduce", options: { bold: true, breakLine: true } },
  { text: "python -m src.run_round3 --lstm   (v3 submission, byte-identical rerun)", options: { bullet: true, breakLine: true } },
  { text: "python -m src.run_round2   ·   python -m src.postmortem_round3", options: { bullet: true, breakLine: true } },
  { text: "76 automated leakage and correctness tests run before every experiment", options: { bullet: true, breakLine: true } },
  { text: "Pinned requirements-lock.txt; deterministic CPU PyTorch; dataset, manifest, simulator and submission hashes recorded", options: { bullet: true } },
], { x: 0.5, y: 1.35, w: 12.3, h: 2.4, fontSize: 14, paraSpaceAfter: 4 });
card(s, 0.5, 3.95, 12.3, 2.85, C.background2, "sources-card");
txt(s, [
  { text: "Future-scope sources (web search; no Reddit threads surfaced)", options: { bold: true, breakLine: true } },
  { text: "Gort et al. 2022, DRL backtest overfitting: arxiv.org/abs/2209.05559", options: { bullet: true, breakLine: true } },
  { text: "Regime-conditioned diffusion, CVaR portfolios (MARCD): arxiv.org/pdf/2510.10807", options: { bullet: true, breakLine: true } },
  { text: "Tail risk under jump-diffusion, CVaR: arxiv.org/abs/2607.00883", options: { bullet: true, breakLine: true } },
  { text: "Generative synthetic financial data (TimeGAN, VAE): arxiv.org/abs/2512.21798", options: { bullet: true, breakLine: true } },
  { text: "HMM regime-adaptive trading: blog.quantinsti.com/regime-adaptive-trading-python", options: { bullet: true } },
], { x: 0.75, y: 4.1, w: 11.8, h: 2.6, fontSize: 12, paraSpaceAfter: 2 });
s.addNotes("Backup slide for Q&A; not presented in the 3–5 minute slot.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
