# KESTREL presentation script (≈ 4 min 45 s)

About 680 spoken words at a calm pace. Slide numbers match `KESTREL_architecture.pptx`; the appendix (slide 14) is for Q&A only. Speakers are a suggestion, balanced across the team; swap as you like.

| Speaker | Slides | Time |
|---|---|---|
| Divisha | 1, 2, 3, 9, 10 | ≈ 1 min 30 s |
| Navjyot | 5, 6, 7, 11 | ≈ 1 min 45 s |
| Friday | 4, 8, 12, 13 | ≈ 1 min 30 s |

---

### Slide 1: Markets are uncertain (Divisha, 0:30)
Markets are uncertain, and the moments that matter most are **regime breaks**: when the rules the market followed yesterday stop working. In this challenge the final fifty days were a crash. Volatility jumped to **6.4 times** its normal level, far above anything in 850 days of history. Most models only ever see calm markets, and they break exactly here. So we built a **crash-aware trading signal engine**: one daily position between minus one and plus one, sized for risk, cost and drawdown.

### Slide 2: Headline (Divisha, 0:15)
The result: a leaderboard score of **53**. Through the crash our strategy returned **plus 12.7 percent** while the market fell 9.2, with a **maximum drawdown of only 1.8 percent**.

### Slide 3: Architecture (Divisha, 0:15)
Every layer answers one question: what happens at six times normal volatility? Leak-free data, causal features, regimes and simulated crashes, robust losses, a crash-aware model hybrid, and finally a risk-sized signal.

### Slide 4: Features (Friday, 0:30)
Our features use only today and earlier, never the future, and our tests prove it. The most important ones are **volatility ratios**: a return or momentum divided by current volatility. A 2 percent move means something very different in a calm market than in a crash; dividing by volatility puts both on the same scale, so the model still recognises the pattern at six-times volatility. We also clip every input, so no model sees numbers it was never trained on.

### Slide 5: Validation (Navjyot, 0:20)
We validated strictly in time order: seven expanding folds, a one-day gap before each test block, never shuffled. We compared logistic regression, gradient boosting, Extra Trees and an LSTM on identical folds, and wrote every selection rule **before** seeing results, so we couldn't fit to the leaderboard.

### Slide 6: Loss regimes (Navjyot, 0:30)
A **regime** is a market state: high or low volatility, high or low RSI. A normal model minimises its *average* error, which can hide a model that is terrible in stress. Our **worst-regime loss** also minimises the error in the worst regime, solved exactly rather than approximated. Arm E combines it with simulated crashes, where the crash is its own regime. That arm became our best model.

### Slide 7: Monte Carlo crash regime (Navjyot, 0:30)
History contains no crash, so we **manufactured** them. From each fold's training data only, we resample real returns, scale them to three or five times normal volatility, inject sudden drops of 8 to 15 percent, and recompute every indicator: ten thousand crash days per fold. One set trains the models; a separate set tests them. A standard model's crash loss was **1.22**; trained with Monte Carlo it stayed near **0.70**.

### Slide 8: Flash-crash adaptation (Friday, 0:30)
Three controls. **Detect:** when volatility exceeds anything seen in training, we switch to the crash-trained ensemble. **Size:** position shrinks as volatility rises, so daily risk stays roughly constant. **Control:** every strategy is judged after trading costs, with drawdown limits. **Sharpe** is return per unit of risk, not raw profit. Only our regime switch scored positive Sharpe on both real data and simulated crashes.

### Slide 9: Predictions in action (Divisha, 0:15)
Here is our signal against the price. As the crash hits, positions shrink, small ones go exactly flat, and direction flips with the daily reversals. Model probabilities stay sensible throughout: no garbage outputs.

### Slide 10: Results (Divisha, 0:15)
The green line is us: **plus 12.7 percent** while buy-and-hold lost 9.2, right on **73 percent** of days, Sharpe 4.75.

### Slide 11: Failure case (Navjyot, 0:25)
We also learned from a failure. Our second submission smoothed the signal and scored 36.4. The crash reversed direction almost daily, so smoothing made us late, and it shrank positions below the organiser's flat threshold. Our backtest favoured smoothing because history trends. The fix: simulate reversal crashes and don't smooth at high volatility.

### Slide 12: Analyst use (Friday, 0:15)
For a real analyst this is useful today: stress-test a portfolio against simulated crashes, get an early warning when markets leave known regimes, and add our sizing overlay to cap drawdowns. It is decision support, not autopilot.

### Slide 13: Future scope (Friday, 0:20)
Next we would extend the Monte Carlo engine into a full risk tool, with VaR, CVaR and reversal crashes, and train a **reinforcement-learning agent inside that simulator**, so crashes are routine for it without forgetting calm markets. Thank you; we're happy to take questions.

---

## Key terms (for Q&A; not read aloud)

| Term | One-line definition |
|---|---|
| volatility_20 | Standard deviation of the last 20 daily returns: how violently the price moves |
| Volatility ratio | A return or momentum divided by volatility, so calm and crash days are comparable |
| No lookahead | Each day's signal uses only that day and earlier; proven by tests that change future rows |
| Purge | One-day gap between training and test, because today's label depends on tomorrow's close |
| Regime | Market state from training-period quantiles: high/low volatility, high/low RSI, or simulated crash |
| Worst-regime loss | (1−λ)·average loss + λ·loss of the worst regime, solved exactly (Group-DRO style) |
| Monte Carlo crash | Synthetic crash path built from training data: resampled returns at 3–5× volatility plus sudden drops |
| Sharpe ratio | Average return ÷ volatility of returns, annualised: return per unit of risk |
| Max drawdown | Largest fall from a previous peak in portfolio value |
| Turnover / cost | Total change in position; each unit traded pays a cost, so flip-flopping is expensive |
| Regime switch | Trust the long-biased blend in normal volatility; switch to the crash-trained models above the training range |
| Flat threshold | The organisers treat any position of size 0.05 or less as no position |
