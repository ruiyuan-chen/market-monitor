# Market Monitor

A free, self-updating website about the U.S. stock market, written for students. It covers:

- **Overview**: S&P 500, Nasdaq, Dow, Russell 2000 and VIX, plus an S&P 500 chart with its 200-day average, drawdown from the record high, and realized vs. implied volatility
- **Sectors**: heatmap of the 11 S&P 500 sectors (1 day to 1 year)
- **Largest firms**: top 20 U.S. companies by market cap, with P/E, forward P/E, dividend yield and 52-week range
- **Rates**: fed funds rate, the Treasury curve, curve slope, real yields, breakevens, high-yield spread and mortgage rates
- **Economy**: CPI, core CPI, core PCE, unemployment, payrolls and GDP, with recession shading
- **Cross-asset**: oil, gold, the dollar index, EUR/USD and Bitcoin
- **Valuation**: Shiller CAPE since 1881, earnings yield and excess CAPE yield
- **Factors**: Fama-French 5 factors + momentum, with growth-of-$1 charts
- **Risk model**: Ryan Chen's drawdown-risk model, retrained every day. It shows today's probability of a 3%+ S&P 500 drop within 5 and 21 trading days, an out-of-sample track record and scorecard, what drives the reading, and the author's evaluation

Every section has a **Learn** panel explaining the numbers. Every chart has a **data table and CSV download**.

## How it works

```
GitHub Actions (weekdays, 5:30 pm ET)
  └─ scripts/update_data.py ── Yahoo Finance · FRED · Shiller · Ken French
        └─ data/*.json ── committed to the repo
              └─ index.html + assets/ ── published on GitHub Pages
```

No server, no database and no paid API. If one source fails, that section keeps its previous data and the
"About the data" panel says so. The rest of the site still updates.

---

## Setup (about 10 minutes, all in the browser)

1. **Create a repository.** On GitHub, click **New repository**. Name it, for example, `market-monitor`,
   make it **Public**, and click **Create repository**.

2. **Upload the files.** On the new repository's page, click **uploading an existing file**. Unzip
   `market-monitor.zip` on your computer, then drag **everything inside** the `market-monitor` folder into the
   page, including the `.github` folder. Click **Commit changes**.
   - Windows: File Explorer shows `.github` normally.
   - Mac: Finder hides folders whose names start with a dot. Press **Cmd + Shift + .** in Finder to show them.
   - Check: the repository should now contain `.github/workflows/update.yml`. If it doesn't, click
     **Add file → Create new file**, type `.github/workflows/update.yml` as the name, paste in the contents
     of that file from the zip, and commit.

3. **Turn on Pages.** Go to **Settings → Pages**. Under *Build and deployment → Source*, choose
   **GitHub Actions**.

4. **Run the first update.** Open the **Actions** tab (click *I understand… enable them* if GitHub asks).
   Choose **Update market data** on the left, then **Run workflow → Run workflow**. It takes 3–5 minutes.
   When it shows a green check, your site is live at

   `https://<your-github-username>.github.io/market-monitor/`

   From then on it updates itself every weekday evening.

### Optional

- **FRED API key** (makes the FRED downloads more robust): get a free key at
  <https://fred.stlouisfed.org/docs/api/api_key.html>, then go to **Settings → Secrets and variables →
  Actions → New repository secret**, name it `FRED_API_KEY` and paste the key.
- **Personalize:** edit `assets/js/config.js` (title, your name, link to your academic site, course name).
- **Change what's tracked:** the lists at the top of `scripts/update_data.py` (indices, sector funds,
  cross-asset tickers, the candidate pool for the "Largest firms" ranking, FRED series).
- **Link it from your academic site**, e.g. a "Market Monitor" item in the Teaching page.

---

## Files

| Path | What it is |
|---|---|
| `index.html` | The page, including all the **Learn** text (edit freely) |
| `assets/css/style.css` | Styles. Colors and fonts are variables at the top (navy masthead, green gains, red losses) |
| `assets/js/app.js` | Draws the tables and charts from `data/*.json` |
| `assets/js/config.js` | Site title, maintainer name and links |
| `assets/vendor/chart.umd.min.js` | Chart.js 4.5.1 (MIT license), bundled so the site has no outside dependencies |
| `scripts/update_data.py` | Downloads and processes the data |
| `scripts/risk_model.py` | The drawdown-risk model (features, walk-forward test, today's prediction) |
| `model/universe.txt` | The ~490 stock tickers the model uses (one per line) |
| `content/commentary.md` | **Your evaluation**, shown beside the model's signal. Edit it on GitHub whenever you like |
| `data/model_log.json` | Append-only record of each day's live prediction, scored once the outcome is known |
| `data/model_oos.json` | Every out-of-sample prediction since 2020 (model and volatility baseline), used by the backtest |
| `data/*.json` | The data the page reads (written by the workflow; don't edit by hand) |
| `.github/workflows/update.yml` | The schedule: fetch → commit → publish |
| `scripts/model_experiment.py` | Compares model variants (feature sets, history length) out of sample; run from Actions |
| `.github/workflows/experiment.yml` | The manual **Model experiment** workflow |
| `scripts/backtest.py` | Simulates simple trading rules on the model's out-of-sample predictions; run from Actions |
| `.github/workflows/backtest.yml` | The manual **Strategy backtest** workflow |
| `tests/` | Offline tests for the data parsers and the model |

## Risk model

`scripts/risk_model.py` runs your notebook's model every weekday as part of the same workflow (it adds
about 3–5 minutes). It downloads prices for the tickers in `model/universe.txt` and the S&P 500,
rebuilds the features, tests the model out of sample, and writes `data/model.json`. The website never
shows the code, only the results.

**What it computes**

1. Clusters of co-moving stocks (PCA on the return-correlation matrix, then k-means with 10 clusters).
2. *Financial turbulence*: for each cluster, the residual volatility of a 21-day regression of the S&P 500
   on the cluster, times sqrt(1 − correlation). An expanding PCA of these signals gives PC1/PC2.
3. *Internal turbulence*: the average 252-day z-score of six breadth measures.
   *Realized volatility* (RV21): the annualized standard deviation of the last 21 daily S&P 500 returns.
4. Target: the S&P 500 closes at least 3% below today's close within the next 5 (or 21) trading days.
5. Logistic regression, refit every day on all labeled history.

The feature code was checked against a re-run of the notebook on `stock_details_5_years.csv`: cluster
returns, turbulence signals, breadth measures and internal turbulence match to machine precision.

**How it differs from the notebook, and why**

| Change | Why |
|---|---|
| Clusters are formed from the trailing 2 years of returns *as of each date*, not from the full sample | Full-sample clusters use future information. With them, the walk-forward 5-day AUC is about 0.66; with clusters fixed on early data only, it falls to about 0.52; re-forming them monthly on trailing data brings it back to about 0.64 with no look-ahead |
| The last 5/21 days have no label | The notebook's `astype("Int64")` turned "future unknown" into 0, so the newest days counted as "no drawdown" |
| Walk-forward test: refit monthly, train only on labels known at the time | One 80/20 split has only 13 drawdown days in the test set, so its AUC is very noisy (0.75 in the notebook, 0.80 in a re-run) |
| Logistic regression with no resampling | SMOTE/oversampling inflates the probabilities (Brier score about 65% worse), which matters once a probability is shown on a website |
| PC changes are first differences, signs kept consistent | The percent change of a series that crosses zero reaches ±18 to ±30 in the sample |
| Correlations use pairwise-complete data | Filling missing returns with 0 shrinks the correlations of recently listed stocks |
| Realized volatility (RV21) is added as a seventh feature (Oct 2026) | Out of sample, volatility alone beat the six turbulence features (5-day AUC 0.67 vs 0.64). The scorecard still compares the model with volatility alone, so it shows whether the other features add anything |

**Settings** are in the `CONFIG` block at the top of `scripts/risk_model.py`: the history start, the
out-of-sample start, the horizons and thresholds, the feature list, and `"pc_change": "pct"` to restore
the notebook's percent changes.

**Run it on your own CSV** (the notebook's format: Date, Close, Volume, Company):

```bash
RISK_MODEL_PRICES_CSV=stock_details_5_years.csv RISK_MODEL_SPX_CSV=spx.csv python scripts/risk_model.py
```

`spx.csv` needs two columns, a date and the S&P 500 close (FRED's `SP500` download works).

**Experiments.** To compare variants without touching the site, open **Actions → Model experiment → Run workflow**. By default it tests the notebook's features, the live model (with RV21), adding the VIX, and VIX or RV21 alone, first on the live setup (daily data since 2016, tested from 2020) and then on a longer history (daily data since 2003, tested from 2008). The results table, with block-bootstrap ranges for each AUC difference, appears on the run's summary page after about 15–20 minutes. Edit `VARIANTS` in `scripts/model_experiment.py` to try other features.

**Strategy backtest.** To see what trading on the model would have done, open **Actions → Strategy backtest → Run workflow** (after the daily update has run at least once). It simulates buy and hold, the model's risk levels as in-or-out and scaled-exposure rules, the volatility-only model, VIX thresholds and volatility targeting, all long-only with trading costs and a simple tax treatment, and reports returns, drawdowns, trade counts and a bootstrap range for each rule's gap to buy and hold. The results appear on the run's summary page in about two minutes; the chart and daily series are in the run's artifact. The cost, tax rate and volatility target are inputs on the Run workflow form. This is a study tool, not investment advice.

**Your evaluation** lives in `content/commentary.md`. On GitHub, open the file, click the pencil icon,
edit, and commit. The site picks it up within a minute. It supports a `# Title` line, an
`Updated: YYYY-MM-DD` line, paragraphs, `- ` bullet points, `**bold**`, `*italic*` and `[links](https://…)`.

## Previewing on your own computer

```bash
pip install -r requirements.txt
python scripts/update_data.py          # fetch data into data/
python -m http.server 8000             # then open http://localhost:8000
python -m pytest -q                    # run the parser tests
```

(The page has to be served over http. Double-clicking `index.html` won't load the data.)

## Troubleshooting

- **A section says "kept previous data".** One source failed on the last run. Open **Actions**, click the
  latest run and expand **Fetch the latest data** to see the error. Yahoo Finance occasionally rate-limits
  GitHub's servers, and the next day's run usually works. If Yahoo changes something,
  `pip install -U yfinance` (the workflow always installs the latest version) normally fixes it.
- **The schedule stopped.** GitHub pauses scheduled workflows in repositories that have had no activity for
  60 days. If that happens, open **Actions → Update market data** and click **Enable workflow**.
- **The site didn't change after I edited a file.** Every push to `main` republishes the site. Wait a minute
  and hard-refresh the page (Ctrl + F5).
- **Shiller's data** is downloaded from <https://shillerdata.com/>. If its page layout changes, set
  `SHILLER_DIRECT` in `scripts/update_data.py` to the current `ie_data.xls` link.

## Data sources and terms

Yahoo Finance data comes through the unofficial open-source `yfinance` library and is meant for research and
educational use. FRED® data: Federal Reserve Bank of St. Louis. The high-yield spread is from ICE Data
Indices, LLC, via FRED. CAPE: Robert J. Shiller. Factor returns: Kenneth R. French Data Library. The site is
for teaching and research and is not investment advice. The risk model's output is a statistical estimate,
not a recommendation to buy or sell anything.
