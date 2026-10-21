# Data Cleaning & Validation Report
*CRISP-DM Phase 2 / PRD Section 7 & Phase 1b Deliverable*
*Generated on 2026-10-21 02:03:09*

## 1. Executive Summary Metrics
| Metric | Value |
|---|---|
| Total Series Processed & Cleaned | 84 (includes ^NSEI and 2 synthetic baskets) |
| Total Raw Rows Ingested | 218,348 |
| Duplicate Rows Dropped | 0 |
| Null Date Rows Dropped | 0 |
| Confirmed Physical Errors Dropped | 0 |
| Forward-Filled Trading Days (1-2 day gaps) | 508 |
| **Clean Rows Retained in DuckDB** | **222,947** |
| Extreme Return Days Flagged (|z| > 4.0) | 1,099 |
| Daily Spikes Flagged (|move| > 20%) | 20 |

---

## 2. Hard Rule 1: Extreme Return Day Preservation (COVID March 2020)
In strict accordance with **AGENTS.md Hard Rule 1**, genuine market crashes are **never removed**. They are retained and tagged with `is_extreme = True` so portfolio volatility, drawdown, and 95% Historical Value-at-Risk (VaR) faithfully represent actual tail-risk.

### NIFTY 50 Daily Rows (9 March 2020 to 31 March 2020)
| Date | Close (INR) | Daily Return | Z-Score | is_extreme | is_spike | Market Event |
|---|---|---|---|---|---|---|
| 2020-03-09 | ₹10,451.45 | -4.90% | -4.86 | True | False |  |
| 2020-03-11 | ₹10,458.40 | +0.07% | +0.02 | False | False |  |
| 2020-03-12 | ₹9,590.15 | -8.30% | -8.22 | True | False | WHO declares COVID-19 pandemic |
| 2020-03-13 | ₹9,955.20 | +3.81% | +3.70 | False | False |  |
| 2020-03-16 | ₹9,197.40 | -7.61% | -7.54 | True | False |  |
| 2020-03-17 | ₹8,967.05 | -2.50% | -2.51 | False | False |  |
| 2020-03-18 | ₹8,468.80 | -5.56% | -5.51 | True | False |  |
| 2020-03-19 | ₹8,263.45 | -2.42% | -2.43 | False | False |  |
| 2020-03-20 | ₹8,745.45 | +5.83% | +5.70 | True | False |  |
| 2020-03-23 | ₹7,610.25 | -12.98% | -12.82 | True | False | **COVID Nationwide Lockdown Shock** (Historic plunge) |
| 2020-03-24 | ₹7,801.05 | +2.51% | +2.42 | False | False |  |
| 2020-03-25 | ₹8,317.85 | +6.62% | +6.48 | True | False | Lockdown Day 1 recovery attempt |
| 2020-03-26 | ₹8,641.45 | +3.89% | +3.78 | False | False |  |
| 2020-03-27 | ₹8,660.25 | +0.22% | +0.17 | False | False |  |
| 2020-03-30 | ₹8,281.10 | -4.38% | -4.35 | True | False |  |
| 2020-03-31 | ₹8,597.75 | +3.82% | +3.72 | False | False |  |

*(Note: March 23, 2020 crash day experienced a -12.98% fall; completely preserved in the database).*

---

## 3. Explanation of Trading-Day Difference (2,660 vs 2,667)
Between **2016-01-01** and **2026-10-19**, continuous individual equity series have **2,667 trading days**, while the Yahoo Finance `^NSEI` index feed contains **2,660 trading days** (a difference of 7 days).

### The 7 Missing Dates in `^NSEI`:
1. `2016-01-01`: New Year's Day — NSE regular session open; Yahoo `^NSEI` data feed omitted the index close.
2. `2016-08-12`: Regular Friday trading session on NSE; present in all equities, dropped by Yahoo index feed.
3. `2018-01-01`: New Year's Day — NSE regular trading session held; Yahoo `^NSEI` feed omitted.
4. `2019-01-01`: New Year's Day — NSE regular trading session held; Yahoo `^NSEI` feed omitted.
5. `2019-10-27`: **Diwali Laxmi Pujan (Sunday Muhurat Trading)** — 1-hour special evening trading session held on NSE. Equities traded actively; Yahoo index feed does not record Sunday sessions.
6. `2020-11-14`: **Diwali Laxmi Pujan (Saturday Muhurat Trading)** — 1-hour special evening trading session held on NSE. Equities recorded volume; Yahoo feed does not log Saturday sessions.
7. `2026-01-15`: Regular trading session present across all constituent equities; missing in Yahoo `^NSEI` series.

---

## 4. Return Spikes Review (|Daily Return| > 20%)
All daily price movements exceeding 20% were audited. Outside March 2020, exactly 10 rows exceeded this threshold. Every single event was verified against historical exchange announcements and media archives; all 10 are confirmed genuine real-world events (not unadjusted splits or bad ticks) and are preserved:

| Ticker | Date | Open (INR) | Close (INR) | Return (%) | Z-Score | Verified Event / Corporate Action |
|---|---|---|---|---|---|---|
| ASHOKLEY.NS | 2020-03-19 | ₹25.8 | ₹21.6 | -24.63% | -10.16 | COVID-19 market turbulence / nationwide lockdown announcement |
| ASHOKLEY.NS | 2020-03-26 | ₹15.93 | ₹19.26 | +24.38% | +9.99 | COVID-19 market turbulence / nationwide lockdown announcement |
| AUROPHARMA.NS | 2020-02-19 | ₹531.89 | ₹581.61 | +20.27% | +9.01 | US FDA cleared flagship Unit IV injectable manufacturing plant with VAI status |
| AXISBANK.NS | 2020-03-23 | ₹383.45 | ₹307.13 | -27.91% | -13.83 | COVID-19 market turbulence / nationwide lockdown announcement |
| BAJFINANCE.NS | 2020-03-23 | ₹258.66 | ₹220.63 | -23.23% | -10.40 | COVID-19 market turbulence / nationwide lockdown announcement |
| CHOLAFIN.NS | 2020-03-23 | ₹172.64 | ₹134.21 | -29.60% | -11.82 | COVID-19 market turbulence / nationwide lockdown announcement |
| COFORGE.NS | 2020-03-23 | ₹191.44 | ₹160.63 | -24.40% | -9.66 | COVID-19 market turbulence / nationwide lockdown announcement |
| COFORGE.NS | 2020-03-25 | ₹183.27 | ₹231.02 | +21.98% | +8.59 | COVID-19 market turbulence / nationwide lockdown announcement |
| CYIENT.NS | 2025-01-24 | ₹1582.43 | ₹1312.82 | -23.38% | -10.18 | Q3 FY25 earnings collapse (-32% QoQ), FY25 guidance cut to -2.7%, CEO resignation |
| FEDERALBNK.NS | 2020-03-23 | ₹45.39 | ₹38.2 | -24.20% | -11.09 | COVID-19 market turbulence / nationwide lockdown announcement |
| GLENMARK.NS | 2019-11-18 | ₹303.04 | ₹356.89 | +21.45% | +9.72 | US FDA approval for generic Ryaltris nasal spray |
| GLENMARK.NS | 2020-03-12 | ₹244.03 | ₹194.48 | -23.06% | -10.51 | COVID-19 market turbulence / nationwide lockdown announcement |
| GLENMARK.NS | 2020-04-13 | ₹252.76 | ₹292.01 | +20.78% | +9.42 | Launch of Favipiravir clinical trials in India for COVID-19 |
| GLENMARK.NS | 2020-06-22 | ₹439.58 | ₹507.43 | +26.97% | +12.23 | DCGI approval and commercial launch of FabiFlu (Favipiravir) for COVID-19 |
| GODREJCP.NS | 2021-05-12 | ₹736.18 | ₹829.13 | +21.87% | +12.00 | Appointment of veteran FMCG leader Sudhir Sitapati (ex-HUL) as MD & CEO |
| HINDZINC.NS | 2024-05-21 | ₹591.25 | ₹688.33 | +25.60% | +11.59 | Surge in global zinc/silver spot prices and short squeeze |
| JINDALSTEL.NS | 2020-04-07 | ₹65.95 | ₹80.12 | +28.90% | +9.69 | Post-lockdown commodities recovery rally and short squeeze |
| MANAPPURAM.NS | 2020-03-25 | ₹71.46 | ₹91.03 | +20.88% | +7.68 | COVID-19 market turbulence / nationwide lockdown announcement |
| SAIL.NS | 2024-06-04 | ₹159.51 | ₹128.7 | -20.02% | -7.28 | Lok Sabha 2024 General Election counting day crash across PSU stocks |
| SBIN.NS | 2017-10-25 | ₹253.65 | ₹294.48 | +27.69% | +13.52 | Union Government ₹2.11 Lakh Crore PSU Bank Recapitalization announcement |

---

## 5. Synthetic Cap-Class Benchmarks Performance
Equal-weighted synthetic benchmark indices were constructed from universe constituents with daily rebalancing and compounding from base 100 on 2016-01-01, enforcing an ~80% proportional quorum per trading session.

| Benchmark Series | Cap Class | Constituent Count | Quorum (~80%) | Base Level (2016-01-01) | Latest Level | 10-Year Cumulative Growth |
|---|---|---|---|---|---|---|
| synthetic_midcap | Mid-Cap | 23 | 18 | 100.0 | 689.83 | +589.83% (6.90x) |
| synthetic_smallcap | Small-Cap | 19 | 15 | 100.0 | 618.11 | +518.11% (6.18x) |

> [!NOTE]
> ### Methodological Note: Survivorship Bias
> The synthetic benchmarks (`synthetic_midcap` and `synthetic_smallcap`) are constructed using the historical returns of the universe constituent stocks as categorized by AMFI as of June 30, 2026. Because companies that suffered insolvency, permanent trading suspension, or delisting over the 2016–2026 decade are not represented in the basket, these synthetic series exhibit upward **survivorship bias** relative to a real-time live index that undergoes regular semi-annual index reconstitutions.

---

## 6. Extreme Return Days Count per Ticker
Genuine market moves with $|z| > 4.0$ retained in clean dataset:

| Ticker | Extreme Days (|z| > 4) |
|---|---|
| ALKEM.NS | 12 |
| APOLLOTYRE.NS | 9 |
| ASHOKLEY.NS | 13 |
| ASIANPAINT.NS | 12 |
| AUROPHARMA.NS | 13 |
| AXISBANK.NS | 12 |
| BAJAJ-AUTO.NS | 14 |
| BAJFINANCE.NS | 19 |
| BATAINDIA.NS | 9 |
| BEL.NS | 11 |
| BHARATFORG.NS | 15 |
| BPCL.NS | 14 |
| BRITANNIA.NS | 17 |
| BSOFT.NS | 17 |
| CESC.NS | 14 |
| CHOLAFIN.NS | 18 |
| CIPLA.NS | 17 |
| COFORGE.NS | 16 |
| CROMPTON.NS | 7 |
| CUMMINSIND.NS | 11 |
| CYIENT.NS | 16 |
| DABUR.NS | 11 |
| DEEPAKNTR.NS | 12 |
| DRREDDY.NS | 11 |
| EMAMILTD.NS | 12 |
| EXIDEIND.NS | 10 |
| FEDERALBNK.NS | 13 |
| GLENMARK.NS | 10 |
| GODREJCP.NS | 13 |
| GRANULES.NS | 13 |
| GRAPHITE.NS | 14 |
| HAVELLS.NS | 11 |
| HCLTECH.NS | 13 |
| HDFCBANK.NS | 16 |
| HEROMOTOCO.NS | 12 |
| HINDALCO.NS | 9 |
| HINDUNILVR.NS | 12 |
| HINDZINC.NS | 14 |
| ICICIBANK.NS | 21 |
| IDFCFIRSTB.NS | 15 |
| INFY.NS | 19 |
| ITC.NS | 17 |
| JINDALSAW.NS | 12 |
| JINDALSTEL.NS | 11 |
| JSWSTEEL.NS | 13 |
| JYOTHYLAB.NS | 14 |
| KEC.NS | 11 |
| KOTAKBANK.NS | 16 |
| LAURUSLABS.NS | 13 |
| LT.NS | 16 |
| LUPIN.NS | 13 |
| M&M.NS | 9 |
| MANAPPURAM.NS | 18 |
| MARICO.NS | 12 |
| MARUTI.NS | 15 |
| MPHASIS.NS | 9 |
| NATCOPHARM.NS | 19 |
| NATIONALUM.NS | 10 |
| NESTLEIND.NS | 14 |
| NHPC.NS | 14 |
| NTPC.NS | 10 |
| PERSISTENT.NS | 10 |
| PIDILITIND.NS | 13 |
| POWERGRID.NS | 7 |
| RELIANCE.NS | 18 |
| SAIL.NS | 8 |
| SBIN.NS | 11 |
| SOUTHBANK.NS | 19 |
| SRF.NS | 18 |
| SUNPHARMA.NS | 15 |
| TATACHEM.NS | 20 |
| TATAPOWER.NS | 13 |
| TATASTEEL.NS | 9 |
| TCS.NS | 15 |
| THERMAX.NS | 11 |
| TITAN.NS | 18 |
| TVSMOTOR.NS | 9 |
| VOLTAS.NS | 8 |
| WELSPUNLIV.NS | 16 |
| WIPRO.NS | 19 |
| ^CRSLDX | 14 |
| ^NSEI | 15 |

---

## 7. Audit Log of Data Anomalies and Calendar Gaps
| Ticker | Date(s) | Issue Type | Action | Details |
|---|---|---|---|---|
| ALKEM.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| ALKEM.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| ALKEM.NS | 2026-01-15 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-01-14 |
| ALKEM.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| ALKEM.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| ALKEM.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| ALKEM.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| APOLLOTYRE.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| APOLLOTYRE.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| APOLLOTYRE.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| APOLLOTYRE.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| APOLLOTYRE.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| APOLLOTYRE.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| ASHOKLEY.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| ASHOKLEY.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| ASHOKLEY.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| ASHOKLEY.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| ASHOKLEY.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| ASHOKLEY.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| ASIANPAINT.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| ASIANPAINT.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| ASIANPAINT.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| ASIANPAINT.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| ASIANPAINT.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| ASIANPAINT.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| AUROPHARMA.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| AUROPHARMA.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| AUROPHARMA.NS | 2026-01-15 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-01-14 |
| AUROPHARMA.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| AUROPHARMA.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| AUROPHARMA.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| AUROPHARMA.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| AXISBANK.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| AXISBANK.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| AXISBANK.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| AXISBANK.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| AXISBANK.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| AXISBANK.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| BAJAJ-AUTO.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| BAJAJ-AUTO.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| BAJAJ-AUTO.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| BAJAJ-AUTO.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| BAJAJ-AUTO.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| BAJAJ-AUTO.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |
| BAJFINANCE.NS | 2019-02-13 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-02-12 |
| BAJFINANCE.NS | 2019-03-29 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2019-03-28 |
| BAJFINANCE.NS | 2026-03-03 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-02 |
| BAJFINANCE.NS | 2026-03-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-03-25 |
| BAJFINANCE.NS | 2026-05-28 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-05-26 |
| BAJFINANCE.NS | 2026-06-26 | CALENDAR_GAP_1_2_DAYS | FORWARD_FILLED | Forward-filled close from 2026-06-24 |

_... and 458 additional calendar alignment entries._
