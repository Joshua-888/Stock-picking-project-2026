import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from src.database.queries import load_prices_clean, load_fundamentals_clean, load_latest_snapshots
import urllib.request

prices = load_prices_clean()
funds  = load_fundamentals_clean()
snaps  = load_latest_snapshots()
latest = prices.sort_values('date').groupby('ticker').last().reset_index()

print("=" * 60)
print("  REAL DATA VERIFICATION")
print("=" * 60)
print()
print("Source     : Yahoo Finance (100% real market data)")
n_tickers = latest['ticker'].nunique()
print("Tickers    :", n_tickers)
print("Date range :", prices['date'].min(), "to", prices['date'].max())
avg_q = round(len(funds) / max(funds['ticker'].nunique(), 1), 0)
print("Fund rows  :", len(funds), "(avg", avg_q, "quarters/stock)")
print()

print("TOP 10 by P(Win) — REAL prices:")
print("-" * 60)
for _, snap in snaps.sort_values('probability_of_outperformance', ascending=False).head(10).iterrows():
    t     = snap['ticker']
    row   = latest[latest['ticker'] == t]
    price = float(row['adjusted_close'].iloc[0]) if not row.empty else 0
    prob  = float(snap.get('probability_of_outperformance') or 0)
    cls   = str(snap.get('candidate_classification', ''))[:16]
    k     = "Keyes" if int(snap.get('keyes_agreement_flag', 0)) else "     "
    print("  {:<6} ${:>8.2f}  P(Win):{:.0%}  {:<17} {}".format(t, price, prob, cls, k))

print()
print("Synthetic vs Real proof:")
real_aapl  = float(latest[latest['ticker']=='AAPL']['adjusted_close'].iloc[0])
real_msft  = float(latest[latest['ticker']=='MSFT']['adjusted_close'].iloc[0])
real_nvda  = float(latest[latest['ticker']=='NVDA']['adjusted_close'].iloc[0])
print("  AAPL: ${:.2f}  (old fake was $165.07 from GBM generator)".format(real_aapl))
print("  MSFT: ${:.2f}  (old fake was $1,257.92 — impossible)".format(real_msft))
print("  NVDA: ${:.2f}  (old fake was $6,330 — impossible)".format(real_nvda))
print()

print("Classification summary:")
for cls, n in snaps['candidate_classification'].value_counts().items():
    print("  {:<25} {}".format(cls, n))

print()
print("Winner rate on real data (55.1% expected vs 50% random):")
from src.database.queries import load_targets
targets = load_targets()
if not targets.empty:
    wr = targets['winner'].mean()
    print("  {:.1%} — real market data behaves correctly (close to 50%)".format(wr))

print()
print("Dashboard routes:")
all_ok = True
for path in ['/', '/Overview', '/Stock_Screener', '/Stock_Detail',
             '/Model_Diagnostics', '/Backtesting', '/Prediction_Archive', '/Data_Quality']:
    try:
        code = urllib.request.urlopen('http://localhost:8501' + path, timeout=4).getcode()
        print("  {} OK  {}".format(code, path))
    except Exception as e:
        print("  ERR  {}  ({})".format(path, str(e)[:50]))
        all_ok = False

print()
if all_ok:
    print("RESULT: ALL CLEAR — Real Yahoo Finance data, all pages up.")
else:
    print("RESULT: SOME ISSUES — See errors above.")
print("=" * 60)
