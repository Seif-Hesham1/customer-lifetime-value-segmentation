import json, sys
sys.path.insert(0, '.')
from deck import build
m = json.load(open('artifacts/metrics.json'))
t = m['test']; two = t['Two-part (Gradient Boosting × Ridge)']; one = t['Single GB on log-spend']; base = t['Baseline (mean £)']
F = 'figures/'
slides = [
 dict(kind='title', title='Customer Lifetime Value & Segmentation Engine', bullets=['Who to reward, who to win back, and where to spend the marketing budget', 'Team: [add names]']),
 dict(title='The problem', bullets=[
   'An online gift retailer treats every customer the same',
   f'The top 20% of customers generate {m["top20_hist_share"]:.0%} of historical revenue',
   f'Only {m["buy_again_rate"]:.0%} of customers buy again in the next 90 days',
   'Decision: who to reward, who to win back, where to spend the budget',
   'Goal: (1) group customers by behaviour, (2) predict next-90-day spend',
   'Metric: MAE in £ (explainable) + R²; ranking quality via top-20% revenue capture']),
 dict(title='Data: Online Retail II (UCI)', bullets=[
   f'{m["rows_raw"]:,} invoice lines, Dec 2009 – Dec 2011, one UK online retailer',
   f'After cleaning: {m["rows_clean"]:,} lines (returns, guest checkouts, postage/fees, duplicates removed)',
   f'{m["n_customers"]:,} customers with purchase history before the cut-off ({m["cutoff"]})',
   'No leakage: features use only data BEFORE the cut-off; target = spend in the 90 days AFTER',
   f'Customer split: {m["n_train"]:,} train / {m["n_test"]:,} test, fitted on train only']),
 dict(title='Approach: two models, one product', bullets=[
   'Unsupervised: RFM (Recency, Frequency, Monetary) → log + scale → K-Means (k=4)',
   'Supervised: expected spend = P(buy again) × spend if buying',
   'Why two parts? 56% of customers spend exactly £0; one model on log-spend recovered only 44% of true total revenue',
   'All preprocessing inside scikit-learn Pipelines; tuning with 5-fold CV on train only',
   'Models compared: Logistic/Ridge, Random Forest, Gradient Boosting (+ baselines)']),
 dict(title='EDA: revenue is seasonal, UK-dominated and highly concentrated', image=F + 'eda_bivariate.png', layout='stack', size=17, bullets=[
   'Revenue peaks every Q4 → our 90-day target window is peak season (a limitation)',
   'Past spend and future spend are related but noisy — many past big spenders go quiet',
   'The UK is the dominant market; country adds little beyond a UK flag']),
 dict(title='EDA: frequency drives repeat purchase', image=F + 'eda_multivariate.png', layout='stack', size=17, bullets=[
   'Share who buy again rises from 28% (bottom 50% by order count) to 91% (top 5%)',
   'Recency is strongly negatively related to future spend; recent-spend features overlap (use regularised/tree models)',
   'Money/count features are right-skewed → log transform']),
 dict(title='Segmentation: choosing k and comparing methods', image=F + 'segment_choose_k.png', bullets=[
   'Elbow + silhouette checked for k = 2…8',
   'k = 4 chosen: reasonable silhouette and four actionable groups',
   f'K-Means silhouette {m["silhouette_kmeans"]:.2f} vs Ward hierarchical 0.33 on a sample',
   'K-Means kept: it can score new customers (hierarchical cannot)']),
 dict(title='Four segments, four actions', image=F + 'segments.png', layout='stack', size=16, bullets=[
   'Champions: 17% of customers → 63% of next-90-day revenue (median 13 orders, last order 20 days ago) → reward & retain',
   'At Risk: 30% of customers, median £1,440 historic spend, silent 166 days (84% repeat for Champions vs 46% here) → win-back offer',
   'Promising (recent, 2 orders) → second-purchase nudge | Lost (1 order, 327 days) → low-cost email only']),
 dict(title='Modelling & tuning (cross-validation on train only)', bullets=[
   'Part A – will they buy? CV ROC-AUC: Logistic 0.796 · Random Forest 0.799 · Gradient Boosting 0.799 → Gradient Boosting',
   'Part B – how much (buyers only, log-£)? CV MAE: baseline 0.82 · Ridge 0.575 · Random Forest 0.580 · Gradient Boosting 0.577 → Ridge',
   'Winners chosen by CV score only; test set never used for selection',
   'Predictions clipped to the training range and calibrated so totals are unbiased',
   'RandomizedSearchCV for every algorithm']),
 dict(title='Results on the untouched test set', image=F + 'evaluation.png', layout='stack', size=17, bullets=[
   f'Will they buy: test ROC-AUC {m["test_roc_auc_buy"]:.2f} | Spend: MAE £{two["MAE £"]:.0f} vs £{base["MAE £"]:.0f} for the mean baseline, R² {two["R² (£)"]:.2f}',
   f'Targeting the top 20% by predicted spend captures {m["top20_revenue_capture"]:.0%} of next-90-day revenue (random = 20%)',
   f'Honest trade-off: a single log-model has lower MAE (£{one["MAE £"]:.0f}) but predicts only {one["Σ pred / Σ actual"]:.0%} of total revenue — two-part model is calibrated']),
 dict(title='Recommendation', bullets=[
   'Protect Champions with service & perks, not blanket discounts (63% of future revenue)',
   'Win back the At-Risk group: 1,582 customers who spent £3.2M historically but went quiet',
   'Nurture Promising customers right after their first/second purchase',
   'Keep Lost customers on low-cost email only (20% repeat rate)',
   'Use the predicted-spend score to prioritise outreach inside each segment',
   'Limits: one cut-off, peak-season window, one UK wholesaler, revenue ≠ profit']),
 dict(title='Live app & next steps', bullets=[
   'Streamlit app: segment dashboard · customer-ID lookup (segment, predicted 90-day spend, action) · manual scoring for new customers',
   'Handles invalid / unknown IDs and inconsistent inputs without crashing',
   'Next: more cut-off dates (rolling backtest), margin instead of revenue, A/B test the win-back campaign',
   'Code, notebook & app link: see README']),
]
build('presentation.pptx', 'Customer Lifetime Value & Segmentation', slides)
print('ok', len(slides))
