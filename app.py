import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from clv_utils import predict_clv

st.set_page_config(page_title="Customer Segments & 90-day Value", page_icon="🛍️", layout="wide")


@st.cache_resource
def load():
    models = joblib.load("artifacts/clv_model.joblib")
    seg = joblib.load("artifacts/segmenter.joblib")
    cust = pd.read_csv("artifacts/customers.csv").set_index("customer_id")
    prof = pd.read_csv("artifacts/segment_profile.csv").set_index("segment")
    rev = pd.read_csv("artifacts/segment_revenue.csv").set_index("segment")
    metrics = json.load(open("artifacts/metrics.json"))
    return models, seg, cust, prof, rev, metrics


models, seg, cust, prof, rev, metrics = load()
NAMES, ACTIONS, FEATURES = seg["names"], seg["actions"], seg["features"]
COLORS = {"Champions": "#16a34a", "Promising": "#2563eb", "At Risk": "#f59e0b", "Lost / Inactive": "#dc2626"}

st.title("🛍️ Customer Segments & 90-Day Value")
st.caption(
    f"Online Retail II (UCI) · history up to {seg['cutoff']} · {metrics['n_customers']:,} customers · "
    f"ranking quality: top 20% of customers = {metrics['top20_revenue_capture']:.0%} of next-90-day revenue"
)

tab1, tab2, tab3 = st.tabs(["📊 Segment dashboard", "🔎 Customer lookup", "🧮 New customer (manual)"])

with tab1:
    cols = st.columns(4)
    for c, n in zip(cols, NAMES):
        r = prof.loc[n]
        c.markdown(f"<div style='border-left:6px solid {COLORS[n]};padding:4px 12px'><b>{n}</b><br>"
                   f"{int(r['customers']):,} customers ({r['share_of_customers_%']:.0f}%)<br>"
                   f"{r['share_of_future_revenue_%']:.0f}% of next-90-day revenue</div>", unsafe_allow_html=True)
    st.markdown("")
    left, right = st.columns(2)
    with left:
        st.subheader("Typical customer in each segment")
        st.dataframe(prof.loc[NAMES, ["median_recency_days", "median_orders", "median_spend", "repeat_rate_next_90d"]]
                     .rename(columns={"median_recency_days": "Days since last order", "median_orders": "Orders",
                                      "median_spend": "Total spend £", "repeat_rate_next_90d": "Buy again in 90d"}),
                     use_container_width=True)
    with right:
        st.subheader("Expected revenue next 90 days (£)")
        import altair as alt
        chart_df = rev.loc[NAMES, ["predicted_90d_revenue"]].reset_index().rename(columns={"predicted_90d_revenue": "Expected £"})
        st.altair_chart(alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("segment:N", sort=NAMES, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Expected £:Q", title="£"),
            color=alt.Color("segment:N", scale=alt.Scale(domain=NAMES, range=[COLORS[n] for n in NAMES]), legend=None),
            tooltip=["segment", alt.Tooltip("Expected £:Q", format=",.0f")]), use_container_width=True)
    st.subheader("Recommended action")
    for n in NAMES:
        st.markdown(f"**:{ {'Champions':'green','Promising':'blue','At Risk':'orange','Lost / Inactive':'red'}[n] }[{n}]** — {ACTIONS[n]}")

with tab2:
    st.write("Enter a customer ID to see their segment, predicted 90-day spend and recommended action.")
    st.caption(f"Example IDs: {', '.join(str(i) for i in cust.sample(4, random_state=1).index)}")
    raw_id = st.text_input("Customer ID", placeholder="e.g. 12347")
    if raw_id.strip():
        try:
            cid = int(float(raw_id.strip()))
        except ValueError:
            st.error("Customer ID must be a number.")
            cid = None
        if cid is not None:
            if cid not in cust.index:
                st.warning("This ID was not found among customers with purchases before the cut-off date. "
                           "Use the **New customer** tab to score someone manually.")
            else:
                r = cust.loc[cid]
                st.markdown(f"### Customer {cid} — :{ {'Champions':'green','Promising':'blue','At Risk':'orange','Lost / Inactive':'red'}[r.segment] }[{r.segment}]")
                a, b, c = st.columns(3)
                a.metric("Predicted spend, next 90 days", f"£{r.pred_spend_90d:,.0f}")
                b.metric("Chance of buying again", f"{r.p_buy:.0%}")
                c.metric("Past orders / total spend", f"{int(r.frequency)} / £{r.monetary:,.0f}")
                st.info(f"**Recommended action:** {ACTIONS[r.segment]}")
                st.caption(f"Last order {int(r.recency_days)} days before {seg['cutoff']} · customer for {int(r.tenure_days)} days · "
                           f"{int(r.n_unique_products)} different products")

with tab3:
    st.write("Describe a customer's purchase history (as of today) to score them. Defaults are typical values.")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Days since last order", 0, 800, 60)
    freq = c2.number_input("Number of orders", 1, 400, 3)
    monetary = c3.number_input("Total spend (£)", 1.0, 1_000_000.0, 900.0, step=50.0)
    c4, c5, c6 = st.columns(3)
    tenure = c4.number_input("Days since first order", 0, 800, 300)
    products = c5.number_input("Different products bought", 1, 3000, 30)
    uk = c6.selectbox("Country", ["United Kingdom", "Other"]) == "United Kingdom"
    if tenure < recency:
        st.error("Days since first order cannot be smaller than days since last order.")
    else:
        aov = monetary / freq
        row = pd.DataFrame([{
            "recency_days": recency, "frequency": freq, "monetary": monetary, "avg_order_value": aov,
            "n_unique_products": products, "total_items": max(products, monetary / 3), "tenure_days": tenure,
            "avg_days_between_orders": (tenure - recency) / (freq - 1) if freq > 1 else np.nan,
            "spend_last_90d": monetary if recency < 90 and tenure < 90 else (aov if recency < 90 else 0.0),
            "spend_last_180d": monetary if recency < 180 and tenure < 180 else (aov if recency < 180 else 0.0),
            "orders_last_90d": 1.0 if recency < 90 else 0.0, "n_cancellations": 0.0, "is_uk": int(uk)}])[FEATURES]
        label = seg["name_of"][int(seg["kmeans"].predict(seg["rfm_pipe"].transform(row[seg["rfm"]]))[0])]
        out = predict_clv(models, row).iloc[0]
        a, b, c = st.columns(3)
        a.metric("Segment", label)
        b.metric("Predicted spend, next 90 days", f"£{out.expected_spend:,.0f}")
        c.metric("Chance of buying again", f"{out.p_buy:.0%}")
        st.info(f"**Recommended action:** {ACTIONS[label]}")
        st.caption("Rough estimate: spend in the last 90/180 days and total items are approximated from the inputs above.")

with st.expander("About this model & limitations"):
    st.markdown(
        f"* Segments: K-Means (k=4) on log-scaled Recency/Frequency/Monetary.\n"
        f"* Value: two-part model — P(buy) ({metrics['part_a']}) × spend if buying ({metrics['part_b']}).\n"
        f"* Trained on one UK gift wholesaler; the prediction window (Sep–Dec 2011) is peak season, so use £ values mainly to **rank** customers.\n"
        f"* Data: [Online Retail II, UCI](https://archive.ics.uci.edu/dataset/502).")
