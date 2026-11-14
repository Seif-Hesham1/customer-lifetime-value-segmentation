Online retailers treat every customer the same — and usually find out too late who stopped buying.

In my NextGen Institute ML internship, our team built a tool that groups ~5,300 real retail customers by behaviour and predicts how much each will spend in the next 90 days.

• Cleaned ~1.07M invoice lines and built features with a strict time cut-off (no data leakage)
• K-Means segmentation (k=4) + a two-part model (P(buy again) × spend if buying), tuned with cross-validation
• Targeting the top 20% by predicted spend captures 75% of next-90-day revenue (random = 20%)
• Biggest lesson: a single regression on spend looked better on MAE but predicted only 44% of the real total revenue — the two-part model fixed that

Try the live app: [APP LINK]
Code: [GITHUB LINK]
Data: Online Retail II (UCI)

#MachineLearning #DataScience #Marketing #Streamlit #NextGenInstitute

(Tag your teammates + NextGen Institute. Add a 30–60 s screen recording of the app.)
