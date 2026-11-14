"""Shared helpers for the CLV notebook and the Streamlit app."""
import numpy as np
import pandas as pd


def predict_clv(models, X):
    """Two-part CLV: expected 90-day spend = P(buy) x E[spend | buy] (log-model with smearing correction)."""
    p_buy = models['clf'].predict_proba(X)[:, 1]
    log_pred = np.clip(models['reg'].predict(X), *models['log_range'])
    spend_if_buy = np.clip(np.expm1(log_pred) * models['smear'], 0, None)
    return pd.DataFrame({'p_buy': p_buy, 'spend_if_buy': spend_if_buy, 'expected_spend': p_buy * spend_if_buy}, index=X.index)
