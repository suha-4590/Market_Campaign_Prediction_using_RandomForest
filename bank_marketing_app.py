# """
# Bank Marketing Campaign Predictor
# Streamlit App — run with: streamlit run bank_marketing_app.py
# Place bank.csv in the same folder before running.
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )

# # ── Page config ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="Bank Campaign Predictor",
#     page_icon="🏦",
#     layout="wide",
# )

# # ── Custom CSS ───────────────────────────────────────────────
# st.markdown("""
# <style>
#     /* Main background */
#     .stApp { background-color: #f0f4f8; }

#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background-color: #1a2e44;
#     }
#     [data-testid="stSidebar"] * { color: #e8f0fe !important; }
#     [data-testid="stSidebar"] .stSelectbox label,
#     [data-testid="stSidebar"] .stSlider label,
#     [data-testid="stSidebar"] .stNumberInput label { color: #a0b4cc !important; font-size: 0.82rem; }

#     /* Header card */
#     .header-card {
#         background: linear-gradient(135deg, #1a2e44 0%, #2d5a8e 100%);
#         border-radius: 16px;
#         padding: 2rem 2.5rem;
#         color: white;
#         margin-bottom: 1.5rem;
#     }
#     .header-card h1 { color: white; margin: 0; font-size: 2rem; }
#     .header-card p  { color: #a8c8f0; margin: 0.3rem 0 0; font-size: 1rem; }

#     /* Metric cards */
#     .metric-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.2rem 1.5rem;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.07);
#         text-align: center;
#     }
#     .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #1a2e44; }
#     .metric-card .label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

#     /* Prediction banner */
#     .pred-yes {
#         background: linear-gradient(135deg, #065f46, #059669);
#         border-radius: 14px; padding: 1.6rem 2rem; color: white; text-align: center;
#     }
#     .pred-no {
#         background: linear-gradient(135deg, #7f1d1d, #dc2626);
#         border-radius: 14px; padding: 1.6rem 2rem; color: white; text-align: center;
#     }
#     .pred-yes h2, .pred-no h2 { color: white; margin: 0; font-size: 1.7rem; }
#     .pred-yes p,  .pred-no p  { color: rgba(255,255,255,0.8); margin: 0.4rem 0 0; }

#     /* Section headers */
#     .section-title {
#         font-size: 1.05rem; font-weight: 700;
#         color: #1a2e44; letter-spacing: 0.03em;
#         margin: 1.2rem 0 0.6rem; text-transform: uppercase;
#     }

#     /* Chart containers */
#     .chart-box {
#         background: white; border-radius: 12px;
#         padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
#     }

#     /* Step badges */
#     .step-badge {
#         display:inline-block;
#         background:#2d5a8e; color:white;
#         border-radius:50%; width:22px; height:22px;
#         text-align:center; line-height:22px;
#         font-size:0.75rem; font-weight:700;
#         margin-right:6px;
#     }

#     /* ── Tab bar: dark text so it shows on light background ── */
#     .stTabs [data-baseweb="tab-list"] {
#         background-color: #dbe6f0;
#         border-radius: 10px;
#         padding: 4px 6px;
#         gap: 4px;
#     }
#     .stTabs [data-baseweb="tab"] {
#         color: #1a2e44 !important;
#         font-weight: 600;
#         font-size: 0.95rem;
#         border-radius: 8px;
#         padding: 0.45rem 1.1rem;
#     }
#     .stTabs [aria-selected="true"] {
#         background-color: #1a2e44 !important;
#         color: white !important;
#     }
#     .stTabs [aria-selected="false"]:hover {
#         background-color: #b8cfe0 !important;
#         color: #1a2e44 !important;
#     }

#     /* ── Customer Summary rows: dark label + value text ── */
#     .summary-row {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         padding: 0.45rem 0.9rem;
#         border-radius: 8px;
#         margin-bottom: 0.3rem;
#         background: #eef3f8;
#     }
#     .summary-row .sum-key {
#         font-weight: 700;
#         color: #1a2e44;
#         font-size: 0.88rem;
#     }
#     .summary-row .sum-val {
#         color: #2d5a8e;
#         font-size: 0.88rem;
#         font-weight: 600;
#     }

#     /* ── Dataset Split metrics: force dark text ── */
#     [data-testid="stMetric"] {
#         background: white;
#         border-radius: 10px;
#         padding: 0.9rem 1.1rem;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.07);
#     }
#     [data-testid="stMetricLabel"] p {
#         color: #6b7280 !important;
#         font-size: 0.78rem !important;
#         text-transform: uppercase;
#         letter-spacing: 0.05em;
#     }
#     [data-testid="stMetricValue"] {
#         color: #1a2e44 !important;
#         font-size: 1.6rem !important;
#         font-weight: 700 !important;
#     }
# </style>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════
# # 1.  LOAD & CACHE DATA + MODEL
# # ══════════════════════════════════════════════════════════════
# @st.cache_data
# def load_and_train():
#     """Load data, encode, train model — cached so it only runs once."""
#     df = pd.read_csv("bank.csv")

#     X = df.drop(columns=["deposit"])
#     y = df["deposit"]

#     # Encode text columns
#     text_cols = X.select_dtypes(include="object").columns.tolist()
#     encoders = {}          # store one encoder per column (needed for prediction)
#     X_enc = X.copy()

#     for col in text_cols:
#         enc = LabelEncoder()
#         X_enc[col] = enc.fit_transform(X[col])
#         encoders[col] = enc

#     # Encode target
#     target_enc = LabelEncoder()
#     y_enc = target_enc.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
#     )

#     model = RandomForestClassifier(
#         n_estimators=300, min_samples_split=10, min_samples_leaf=2,
#         max_features=0.5, max_depth=20, bootstrap=True, random_state=42, n_jobs=-1
#     )
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     acc    = accuracy_score(y_test, y_pred)
#     cm     = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred,
#                                    target_names=["No", "Yes"], output_dict=True)

#     importances = pd.DataFrame({
#         "Feature":    X_enc.columns,
#         "Importance": model.feature_importances_,
#     }).sort_values("Importance", ascending=False).head(10)

#     return df, model, encoders, target_enc, X_enc.columns.tolist(), \
#            acc, cm, report, importances, X_train.shape[0], X_test.shape[0]


# # ══════════════════════════════════════════════════════════════
# # 2.  SIDEBAR — Prediction Inputs
# # ══════════════════════════════════════════════════════════════
# def sidebar_inputs():
#     st.sidebar.markdown("## 🔮 Predict a Customer")
#     st.sidebar.markdown("Fill in the customer's details below.")
#     st.sidebar.markdown("---")

#     st.sidebar.markdown("**👤 Personal Info**")
#     age      = st.sidebar.slider("Age", 18, 95, 35)
#     job      = st.sidebar.selectbox("Job", ['admin.','blue-collar','entrepreneur',
#                                             'housemaid','management','retired',
#                                             'self-employed','services','student',
#                                             'technician','unemployed','unknown'])
#     marital  = st.sidebar.selectbox("Marital Status", ['divorced','married','single'])
#     education= st.sidebar.selectbox("Education", ['primary','secondary','tertiary','unknown'])

#     st.sidebar.markdown("**💳 Financial Info**")
#     balance  = st.sidebar.number_input("Account Balance (€)", -6847, 81204, 1500)
#     default  = st.sidebar.selectbox("Has Credit Default?", ['no','yes'])
#     housing  = st.sidebar.selectbox("Has Housing Loan?",   ['no','yes'])
#     loan     = st.sidebar.selectbox("Has Personal Loan?",  ['no','yes'])

#     st.sidebar.markdown("**📞 Campaign Info**")
#     contact   = st.sidebar.selectbox("Contact Type",         ['cellular','telephone','unknown'])
#     month     = st.sidebar.selectbox("Last Contact Month",   ['jan','feb','mar','apr','may','jun',
#                                                                'jul','aug','sep','oct','nov','dec'])
#     day       = st.sidebar.slider("Day of Month", 1, 31, 15)
#     duration  = st.sidebar.number_input("Last Call Duration (secs)", 0, 3881, 300)
#     campaign  = st.sidebar.number_input("Contacts This Campaign", 1, 63, 2)
#     pdays     = st.sidebar.number_input("Days Since Last Contact (-1 = never)", -1, 854, -1)
#     previous  = st.sidebar.number_input("Previous Contacts", 0, 58, 0)
#     poutcome  = st.sidebar.selectbox("Previous Campaign Outcome",
#                                      ['failure','other','success','unknown'])

#     return {
#         "age": age, "job": job, "marital": marital, "education": education,
#         "balance": balance, "default": default, "housing": housing, "loan": loan,
#         "contact": contact, "day": day, "month": month, "duration": duration,
#         "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
#     }


# # ══════════════════════════════════════════════════════════════
# # 3.  PREDICT on sidebar input
# # ══════════════════════════════════════════════════════════════
# def predict_customer(inputs, model, encoders, target_enc, feature_cols):
#     row = pd.DataFrame([inputs])
#     row_enc = row.copy()
#     for col, enc in encoders.items():
#         if col in row_enc.columns:
#             # Handle unseen labels gracefully
#             known = set(enc.classes_)
#             row_enc[col] = row_enc[col].apply(
#                 lambda v: enc.transform([v])[0] if v in known else 0
#             )
#     row_enc = row_enc[feature_cols]
#     pred       = model.predict(row_enc)[0]
#     proba      = model.predict_proba(row_enc)[0]
#     label      = target_enc.inverse_transform([pred])[0]
#     confidence = proba[pred] * 100
#     return label, confidence, proba


# # ══════════════════════════════════════════════════════════════
# # 4.  MAIN APP
# # ══════════════════════════════════════════════════════════════
# def main():
#     # ── Load ──
#     with st.spinner("Training the model..."):
#         (df, model, encoders, target_enc, feature_cols,
#          acc, cm, report, importances,
#          n_train, n_test) = load_and_train()

#     # ── Sidebar ──
#     inputs = sidebar_inputs()
#     label, confidence, proba = predict_customer(
#         inputs, model, encoders, target_enc, feature_cols
#     )

#     # ── Header ──
#     st.markdown("""
#     <div class="header-card">
#         <h1>🏦 Bank Marketing Campaign Predictor</h1>
#         <p>Random Forest model trained on 11,162 customers · Predict who will subscribe to a term deposit</p>
#     </div>
#     """, unsafe_allow_html=True)

#     # ── Tabs ──
#     tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📂 Data Explorer"])

#     # ══════════════════════════════════════
#     # TAB 1 — PREDICTION
#     # ══════════════════════════════════════
#     with tab1:
#         col_pred, col_proba = st.columns([1, 1], gap="large")

#         with col_pred:
#             if label == "yes":
#                 st.markdown(f"""
#                 <div class="pred-yes">
#                     <h2>✅ Will Subscribe</h2>
#                     <p>Confidence: <strong>{confidence:.1f}%</strong></p>
#                 </div>""", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                 <div class="pred-no">
#                     <h2>❌ Will Not Subscribe</h2>
#                     <p>Confidence: <strong>{confidence:.1f}%</strong></p>
#                 </div>""", unsafe_allow_html=True)
 
#             st.markdown("<br>", unsafe_allow_html=True)
#             st.markdown('<p class="section-title">What does this mean?</p>', unsafe_allow_html=True)
#             if label == "yes":
#                 st.markdown("""
#                 <div style="background:#d1fae5; border-left:5px solid #059669;
#                             border-radius:8px; padding:1rem 1.2rem;">
#                     <span style="font-size:1.1rem;">✅</span>
#                     <span style="color:#065f46; font-weight:600; font-size:0.95rem;">
#                         Likely to Subscribe
#                     </span>
#                     <p style="color:#065f46; margin:0.4rem 0 0; font-size:0.88rem; line-height:1.5;">
#                         This customer is likely to subscribe to the term deposit.
#                         Consider prioritising them in your campaign outreach.
#                     </p>
#                 </div>""", unsafe_allow_html=True)
#             else:
#                 st.markdown("""
#                 <div style="background:#fef3c7; border-left:5px solid #d97706;
#                             border-radius:8px; padding:1rem 1.2rem;">
#                     <span style="font-size:1.1rem;">⚠️</span>
#                     <span style="color:#92400e; font-weight:600; font-size:0.95rem;">
#                         Unlikely to Subscribe
#                     </span>
#                     <p style="color:#92400e; margin:0.4rem 0 0; font-size:0.88rem; line-height:1.5;">
#                         This customer is unlikely to subscribe. You may want to adjust
#                         your approach or deprioritise them to save campaign resources.
#                     </p>
#                 </div>""", unsafe_allow_html=True)

#         with col_proba:
#             st.markdown('<p class="section-title">Probability Breakdown</p>', unsafe_allow_html=True)
#             fig, ax = plt.subplots(figsize=(5, 3))
#             bars = ax.barh(["Will NOT Subscribe", "Will Subscribe"],
#                            [proba[0]*100, proba[1]*100],
#                            color=["#dc2626", "#059669"], edgecolor="none", height=0.45)
#             ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
#             ax.set_xlim(0, 115)
#             ax.set_xlabel("Probability (%)")
#             ax.spines[["top","right","left"]].set_visible(False)
#             ax.tick_params(left=False)
#             plt.tight_layout()
#             st.pyplot(fig)
#             plt.close()

#             st.markdown('<p class="section-title">Customer Summary</p>', unsafe_allow_html=True)
#             summary_data = {
#                 "Age":               inputs["age"],
#                 "Job":               inputs["job"],
#                 "Balance":           f"€{inputs['balance']:,}",
#                 "Call Duration":     f"{inputs['duration']}s",
#                 "Campaign Contacts": inputs["campaign"],
#                 "Prev. Outcome":     inputs["poutcome"],
#             }
#             rows_html = "".join(
#                 f'<div class="summary-row">'
#                 f'<span class="sum-key">{k}</span>'
#                 f'<span class="sum-val">{v}</span>'
#                 f'</div>'
#                 for k, v in summary_data.items()
#             )
#             st.markdown(rows_html, unsafe_allow_html=True)

#     # ══════════════════════════════════════
#     # TAB 2 — MODEL PERFORMANCE
#     # ══════════════════════════════════════
#     with tab2:
#         # Metric cards row
#         m1, m2, m3, m4 = st.columns(4, gap="medium")
#         metrics = [
#             (f"{acc*100:.1f}%",   "Overall Accuracy"),
#             (f"{report['Yes']['precision']*100:.1f}%", "Yes Precision"),
#             (f"{report['Yes']['recall']*100:.1f}%",    "Yes Recall"),
#             (f"{report['Yes']['f1-score']*100:.1f}%",  "Yes F1-Score"),
#         ]
#         for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
#             col.markdown(f"""
#             <div class="metric-card">
#                 <div class="value">{val}</div>
#                 <div class="label">{lbl}</div>
#             </div>""", unsafe_allow_html=True)

#         st.markdown("<br>", unsafe_allow_html=True)

#         c_left, c_right = st.columns(2, gap="large")

#         # Confusion Matrix
#         with c_left:
#             st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
#             fig, ax = plt.subplots(figsize=(4.5, 3.8))
#             disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
#             disp.plot(ax=ax, colorbar=False, cmap="Blues")
#             ax.set_title("Predicted vs Actual", fontsize=11, pad=10)
#             plt.tight_layout()
#             st.pyplot(fig)
#             plt.close()
#             st.caption("Diagonal = correct predictions · Off-diagonal = errors")

#         # Feature Importance
#         with c_right:
#             st.markdown('<p class="section-title">Top 10 Features by Importance</p>',
#                         unsafe_allow_html=True)
#             fig, ax = plt.subplots(figsize=(5, 4))
#             colors = plt.cm.Blues(
#                 np.linspace(0.4, 0.85, len(importances))[::-1]
#             )
#             bars = ax.barh(importances["Feature"][::-1],
#                            importances["Importance"][::-1],
#                            color=colors, edgecolor="none")
#             ax.set_xlabel("Importance Score")
#             ax.spines[["top","right","left"]].set_visible(False)
#             ax.tick_params(left=False)
#             plt.tight_layout()
#             st.pyplot(fig)
#             plt.close()

#         # Class report table
#         st.markdown('<p class="section-title">Full Classification Report</p>',
#                     unsafe_allow_html=True)
#         report_df = pd.DataFrame(report).T.iloc[:2].round(3)
#         report_df.index.name = "Class"
#         st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

#         # Train/test info
#         st.markdown('<p class="section-title">Dataset Split</p>', unsafe_allow_html=True)
#         ts1, ts2, ts3 = st.columns(3)
#         ts1.metric("Total rows",  f"{len(df):,}")
#         ts2.metric("Train rows",  f"{n_train:,}  (80%)")
#         ts3.metric("Test rows",   f"{n_test:,}  (20%)")

#     # ══════════════════════════════════════
#     # TAB 3 — DATA EXPLORER
#     # ══════════════════════════════════════
#     with tab3:
#         st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
#         st.dataframe(df.head(50), use_container_width=True)

#         st.markdown('<p class="section-title">Column Statistics</p>', unsafe_allow_html=True)
#         st.dataframe(df.describe().round(2), use_container_width=True)

#         st.markdown('<p class="section-title">Explore a Column</p>', unsafe_allow_html=True)
#         chosen_col = st.selectbox("Choose any column to visualise", df.columns.tolist())

#         fig, ax = plt.subplots(figsize=(7, 3.5))
#         if df[chosen_col].dtype == "object":
#             vc = df[chosen_col].value_counts()
#             colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(vc))[::-1])
#             vc.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
#             ax.set_title(f"Distribution of '{chosen_col}'")
#             ax.set_xlabel("")
#             ax.set_ylabel("Count")
#             plt.xticks(rotation=30, ha="right")
#         else:
#             ax.hist(df[chosen_col].dropna(), bins=30,
#                     color="#2d5a8e", edgecolor="white", linewidth=0.5)
#             ax.set_title(f"Distribution of '{chosen_col}'")
#             ax.set_xlabel(chosen_col)
#             ax.set_ylabel("Count")
#         ax.spines[["top","right"]].set_visible(False)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()

#         st.markdown('<p class="section-title">Missing Values Check</p>',
#                     unsafe_allow_html=True)
#         missing = df.isnull().sum().reset_index()
#         missing.columns = ["Column", "Missing Count"]
#         missing["Status"] = missing["Missing Count"].apply(
#             lambda x: "✅ No missing" if x == 0 else f"⚠️ {x} missing"
#         )
#         st.dataframe(missing, use_container_width=True, hide_index=True)


# if __name__ == "__main__":
#     main()
"""
Bank Marketing Campaign Predictor
Streamlit App — run with: streamlit run bank_marketing_app.py
Place bank.csv in the same folder before running.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Campaign Predictor",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a2e44;
    }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #a0b4cc !important; font-size: 0.82rem; }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1a2e44 0%, #2d5a8e 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .header-card h1 { color: white; margin: 0; font-size: 2rem; }
    .header-card p  { color: #a8c8f0; margin: 0.3rem 0 0; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #1a2e44; }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

    /* Prediction banner */
    .pred-yes {
        background: linear-gradient(135deg, #065f46, #059669);
        border-radius: 14px; padding: 1.6rem 2rem; color: white; text-align: center;
    }
    .pred-no {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        border-radius: 14px; padding: 1.6rem 2rem; color: white; text-align: center;
    }
    .pred-yes h2, .pred-no h2 { color: white; margin: 0; font-size: 1.7rem; }
    .pred-yes p,  .pred-no p  { color: rgba(255,255,255,0.8); margin: 0.4rem 0 0; }

    /* Section headers */
    .section-title {
        font-size: 1.05rem; font-weight: 700;
        color: #1a2e44; letter-spacing: 0.03em;
        margin: 1.2rem 0 0.6rem; text-transform: uppercase;
    }

    /* Chart containers */
    .chart-box {
        background: white; border-radius: 12px;
        padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    /* Step badges */
    .step-badge {
        display:inline-block;
        background:#2d5a8e; color:white;
        border-radius:50%; width:22px; height:22px;
        text-align:center; line-height:22px;
        font-size:0.75rem; font-weight:700;
        margin-right:6px;
    }

    /* ── Tab bar: dark text so it shows on light background ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #dbe6f0;
        border-radius: 10px;
        padding: 4px 6px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1a2e44 !important;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px;
        padding: 0.45rem 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a2e44 !important;
        color: white !important;
    }
    .stTabs [aria-selected="false"]:hover {
        background-color: #b8cfe0 !important;
        color: #1a2e44 !important;
    }

    /* ── Customer Summary rows: dark label + value text ── */
    .summary-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.3rem;
        background: #eef3f8;
    }
    .summary-row .sum-key {
        font-weight: 700;
        color: #1a2e44;
        font-size: 0.88rem;
    }
    .summary-row .sum-val {
        color: #2d5a8e;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* ── Dataset Split metrics: force dark text ── */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    [data-testid="stMetricLabel"] p {
        color: #6b7280 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #1a2e44 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1.  LOAD & CACHE DATA + MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_train():
    """Load data, encode with OHE, train model — cached so it only runs once."""
    df = pd.read_csv("bank.csv")

    X = df.drop(columns=["deposit"])
    y = df["deposit"]

    # ── One-Hot Encode all text columns ──────────────────────
    # Creates binary (0/1) columns per category value.
    # drop_first=True avoids the dummy variable trap.
    text_cols = X.select_dtypes(include="object").columns.tolist()
    X_enc = pd.get_dummies(X, columns=text_cols, drop_first=True)

    # Store the exact column list so prediction rows can be aligned
    ohe_columns = X_enc.columns.tolist()

    # ── Encode target (binary: yes → 1, no → 0) ──────────────
    target_enc = LabelEncoder()
    y_enc = target_enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=300, min_samples_split=10, min_samples_leaf=2,
        max_features=0.5, max_depth=20, bootstrap=True, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["No", "Yes"], output_dict=True)

    importances = pd.DataFrame({
        "Feature":    X_enc.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)

    return (df, model, target_enc, ohe_columns, text_cols,
            acc, cm, report, importances, X_train.shape[0], X_test.shape[0])


# ══════════════════════════════════════════════════════════════
# 2.  SIDEBAR — Prediction Inputs
# ══════════════════════════════════════════════════════════════
def sidebar_inputs():
    st.sidebar.markdown("## 🔮 Predict a Customer")
    st.sidebar.markdown("Fill in the customer's details below.")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**👤 Personal Info**")
    age      = st.sidebar.slider("Age", 18, 95, 35)
    job      = st.sidebar.selectbox("Job", ['admin.','blue-collar','entrepreneur',
                                            'housemaid','management','retired',
                                            'self-employed','services','student',
                                            'technician','unemployed','unknown'])
    marital  = st.sidebar.selectbox("Marital Status", ['divorced','married','single'])
    education= st.sidebar.selectbox("Education", ['primary','secondary','tertiary','unknown'])

    st.sidebar.markdown("**💳 Financial Info**")
    balance  = st.sidebar.number_input("Account Balance (€)", -6847, 81204, 1500)
    default  = st.sidebar.selectbox("Has Credit Default?", ['no','yes'])
    housing  = st.sidebar.selectbox("Has Housing Loan?",   ['no','yes'])
    loan     = st.sidebar.selectbox("Has Personal Loan?",  ['no','yes'])

    st.sidebar.markdown("**📞 Campaign Info**")
    contact   = st.sidebar.selectbox("Contact Type",         ['cellular','telephone','unknown'])
    month     = st.sidebar.selectbox("Last Contact Month",   ['jan','feb','mar','apr','may','jun',
                                                               'jul','aug','sep','oct','nov','dec'])
    day       = st.sidebar.slider("Day of Month", 1, 31, 15)
    duration  = st.sidebar.number_input("Last Call Duration (secs)", 0, 3881, 300)
    campaign  = st.sidebar.number_input("Contacts This Campaign", 1, 63, 2)
    pdays     = st.sidebar.number_input("Days Since Last Contact (-1 = never)", -1, 854, -1)
    previous  = st.sidebar.number_input("Previous Contacts", 0, 58, 0)
    poutcome  = st.sidebar.selectbox("Previous Campaign Outcome",
                                     ['failure','other','success','unknown'])

    return {
        "age": age, "job": job, "marital": marital, "education": education,
        "balance": balance, "default": default, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
    }


# ══════════════════════════════════════════════════════════════
# 3.  PREDICT on sidebar input
# ══════════════════════════════════════════════════════════════
def predict_customer(inputs, model, target_enc, ohe_columns, text_cols):
    # Build a single-row DataFrame from the sidebar inputs
    row = pd.DataFrame([inputs])

    # Apply the same One-Hot Encoding used during training
    row_enc = pd.get_dummies(row, columns=text_cols, drop_first=True)

    # Align to training columns:
    #   - add any missing OHE columns (category not present in this row) as 0
    #   - drop any extra columns (shouldn't happen, but safe guard)
    row_enc = row_enc.reindex(columns=ohe_columns, fill_value=0)

    pred       = model.predict(row_enc)[0]
    proba      = model.predict_proba(row_enc)[0]
    label      = target_enc.inverse_transform([pred])[0]
    confidence = proba[pred] * 100
    return label, confidence, proba


# ══════════════════════════════════════════════════════════════
# 4.  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # ── Load ──
    with st.spinner("Training the model..."):
        (df, model, target_enc, ohe_columns, text_cols,
         acc, cm, report, importances,
         n_train, n_test) = load_and_train()

    # ── Sidebar ──
    inputs = sidebar_inputs()
    label, confidence, proba = predict_customer(
        inputs, model, target_enc, ohe_columns, text_cols
    )

    # ── Header ──
    st.markdown("""
    <div class="header-card">
        <h1>🏦 Bank Marketing Campaign Predictor</h1>
        <p>Random Forest model trained on 11,162 customers · Predict who will subscribe to a term deposit</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📂 Data Explorer"])

    # ══════════════════════════════════════
    # TAB 1 — PREDICTION
    # ══════════════════════════════════════
    with tab1:
        col_pred, col_proba = st.columns([1, 1], gap="large")

        with col_pred:
            if label == "yes":
                st.markdown(f"""
                <div class="pred-yes">
                    <h2>✅ Will Subscribe</h2>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-no">
                    <h2>❌ Will Not Subscribe</h2>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>""", unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">What does this mean?</p>', unsafe_allow_html=True)
            if label == "yes":
                st.markdown("""
                <div style="background:#d1fae5; border-left:5px solid #059669;
                            border-radius:8px; padding:1rem 1.2rem;">
                    <span style="font-size:1.1rem;">✅</span>
                    <span style="color:#065f46; font-weight:600; font-size:0.95rem;">
                        Likely to Subscribe
                    </span>
                    <p style="color:#065f46; margin:0.4rem 0 0; font-size:0.88rem; line-height:1.5;">
                        This customer is likely to subscribe to the term deposit.
                        Consider prioritising them in your campaign outreach.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#fef3c7; border-left:5px solid #d97706;
                            border-radius:8px; padding:1rem 1.2rem;">
                    <span style="font-size:1.1rem;">⚠️</span>
                    <span style="color:#92400e; font-weight:600; font-size:0.95rem;">
                        Unlikely to Subscribe
                    </span>
                    <p style="color:#92400e; margin:0.4rem 0 0; font-size:0.88rem; line-height:1.5;">
                        This customer is unlikely to subscribe. You may want to adjust
                        your approach or deprioritise them to save campaign resources.
                    </p>
                </div>""", unsafe_allow_html=True)

        with col_proba:
            st.markdown('<p class="section-title">Probability Breakdown</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(["Will NOT Subscribe", "Will Subscribe"],
                           [proba[0]*100, proba[1]*100],
                           color=["#dc2626", "#059669"], edgecolor="none", height=0.45)
            ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)")
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown('<p class="section-title">Customer Summary</p>', unsafe_allow_html=True)
            summary_data = {
                "Age":               inputs["age"],
                "Job":               inputs["job"],
                "Balance":           f"€{inputs['balance']:,}",
                "Call Duration":     f"{inputs['duration']}s",
                "Campaign Contacts": inputs["campaign"],
                "Prev. Outcome":     inputs["poutcome"],
            }
            rows_html = "".join(
                f'<div class="summary-row">'
                f'<span class="sum-key">{k}</span>'
                f'<span class="sum-val">{v}</span>'
                f'</div>'
                for k, v in summary_data.items()
            )
            st.markdown(rows_html, unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB 2 — MODEL PERFORMANCE
    # ══════════════════════════════════════
    with tab2:
        # Metric cards row
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        metrics = [
            (f"{acc*100:.1f}%",   "Overall Accuracy"),
            (f"{report['Yes']['precision']*100:.1f}%", "Yes Precision"),
            (f"{report['Yes']['recall']*100:.1f}%",    "Yes Recall"),
            (f"{report['Yes']['f1-score']*100:.1f}%",  "Yes F1-Score"),
        ]
        for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c_left, c_right = st.columns(2, gap="large")

        # Confusion Matrix
        with c_left:
            st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Predicted vs Actual", fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Diagonal = correct predictions · Off-diagonal = errors")

        # Feature Importance
        with c_right:
            st.markdown('<p class="section-title">Top 10 Features by Importance</p>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = plt.cm.Blues(
                np.linspace(0.4, 0.85, len(importances))[::-1]
            )
            bars = ax.barh(importances["Feature"][::-1],
                           importances["Importance"][::-1],
                           color=colors, edgecolor="none")
            ax.set_xlabel("Importance Score")
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Class report table
        st.markdown('<p class="section-title">Full Classification Report</p>',
                    unsafe_allow_html=True)
        report_df = pd.DataFrame(report).T.iloc[:2].round(3)
        report_df.index.name = "Class"
        st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

        # Train/test info
        st.markdown('<p class="section-title">Dataset Split</p>', unsafe_allow_html=True)
        ts1, ts2, ts3 = st.columns(3)
        ts1.metric("Total rows",  f"{len(df):,}")
        ts2.metric("Train rows",  f"{n_train:,}  (80%)")
        ts3.metric("Test rows",   f"{n_test:,}  (20%)")

    # ══════════════════════════════════════
    # TAB 3 — DATA EXPLORER
    # ══════════════════════════════════════
    with tab3:
        st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown('<p class="section-title">Column Statistics</p>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown('<p class="section-title">Explore a Column</p>', unsafe_allow_html=True)
        chosen_col = st.selectbox("Choose any column to visualise", df.columns.tolist())

        fig, ax = plt.subplots(figsize=(7, 3.5))
        if df[chosen_col].dtype == "object":
            vc = df[chosen_col].value_counts()
            colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(vc))[::-1])
            vc.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
            ax.set_title(f"Distribution of '{chosen_col}'")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right")
        else:
            ax.hist(df[chosen_col].dropna(), bins=30,
                    color="#2d5a8e", edgecolor="white", linewidth=0.5)
            ax.set_title(f"Distribution of '{chosen_col}'")
            ax.set_xlabel(chosen_col)
            ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<p class="section-title">Missing Values Check</p>',
                    unsafe_allow_html=True)
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Status"] = missing["Missing Count"].apply(
            lambda x: "✅ No missing" if x == 0 else f"⚠️ {x} missing"
        )
        st.dataframe(missing, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()