import streamlit as st
from prediction_helper import predict

st.markdown("""
<style>
    /* Page layout */
    .block-container { padding-top: 2rem; max-width: 960px; }

    /* Page title */
    .page-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 0.3rem;
        line-height: 1.25;
        padding-top: 0.4rem;
        overflow: visible;
    }
    .page-title span {
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .page-subtitle {
        font-size: 0.75rem;
        color: #444;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 2.2rem;
        padding-left: 2px;
    }

    /* Section headers */
    .section-header {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #444;
        margin: 1.6rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1f1f1f;
    }

    /* Loan-to-income badge */
    .lti-wrapper {
        background: #111;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        height: 100%;
    }
    .lti-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.4rem;
    }
    .lti-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #a78bfa;
        line-height: 1;
    }
    .lti-sub {
        font-size: 0.72rem;
        color: #3a3a3a;
        margin-top: 0.3rem;
    }

    /* Submit button */
    div.stButton > button {
        background: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 1.4rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #222;
        border-color: #555;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">Vox <span>Finance</span></div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Credit Risk Modelling</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Applicant Profile</div>', unsafe_allow_html=True)
row1 = st.columns(3)
with row1[0]:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
with row1[1]:
    income = st.number_input("Annual Income", min_value=0, step=100000)
with row1[2]:
    loan_amount = st.number_input("Loan Amount", min_value=0, step=100000)

st.markdown('<div class="section-header">Loan Details</div>', unsafe_allow_html=True)
loan_to_income_ratio = loan_amount / income if income > 0 else 0
row2 = st.columns(3)
with row2[0]:
    st.markdown(
        f"""<div class="lti-wrapper">
            <div class="lti-label">Loan to Income Ratio</div>
            <div class="lti-value">{loan_to_income_ratio:.2f}</div>
            <div class="lti-sub">Computed automatically</div>
        </div>""",
        unsafe_allow_html=True
    )
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD per Delinquency', min_value=0, value=20)

st.markdown('<div class="section-header">Risk Indicators</div>', unsafe_allow_html=True)
row3 = st.columns(3)
with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization (%)', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

st.markdown('<div class="section-header">Classification</div>', unsafe_allow_html=True)
row4 = st.columns(3)
with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

if st.button("Calculate Risk"):
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    rating_colors = {
        'Poor':      {'bg': '#2d1b1b', 'border': '#c0392b', 'text': '#e74c3c'},
        'Average':   {'bg': '#2d2410', 'border': '#d68910', 'text': '#f39c12'},
        'Good':      {'bg': '#1b2d20', 'border': '#1e8449', 'text': '#2ecc71'},
        'Excellent': {'bg': '#1a2530', 'border': '#1a6fa8', 'text': '#3498db'},
        'Undefined': {'bg': '#1e1e1e', 'border': '#555555', 'text': '#aaaaaa'},
    }
    rc = rating_colors.get(rating, rating_colors['Undefined'])

    prob_color = '#e74c3c' if probability >= 0.5 else '#2ecc71'

    st.markdown(
        f"""
        <style>
            .results-wrapper {{
                display: flex;
                gap: 1.2rem;
                margin-top: 1.8rem;
            }}
            .metric-card {{
                flex: 1;
                padding: 1.6rem 1.4rem 1.4rem;
                border-radius: 12px;
                border: 1px solid #2a2a2a;
                background: #121212;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }}
            .metric-label {{
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #666;
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                line-height: 1;
            }}
            .metric-sub {{
                font-size: 0.78rem;
                color: #555;
                margin-top: 0.2rem;
            }}
            .divider {{
                width: 32px;
                height: 2px;
                border-radius: 2px;
                margin-bottom: 0.3rem;
            }}
            .rating-card {{
                background: {rc['bg']} !important;
                border-color: {rc['border']} !important;
            }}
        </style>

        <div class="results-wrapper">
            <div class="metric-card">
                <div class="metric-label">Default Probability</div>
                <div class="divider" style="background:{prob_color};"></div>
                <div class="metric-value" style="color:{prob_color};">{probability:.2%}</div>
                <div class="metric-sub">Likelihood of loan default</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Credit Score</div>
                <div class="divider" style="background:#8e44ad;"></div>
                <div class="metric-value" style="color:#c39bd3;">{credit_score}</div>
                <div class="metric-sub">Range: 300 — 900</div>
            </div>
            <div class="metric-card rating-card">
                <div class="metric-label">Risk Rating</div>
                <div class="divider" style="background:{rc['border']};"></div>
                <div class="metric-value" style="color:{rc['text']};">{rating}</div>
                <div class="metric-sub" style="color:{rc['border']};">Creditworthiness assessment</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
