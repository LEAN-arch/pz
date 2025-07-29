# ======================================================================================
# GRC COMMAND CENTER: COMPLIANCE OVERSIGHT & PROCESS EXCELLENCE
#
# A single-file Streamlit application designed for a CO&PE Head role.
#
# VERSION: Corrected for pandas dtype errors and deprecated parameters.
#
# This dashboard provides a real-time, risk-based view of the Pharmaceutical
# Quality System (PQS), integrating principles from key global regulations
# and standards including:
#   - US FDA 21 CFR Part 11 (Electronic Records)
#   - US FDA 21 CFR Part 820 (Quality System Regulation)
#   - EU GxP (Good Practices, including Annex 11 for Computerised Systems)
#   - ICH Q9 (Quality Risk Management)
#   - ICH Q10 (Pharmaceutical Quality System)
#   - ISO 13485 (Medical Devices - Quality Management Systems)
#   - ISO 14971 (Medical Devices - Application of Risk Management)
#
# To Run:
# 1. Save this code as 'grc_command_center_final.py'
# 2. Install dependencies: pip install streamlit pandas numpy plotly scikit-learn
# 3. Run from your terminal: streamlit run grc_command_center_final.py
#
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

# ======================================================================================
# SECTION 1: APP CONFIGURATION & STYLING
# ======================================================================================

st.set_page_config(
    page_title="GRC Command Center | CO&PE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, executive dashboard look and feel
st.markdown("""
<style>
    .main .block-container { padding: 1rem 3rem 3rem; }
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# SECTION 2: SME-ENHANCED DATA SIMULATION (WITH REGULATORY CONTEXT)
# ======================================================================================

@st.cache_data(ttl=600)
def generate_capa_data():
    """Simulates CAPA data with regulatory references and effectiveness checks, reflecting a real-world QMS."""
    np.random.seed(42)
    reg_refs = ['21 CFR 820.100', 'ISO 13485: 8.5.2', 'ICH Q10: 3.2.2', 'EU GxP Annex 11', 'Self-Identified']
    data = {
        'CAPA_ID': [f'CAPA-{i:04d}' for i in range(1, 151)],
        'Source': np.random.choice(['Internal Audit', 'EMA Inspection', 'FDA 483', 'Quality Event', 'Process Monitoring'], 150),
        'Regulatory_Reference': np.random.choice(reg_refs, 150, p=[0.2, 0.2, 0.1, 0.2, 0.3]),
        'Status': np.random.choice(['Open', 'In Progress', 'Pending Verification', 'Closed-Effective', 'Closed-Ineffective'], 150, p=[0.1, 0.2, 0.1, 0.55, 0.05]),
        'Creation_Date': [datetime.date(2022, 1, 1) + datetime.timedelta(days=int(d)) for d in np.random.randint(0, 700, 150)],
        'Risk_Level': np.random.choice(['Critical', 'Major', 'Minor'], 150, p=[0.05, 0.35, 0.6])
    }
    df = pd.DataFrame(data)

    df['Creation_Date'] = pd.to_datetime(df['Creation_Date'])
    
    df['Due_Date'] = df['Creation_Date'] + pd.to_timedelta(np.select([df['Risk_Level']=='Critical', df['Risk_Level']=='Major'], [30, 60], 90), unit='d')
    df['Days_Open'] = (pd.to_datetime('today') - df['Creation_Date']).dt.days
    df['Is_Overdue'] = (pd.to_datetime('today') > df['Due_Date']) & (~df['Status'].str.contains('Closed'))
    return df

@st.cache_data(ttl=600)
def generate_process_metrics_data():
    """Simulates process data with GxP-relevant sub-phases and error types for Value Stream Mapping and Pareto analysis."""
    np.random.seed(0)
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=24, freq='MS'))
    error_types = ['Incorrect Component Spec', 'Wrong Barcode', 'Typographical Error', 'Missed Regulatory Text', 'Formatting Issue']
    data = {
        'Month': dates,
        'Phase_Design_Control_Days': np.random.normal(5, 1, 24),
        'Phase_RA_Review_Days': np.random.normal(10, 2, 24),
        'Phase_QA_Approval_Days': np.random.normal(8, 1.5, 24),
        'eCTD_Submission_Success_Rate': np.random.uniform(0.97, 1.0, 24),
        'COPQ_USD': np.random.normal(50000, 10000, 24) * np.linspace(1, 0.6, 24),
        'Error_Type_for_COPQ': np.random.choice(error_types, 24, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    }
    df = pd.DataFrame(data)
    df['Total_Cycle_Time'] = df['Phase_Design_Control_Days'] + df['Phase_RA_Review_Days'] + df['Phase_QA_Approval_Days']
    return df

@st.cache_data(ttl=600)
def generate_team_data():
    """Simulates team data with GxP-specific competencies and performance metrics (RFT, Throughput)."""
    np.random.seed(55)
    team = ['Alex Chen', 'Brenda Smith', 'Carlos Ruiz', 'Diana Ivanova', 'Ethan Wong', 'Fiona Gallagher']
    data = {
        'Team_Member': team,
        'Tasks_Completed_90d': np.random.randint(20, 50, len(team)),
        'RFT_Rate': np.random.uniform(0.85, 0.99, len(team)),
        'Training_Overdue': np.random.choice([True, False], len(team), p=[0.1, 0.9]),
        # GxP Competencies (1-5 Scale)
        'ICH_Q9_Risk_Mgmt': np.random.randint(2, 6, len(team)),
        'ICH_Q10_PQS': np.random.randint(1, 6, len(team)),
        'CFR_21_Part_11_Compliance': np.random.randint(3, 6, len(team)),
        'ISO_13485_Auditing': np.random.randint(1, 5, len(team)),
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def generate_document_data():
    """Simulates QMS document control data to monitor compliance with periodic review cycles."""
    np.random.seed(101)
    doc_types = ['SOP', 'Work Instruction', 'Policy', 'Form']
    statuses = ['Active', 'Pending Review', 'Draft', 'Obsolete']
    data = {
        'Doc_ID': [f'QMS-{i:03d}' for i in range(1, 51)],
        'Document_Title': [f'{t} for Area {a}' for t, a in zip(np.random.choice(doc_types, 50), np.random.randint(1, 10, 50))],
        'Status': np.random.choice(statuses, 50, p=[0.8, 0.1, 0.05, 0.05]),
        'Last_Review_Date': [datetime.date.today() - datetime.timedelta(days=np.random.randint(30, 365*2)) for _ in range(50)],
        'Review_Cycle_Days': np.random.choice([365, 730], 50)
    }
    df = pd.DataFrame(data)
    
    df['Last_Review_Date'] = pd.to_datetime(df['Last_Review_Date'])
    
    df['Next_Review_Date'] = df['Last_Review_Date'] + pd.to_timedelta(df['Review_Cycle_Days'], unit='d')
    df['Days_Until_Review'] = (df['Next_Review_Date'] - pd.to_datetime('today')).dt.days
    return df

# ======================================================================================
# SECTION 3: PREDICTIVE MODEL
# ======================================================================================
@st.cache_resource
def get_predictive_model():
    """
    Returns a mock predictive model and its associated metadata.
    In a real application, this would load a pre-trained model artifact (e.g., from pickle or a model registry).
    The mock logic simulates risk based on key inputs for demonstration purposes.
    """
    class MockModel:
        def predict_proba(self, X):
            base_prob = 0.05
            if X['Complexity_High'].iloc[0] == 1: base_prob += 0.30
            if X['Is_CMO_Involved_True'].iloc[0] == 1: base_prob += 0.15
            if X['Submission_Type_Safety Update'].iloc[0] == 1: base_prob += 0.25
            if X['Post_Approval_Commitment_Yes'].iloc[0] == 1: base_prob += 0.10
            if X['Related_to_Field_Action_Yes'].iloc[0] == 1: base_prob += 0.40
            prob = min(0.98, base_prob + np.random.uniform(-0.05, 0.05))
            return np.array([[1-prob, prob]])

    model_cols = ['Complexity_High', 'Complexity_Low', 'Complexity_Medium', 'Is_CMO_Involved_True',
                  'Submission_Type_Safety Update', 'Post_Approval_Commitment_Yes', 'Related_to_Field_Action_Yes']
    feature_importance = pd.DataFrame({
        'feature': ['Related_to_Field_Action_Yes', 'Complexity_High', 'Submission_Type_Safety Update', 'Is_CMO_Involved_True', 'Post_Approval_Commitment_Yes'],
        'importance': [0.45, 0.30, 0.15, 0.07, 0.03]
    }).sort_values('importance', ascending=False)
    
    return MockModel(), model_cols, 0.88, feature_importance

# ======================================================================================
# SECTION 4: SME-ENHANCED PLOTTING FUNCTIONS
# ======================================================================================
def plot_capa_risk_heatmap(df):
    """Creates a heatmap to visualize the intersection of CAPA risk level and aging, a key QRM tool."""
    df_open = df[~df['Status'].str.contains('Closed')].copy()
    df_open['Age_Bucket'] = pd.cut(df_open['Days_Open'], bins=[0, 30, 60, 90, np.inf], labels=['0-30d', '31-60d', '61-90d', '>90d'])
    heatmap_data = pd.crosstab(df_open['Risk_Level'], df_open['Age_Bucket'])
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
        colorscale='Reds', colorbar=dict(title='Count of Open CAPAs')))
    fig.update_layout(
        title='<b>Open CAPA Risk & Timeliness Posture</b> (per 21 CFR 820.100)',
        xaxis_title='Aging Bucket (Days Open)', yaxis_title='CAPA Risk Level',
        yaxis_categoryorder='array', yaxis_categoryarray=['Minor', 'Major', 'Critical'])
    return fig

def plot_pareto_chart(df, category_col, value_col, title):
    """Generates a Pareto chart to identify the 'vital few' causes of the 'trivial many' problems (80/20 rule)."""
    pareto_df = df.groupby(category_col)[value_col].sum().reset_index()
    pareto_df = pareto_df.sort_values(by=value_col, ascending=False)
    pareto_df['Cumulative_Percentage'] = (pareto_df[value_col].cumsum() / pareto_df[value_col].sum()) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pareto_df[category_col], y=pareto_df[value_col], name='COPQ (USD)', marker_color='crimson'))
    fig.add_trace(go.Scatter(x=pareto_df[category_col], y=pareto_df['Cumulative_Percentage'], name='Cumulative %',
                             mode='lines+markers', yaxis='y2', line=dict(color='royalblue', width=2)))
    fig.update_layout(
        title=f'<b>{title}</b>', xaxis=dict(title='Error Type'),
        yaxis=dict(title='Cost (USD)', color='crimson'),
        yaxis2=dict(title='Cumulative Percentage', overlaying='y', side='right', range=[0, 105], showgrid=False, color='royalblue'),
        legend=dict(x=0.01, y=0.99), plot_bgcolor='white')
    return fig

def plot_talent_readiness_quadrant(df):
    """Creates a Gartner-style quadrant chart to assess team performance vs. GxP readiness."""
    avg_rft = df['RFT_Rate'].mean()
    avg_throughput = df['Tasks_Completed_90d'].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Tasks_Completed_90d'], y=df['RFT_Rate'], text=df['Team_Member'], mode='markers+text', textposition='top center',
        marker=dict(size=15, color=df['Training_Overdue'].map({True: 'red', False: 'green'}),
                    symbol=df['Training_Overdue'].map({True: 'x', False: 'circle'}),
                    line=dict(width=2, color='DarkSlateGrey'), showscale=False)))
    fig.add_hline(y=avg_rft, line_dash="dot", line_color="gray")
    fig.add_vline(x=avg_throughput, line_dash="dot", line_color="gray")
    fig.update_layout(
        title='<b>Talent Performance & GxP Role Readiness</b>',
        xaxis_title='Throughput (Tasks Completed in 90 Days)', yaxis_title='Right-First-Time (RFT) Rate',
        plot_bgcolor='#f9f9f9', yaxis_tickformat=".0%")
    fig.add_annotation(x=avg_throughput*1.2, y=avg_rft*1.01, text="<b>Top Performers</b>", font=dict(color="green"), showarrow=False)
    fig.add_annotation(x=avg_throughput*0.8, y=avg_rft*1.01, text="<b>Quality Focused</b>", font=dict(color="blue"), showarrow=False)
    fig.add_annotation(x=avg_throughput*0.8, y=avg_rft*0.99, text="<b>Needs Coaching</b>", font=dict(color="orange"), showarrow=False)
    fig.add_annotation(x=avg_throughput*1.2, y=avg_rft*0.99, text="<b>Throughput Focused</b>", font=dict(color="purple"), showarrow=False)
    return fig

def plot_risk_gauge(probability):
    """Displays a risk probability on a clear, color-coded gauge for immediate interpretation."""
    if probability > 0.6: color, level = "#d62728", "High Risk"
    elif probability > 0.3: color, level = "#ff7f0e", "Medium Risk"
    else: color, level = "#2ca02c", "Low Risk"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': level, 'font': {'size': 24, 'color': color}},
        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                 'bar': {'color': color}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                 'steps': [{'range': [0, 30], 'color': '#eafaf1'}, {'range': [30, 60], 'color': '#fef5e7'}],
                 'threshold': {'line': {'color': "#d62728", 'width': 4}, 'thickness': 0.75, 'value': 60}}))
    fig.update_layout(height=250, margin=dict(t=50, b=40, l=40, r=40))
    return fig

# ======================================================================================
# SECTION 5: MAIN APPLICATION LAYOUT & LOGIC
# ======================================================================================

# --- HEADER ---
st.title("🛡️ GRC Command Center: Compliance Oversight & Process Excellence")
st.markdown("##### A real-time, risk-based dashboard for the Global & International Labeling and Artwork (GILA) Organization")

# --- LOAD ALL DATA UPFRONT ---
capa_df = generate_capa_data()
process_df = generate_process_metrics_data()
team_df = generate_team_data()
doc_df = generate_document_data()
model, model_cols, model_accuracy, feature_importance = get_predictive_model()

# --- EXECUTIVE GRC METRICS ---
st.markdown("### Executive GRC & QMS Health Dashboard")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# KPI 1: QMS Health Index
overdue_major_critical = capa_df[capa_df['Is_Overdue'] & (capa_df['Risk_Level'].isin(['Major', 'Critical']))].shape[0]
ineffective_capas = capa_df[capa_df['Status'] == 'Closed-Ineffective'].shape[0]
qms_health = max(0, 100 - (overdue_major_critical * 10) - (ineffective_capas * 5))
kpi_col1.metric("QMS Health Index", f"{qms_health:.0f}%", f"{ineffective_capas} Ineffective CAPAs", "inverse", help="ICH Q10 Metric: Reflects effectiveness of the corrective action system. Target > 90%. Penalized by overdue major/critical and ineffective CAPAs.")

# KPI 2: Critical GxP Risk Exposure
kpi_col2.metric("Critical GxP Risk Exposure", overdue_major_critical, "Overdue Major/Critical CAPAs", "inverse", help="ICH Q9 Metric: Number of open, overdue CAPAs with high potential impact on product quality or patient safety. Target is 0.")

# KPI 3: Labeling Velocity (Right-First-Time)
avg_rft = team_df['RFT_Rate'].mean() * 100
avg_cycle_time = process_df['Total_Cycle_Time'].mean()
kpi_col3.metric("Labeling Velocity (RFT)", f"{avg_rft:.1f}%", f"{avg_cycle_time:.1f} Day Avg Cycle", "normal", help="Combines speed and quality. High RFT rate reduces rework and shortens timelines. Target > 95%.")

# KPI 4: GRC Readiness Score
docs_pending_review = doc_df[doc_df['Days_Until_Review'] < 0].shape[0]
training_non_compliant = team_df[team_df['Training_Overdue']].shape[0]
grc_readiness = max(0, 100 - (docs_pending_review * 5) - (training_non_compliant * 5))
kpi_col4.metric("GRC Readiness Score", f"{grc_readiness:.0f}%", f"{docs_pending_review} Docs Overdue", "inverse", help="ISO 13485 / 21 CFR 820 Metric: Composite score based on document control and training record compliance. Target > 95%.")
st.markdown("---")

# --- TABS FOR DETAILED ANALYSIS ---
tab1, tab2, tab3, tab4 = st.tabs(["**🚨 GRC & Audit Readiness**", "**⚙️ PEx & Digital Transformation**", "**👥 Org. Capability (ICH Q10)**", "**🔮 Predictive Compliance**"])

# ============================ TAB 1: GRC & AUDIT READINESS ============================
with tab1:
    st.header("GRC & Audit Readiness Command Center")
    st.markdown("Monitoring key elements of the Pharmaceutical Quality System (PQS) per ICH Q10.")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Document Control Hub")
        st.markdown("_(per ISO 13485:4.2.4 / 21 CFR 820.40)_")
        
        st.error(f"**Action Required: {docs_pending_review} documents are past their periodic review date.**")
        overdue_docs = doc_df[doc_df['Days_Until_Review'] < 0]
        st.dataframe(overdue_docs[['Doc_ID', 'Document_Title', 'Next_Review_Date']], use_container_width=True, height=200)

        st.warning(f"**Upcoming Reviews: {doc_df[(doc_df['Days_Until_Review'] >= 0) & (doc_df['Days_Until_Review'] < 30)].shape[0]} documents need review within 30 days.**")
        upcoming_docs = doc_df[(doc_df['Days_Until_Review'] >= 0) & (doc_df['Days_Until_Review'] < 30)]
        st.dataframe(upcoming_docs[['Doc_ID', 'Days_Until_Review']], use_container_width=True, height=200)

    with col2:
        st.subheader("CAPA Health & Effectiveness")
        st.markdown("_(per ICH Q9 / 21 CFR 820.100)_")
        st.info("**Actionable Insight:** This heatmap shows the intersection of risk and age. The top-right quadrant represents the highest regulatory risk and requires immediate escalation and resource allocation.")
        st.plotly_chart(plot_capa_risk_heatmap(capa_df), use_container_width=True)
        st.dataframe(capa_df[capa_df['Is_Overdue']][['CAPA_ID', 'Risk_Level', 'Regulatory_Reference', 'Due_Date']].sort_values('Risk_Level'), use_container_width=True)

# ============================ TAB 2: PEx & Digital Transformation ============================
with tab2:
    st.header("Process Excellence & Digital Transformation (PEx/DT)")
    st.markdown("Using data to drive continuous improvement and efficiency in line with Six Sigma and Lean principles.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Process Cycle Time Breakdown")
        st.markdown("_(Value Stream Mapping Approach)_")
        st.info("**Actionable Insight:** Identify bottlenecks in the labeling lifecycle. A consistently long `RA_Review` phase may indicate unclear requirements or resource constraints in Regulatory Affairs.")
        cycle_time_df = process_df[['Month', 'Phase_Design_Control_Days', 'Phase_RA_Review_Days', 'Phase_QA_Approval_Days']].set_index('Month')
        fig = px.area(cycle_time_df, title='<b>Labeling Cycle Time by Phase</b>', labels={'value': 'Days', 'variable': 'Phase'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("COPQ Pareto Analysis")
        st.markdown("_(Six Sigma: DMAIC - Analyze Phase)_")
        st.info("**Actionable Insight:** Focus improvement efforts on the 'vital few' error types that cause 80% of the cost. Targeting 'Incorrect Component Spec' would yield the highest ROI for CI projects.")
        st.plotly_chart(plot_pareto_chart(process_df, 'Error_Type_for_COPQ', 'COPQ_USD', "Pareto Analysis of Cost of Poor Quality (COPQ)"), use_container_width=True)
        
# ============================ TAB 3: Org. Capability (ICH Q10) ============================
with tab3:
    st.header("Organizational Capability & GxP Readiness")
    st.markdown("Ensuring the team has the appropriate resources, skills, and training to maintain the PQS as per ICH Q10.")

    st.subheader("Talent Performance & GxP Role Readiness")
    st.info("**Actionable Insight:** Green circles are GxP-ready; Red 'X's have overdue training and are an audit risk. Use the quadrants to guide development: coach 'Needs Coaching' on RFT, and challenge 'Quality Focused' with more tasks.")
    st.plotly_chart(plot_talent_readiness_quadrant(team_df), use_container_width=True)
    
    st.subheader("GxP Competency Matrix")
    st.info("**Actionable Insight:** This matrix identifies critical skill gaps against auditable GxP competencies. Prioritize training in 'ICH Q9 Risk Mgmt' and 'ISO 13485 Auditing' to strengthen our quality culture.")
    st.dataframe(team_df[['Team_Member', 'ICH_Q9_Risk_Mgmt', 'ICH_Q10_PQS', 'CFR_21_Part_11_Compliance', 'ISO_13485_Auditing']].set_index('Team_Member'), use_container_width=True)


# ============================ TAB 4: Predictive Compliance ============================
with tab4:
    st.header("🔮 Predictive Compliance & Risk Forecasting")
    st.markdown("Using AI to proactively identify risks before they become compliance issues, supporting a state of continuous control.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Risk Prediction Simulator")
        complexity = st.selectbox("Artwork Complexity", options=['Low', 'Medium', 'High'])
        sub_type = st.selectbox("Submission Type", options=['New', 'Revision', 'Safety Update'])
        cmo_involved = st.checkbox("Is a Contract Manufacturing Org (CMO) involved?")
        pac = st.checkbox("Is this a Post-Approval Commitment (PAC)?")
        field_action = st.checkbox("Is this related to a previous Field Action/Recall?")

        if st.button("Forecast Risk Profile", type="primary"):
            input_dict = {
                'Complexity_High': 1 if complexity == 'High' else 0, 'Complexity_Medium': 1 if complexity == 'Medium' else 0, 'Complexity_Low': 1 if complexity == 'Low' else 0,
                'Is_CMO_Involved_True': int(cmo_involved), 'Submission_Type_Safety Update': 1 if sub_type == 'Safety Update' else 0,
                'Post_Approval_Commitment_Yes': int(pac), 'Related_to_Field_Action_Yes': int(field_action)
            }
            input_df = pd.DataFrame([input_dict], columns=model_cols).fillna(0)
            prediction_proba = model.predict_proba(input_df)[0][1]

            st.plotly_chart(plot_risk_gauge(prediction_proba), use_container_width=True)
            if prediction_proba > 0.6:
                st.error("**High Risk Profile Detected:**\n\n**Action (per ICH Q9):** Escalate to Quality Review Board. Initiate formal risk assessment per ISO 14971. Document rationale for proceeding in QMS. High level of verification and validation required.")
            elif prediction_proba > 0.3:
                st.warning("**Medium Risk Profile Detected:**\n\n**Action:** Assign a senior team member. Add to weekly 'watch list'. Ensure risk mitigation strategies are documented in the project plan. Enhanced verification activities recommended.")
            else:
                st.success("**Low Risk Profile Detected:**\n\n**Action:** Standard process is sufficient. Standard verification activities apply. No additional escalation required.")

    with col2:
        st.subheader("Predictive Key Risk Drivers")
        st.info("**Actionable Insight:** These factors are statistically the most likely to cause delays and compliance issues. This data should be a direct input into our Quality Risk Management (QRM) program to create proactive, systemic controls.")
        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h', title='<b>Top Factors Driving Delay & Compliance Risk</b>')
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='white', xaxis_title='Relative Importance')
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown(f"**Model Confidence:** The underlying model predicts delays with an accuracy of **{model_accuracy:.1%}** on validation data.")

# --- SIDEBAR & FOOTER ---
st.sidebar.markdown("---")
# --- FIX APPLIED HERE ---
# Changed 'use_column_width' to the modern 'use_container_width' parameter.
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Pfizer_logo.svg/2560px-Pfizer_logo.svg.png", use_container_width=True)
st.sidebar.markdown("### About this Dashboard")
st.sidebar.info(
    "This **GRC Command Center** is a prototype for the CO&PE Head, integrating data from QMS, ERP, and project systems. It is designed to facilitate proactive governance, risk management, and compliance with key global regulations including **21 CFR Part 11/820, ICH Q9/Q10, and ISO 13485**."
)
st.sidebar.markdown("### Regulatory Context")
st.sidebar.markdown("""
- **GRC:** Governance, Risk, & Compliance
- **PQS:** Pharmaceutical Quality System (ICH Q10)
- **QRM:** Quality Risk Management (ICH Q9)
- **CAPA:** Corrective and Preventive Action
""")
