import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling the elements
st.markdown("""
    <style>
    /* Main Section Styling */
    .main > div {
        padding-top: 2rem;
    }

    /* General Text Colors */
    h1, h2, h3 {
        color: white !important;
    }

    /* Button Styles */
    .stButton>button {
        background-color: #dc2626; /* Red background for button */
        color: white;
        font-weight: 500;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #991b1b; /* Darker red on hover */
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4); /* Red glow effect */
    }

    /* Input and Select Box Styling */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div {
        background-color: #2d2d2d; /* Dark background for inputs */
        border-radius: 5px;
        color: white; /* White text inside inputs */
        border: 1px solid #444444; /* Border color similar to input background */
    }

    /* Focus Styling */
    .stTextInput > div > div > input:focus, 
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #dc2626; /* Red border when focused */
        outline: none;
    }

    /* Sidebar Background */
    div[data-testid="stSidebar"] {
        background-color: #1f2937; /* Dark background for sidebar */
    }

    /* Data Table Styles */
    .stDataFrame table {
        background-color: #27272a; /* Dark table background */
        color: #f8fafc; /* Light text for readability */
    }

    .stDataFrame table th {
        background-color: #dc2626; /* Red background for table header */
        color: white; /* White text for header */
    }

    .stDataFrame table td {
        color: #f8fafc; /* Light text color for table data */
    }

    /* Dropdown and MultiSelect Boxes */
    .stSelectbox > div > div {
        background-color: #2d2d2d; /* Dark background for dropdowns */
        color: white;
    }

    .stSelectbox > div > div:hover {
        background-color: #444444; /* Hover effect for dropdown */
    }

    /* Metrics Styling */
    .stMetric > div {
        color: #dc2626 !important; /* Red for metrics */
    }

    /* Other form inputs styles */
    .stTextInput input {
        background-color: #2d2d2d;
        color: white;
    }

    .stCheckbox > div > div, 
    .stRadio > div > div {
        background-color: #2d2d2d; /* Darker background for checkboxes and radio buttons */
        color: white; /* White text for checkboxes/radio */
    }

    /* Specific focus style for numeric inputs */
    .stNumberInput > div > div > input:focus {
        background-color: #444444; /* Slightly lighter background when focused */
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('src/credit_risk_dataset.csv')
    return df

@st.cache_resource
def load_lstm_model():
    models = {}
    models['lstm'] = keras.models.load_model('src/credit_risk_lstm_model.keras')
    models['preprocessing'] = joblib.load('src/preprocessing_artifacts.pkl')
    models['evaluation_results'] = joblib.load('src/evaluation_results.pkl')
    return models

@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    if df_processed['person_emp_length'].isnull().any():
        df_processed['person_emp_length'] = df_processed.groupby('loan_intent')['person_emp_length'].transform(
            lambda x: x.fillna(x.median() if not x.median() != x.median() else x.mean())
        )
        df_processed['person_emp_length'].fillna(df_processed['person_emp_length'].median(), inplace=True)
    
    df_processed['credit_risk_score'] = (
        df_processed['loan_int_rate'] * 0.3 +
        df_processed['loan_percent_income'] * 100 * 0.3 +
        (df_processed['cb_person_default_on_file'].map({'Y': 20, 'N': 0})) * 0.2 +
        (100 - df_processed['cb_person_cred_hist_length'].clip(0, 30) * 3.33) * 0.2
    )
    
    df_processed['debt_service_ratio'] = (
        df_processed['loan_amnt'] * (df_processed['loan_int_rate'] / 100) / 
        (df_processed['person_income'] / 12)
    ).clip(0, 2)
    
    df_processed['employment_stability_score'] = np.where(
        df_processed['person_emp_length'] < 1, 0,
        np.where(df_processed['person_emp_length'] < 3, 25,
        np.where(df_processed['person_emp_length'] < 5, 50,
        np.where(df_processed['person_emp_length'] < 10, 75, 100)))
    )
    
    df_processed['age_adjusted_income'] = df_processed['person_income'] / (
        1 + np.exp(-(df_processed['person_age'] - 35) / 10)
    )
    
    df_processed['loan_affordability_index'] = (
        df_processed['person_income'] / df_processed['loan_amnt']
    ).clip(0, 10)
    
    df_processed['estimated_credit_utilization'] = (
        df_processed['loan_amnt'] / 
        (df_processed['person_income'] * 0.3) 
    ).clip(0, 2)
    
    df_processed['risk_category'] = pd.cut(
        df_processed['credit_risk_score'],
        bins=[0, 30, 50, 70, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    )
    
    df_processed['debt_to_income_category'] = pd.cut(
        df_processed['loan_percent_income'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    df_processed['age_group'] = pd.cut(
        df_processed['person_age'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Senior', 'Elder']
    )
    
    df_processed['income_category'] = pd.qcut(
        df_processed['person_income'],
        q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        duplicates='drop'
    )
    
    return df_processed

def calculate_business_metrics(df):
    metrics = {
        'portfolio_size': len(df),
        'total_exposure': df['loan_amnt'].sum(),
        'default_rate': (df['loan_status'] == 1).mean(),
        'avg_interest_rate': df['loan_int_rate'].mean(),
        'avg_loan_amount': df['loan_amnt'].mean(),
        'avg_income': df['person_income'].mean(),
        'risk_adjusted_return': (
            df['loan_int_rate'].mean() - 
            (df['loan_status'] == 1).mean() * 100
        ),
        'portfolio_quality_score': 100 - (df['loan_status'] == 1).mean() * 100
    }
    return metrics

def predict_with_lstm(input_data, models):
    try:
        scaler = models['preprocessing']['scaler']
        imputer = models['preprocessing']['imputer']
        feature_names = models['preprocessing']['feature_names']
        
        input_df = pd.DataFrame([input_data])
        
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        input_scaled = np.nan_to_num(input_scaled, nan=0.0)
        
        input_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        probability = models['lstm'].predict(input_lstm, verbose=0).flatten()[0]
        prediction = int(probability > 0.5)
        
        if probability < 0.3:
            decision = "APPROVE"
            recommendation = "Low risk applicant. Recommend standard terms."
        elif probability < 0.5:
            decision = "APPROVE WITH CONDITIONS"
            recommendation = "Medium risk. Suggest higher interest rate or collateral."
        elif probability < 0.7:
            decision = "MANUAL REVIEW"
            recommendation = "High risk. Requires senior credit officer review."
        else:
            decision = "REJECT"
            recommendation = "Very high risk. Recommend rejection."
        
        return {
            'prediction': prediction,
            'probability': probability,
            'decision': decision,
            'recommendation': recommendation
        }
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {
            'prediction': 0,
            'probability': 0.5,
            'decision': "ERROR",
            'recommendation': "System error. Please review manually."
        }

def render_dashboard(df, models):
    st.title("Credit Risk Overview")
    
    st.sidebar.header("Dashboard Filters")
    
    loan_grade_filter = st.sidebar.multiselect(
        "Loan Grade",
        options=sorted(df['loan_grade'].unique()),
        default=sorted(df['loan_grade'].unique())
    )
    
    loan_intent_filter = st.sidebar.multiselect(
        "Loan Purpose",
        options=sorted(df['loan_intent'].unique()),
        default=sorted(df['loan_intent'].unique())
    )
    
    home_ownership_filter = st.sidebar.multiselect(
        "Home Ownership Status",
        options=sorted(df['person_home_ownership'].unique()),
        default=sorted(df['person_home_ownership'].unique())
    )
    
    income_range = st.sidebar.slider(
        "Income Range ($)",
        min_value=int(df['person_income'].min()),
        max_value=int(df['person_income'].max()),
        value=(int(df['person_income'].min()), int(df['person_income'].max())),
        step=5000
    )
    
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['person_age'].min()),
        max_value=int(df['person_age'].max()),
        value=(int(df['person_age'].min()), int(df['person_age'].max())),
        step=1
    )
    
    df_filtered = df[
        (df['loan_grade'].isin(loan_grade_filter)) &
        (df['loan_intent'].isin(loan_intent_filter)) &
        (df['person_home_ownership'].isin(home_ownership_filter)) &
        (df['person_income'] >= income_range[0]) &
        (df['person_income'] <= income_range[1]) &
        (df['person_age'] >= age_range[0]) &
        (df['person_age'] <= age_range[1])
    ]
    
    if len(df_filtered) == 0:
        st.warning("No data matches the selected filters. Please adjust your filters.")
        return
    
    metrics = calculate_business_metrics(df_filtered)
    df_processed = preprocess_data(df_filtered)
    
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_portfolio = len(df_filtered) - len(df)
        st.metric(
            label="Portfolio Size",
            value=f"{metrics['portfolio_size']:,}",
            delta=f"{delta_portfolio:+,} vs total" if delta_portfolio != 0 else "All loans"
        )
    
    with col2:
        st.metric(
            label="Total Exposure",
            value=f"${metrics['total_exposure']:,.0f}",
            delta=f"Avg: ${metrics['avg_loan_amount']:,.0f}"
        )
    
    with col3:
        total_default_rate = (df['loan_status'] == 1).mean()
        delta_default = metrics['default_rate'] - total_default_rate
        st.metric(
            label="Default Rate",
            value=f"{metrics['default_rate']:.2%}",
            delta=f"{delta_default:+.2%} vs total"
        )
    
    with col4:
        st.metric(
            label="Risk-Adjusted Return",
            value=f"{metrics['risk_adjusted_return']:.2f}%",
            delta=f"Avg Rate: {metrics['avg_interest_rate']:.2f}%"
        )
    
    st.markdown("---")
    st.markdown("### Portfolio Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_dist = df_processed['risk_category'].value_counts()
        fig_risk = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="Portfolio Risk Distribution",
            color_discrete_map={
                'Low Risk': '#10b981',
                'Medium Risk': '#f59e0b',
                'High Risk': '#ef4444',
                'Very High Risk': '#7c3aed'
            }
        )
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        default_by_grade = df_filtered.groupby('loan_grade')['loan_status'].agg(['mean', 'count'])
        default_by_grade['default_rate'] = default_by_grade['mean'] * 100
        
        fig_grade = px.bar(
            x=default_by_grade.index,
            y=default_by_grade['default_rate'],
            title="Default Rate by Loan Grade",
            labels={'x': 'Loan Grade', 'y': 'Default Rate (%)'},
            color=default_by_grade['default_rate'],
            color_continuous_scale='Reds'
        )
        fig_grade.update_traces(text=default_by_grade['default_rate'].round(1), textposition='outside')
        st.plotly_chart(fig_grade, use_container_width=True)
    
    st.markdown("### Portfolio Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_by_intent = df_filtered.groupby('loan_intent')['loan_status'].mean() * 100
        fig_intent = px.bar(
            x=default_by_intent.values,
            y=default_by_intent.index,
            orientation='h',
            title="Default Rate by Loan Purpose",
            labels={'x': 'Default Rate (%)', 'y': 'Loan Purpose'},
            color=default_by_intent.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_intent, use_container_width=True)
    
    with col2:
        income_bins = pd.qcut(df_filtered['person_income'], q=5, duplicates='drop')
        default_by_income = df_filtered.groupby(income_bins)['loan_status'].mean() * 100
        
        fig_income = px.line(
            x=[str(x) for x in default_by_income.index],
            y=default_by_income.values,
            title="Default Rate by Income Level",
            labels={'x': 'Income Range', 'y': 'Default Rate (%)'},
            markers=True
        )
        fig_income.update_traces(line_color='#ef4444', line_width=3, marker_size=10)
        st.plotly_chart(fig_income, use_container_width=True)
    
    st.markdown("### Risk Concentration Analysis")
    
    risk_matrix = pd.crosstab(
        df_filtered['loan_intent'],
        df_filtered['person_home_ownership'],
        values=df_filtered['loan_status'],
        aggfunc='mean'
    ) * 100
    
    fig_heatmap = px.imshow(
        risk_matrix,
        labels=dict(x="Home Ownership", y="Loan Intent", color="Default Rate (%)"),
        title="Default Rate Heatmap by Loan Intent and Home Ownership",
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    fig_heatmap.update_traces(text=risk_matrix.round(1), texttemplate='%{text}')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("### Dynamic Data Preview")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### Filtered Loan Portfolio Data")
    
    with col2:
        num_rows = st.selectbox(
            "Rows to display",
            options=[10, 25, 50, 100, 200],
            index=2
        )
    
    display_columns = st.multiselect(
        "Select columns to display",
        options=df_filtered.columns.tolist(),
        default=['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 
                'loan_grade', 'loan_intent', 'loan_status', 'person_home_ownership']
    )
    
    if display_columns:
        st.dataframe(
            df_filtered[display_columns].head(num_rows),
            use_container_width=True,
            height=400
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Showing {min(num_rows, len(df_filtered))} of {len(df_filtered)} filtered records")
        with col2:
            st.info(f"Total records in dataset: {len(df):,}")
        with col3:
            filter_percentage = (len(df_filtered) / len(df) * 100) if len(df) > 0 else 0
            st.info(f"Filter coverage: {filter_percentage:.1f}%")
    else:
        st.warning("Please select at least one column to display.")

def render_credit_assessment(models):
    st.title("Credit Assessment & Decision Engine")
    
    st.markdown("""
    Assess credit applications 
    and provide automated decision recommendations based on risk analysis.
    """)
    
    with st.form("credit_application"):
        st.markdown("### Applicant Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Personal Details")
            person_age = st.number_input("Age", min_value=18, max_value=100, value=35)
            person_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
            person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        
        with col2:
            st.markdown("#### Loan Details")
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=15000, step=500)
            loan_int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)
            loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            loan_intent = st.selectbox(
                "Loan Purpose",
                ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
            )
        
        with col3:
            st.markdown("#### Credit History")
            cb_person_default_on_file = st.selectbox("Previous Default", ['N', 'Y'])
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)
            
            st.markdown("#### Additional Assessment")
            has_collateral = st.selectbox("Collateral Available", ['No', 'Yes'])
            existing_customer = st.selectbox("Existing Customer", ['No', 'Yes'])
        
        submitted = st.form_submit_button("Assess Application", type="primary")
    
    if submitted:
        loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
        
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }
        
        grade_risk = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        input_data['grade_risk_score'] = grade_risk[loan_grade]
        input_data['combined_risk_score'] = loan_int_rate * input_data['grade_risk_score'] / 10
        input_data['income_to_age_ratio'] = person_income / (person_age + 1)
        input_data['loan_to_income_ratio'] = loan_amnt / (person_income + 1)
        input_data['credit_utilization'] = loan_amnt / (person_income * cb_person_cred_hist_length + 1)
        input_data['person_income_log'] = np.log1p(person_income)
        input_data['loan_amnt_log'] = np.log1p(loan_amnt)
        
        input_data['loan_grade_encoded'] = grade_risk.get(loan_grade, 4)
        
        input_data['person_home_ownership_OTHER'] = 1 if person_home_ownership == 'OTHER' else 0
        input_data['person_home_ownership_OWN'] = 1 if person_home_ownership == 'OWN' else 0
        input_data['person_home_ownership_RENT'] = 1 if person_home_ownership == 'RENT' else 0
        input_data['cb_person_default_on_file_Y'] = 1 if cb_person_default_on_file == 'Y' else 0
        
        intent_encoding = {
            'PERSONAL': 0.091, 'EDUCATION': 0.089, 'MEDICAL': 0.081,
            'VENTURE': 0.145, 'HOMEIMPROVEMENT': 0.083, 'DEBTCONSOLIDATION': 0.090
        }
        input_data['loan_intent_target_encoded'] = intent_encoding.get(loan_intent, 0.090)
        
        if person_emp_length < 2:
            input_data['employment_stability_encoded'] = 1
        elif person_emp_length < 5:
            input_data['employment_stability_encoded'] = 2
        elif person_emp_length < 10:
            input_data['employment_stability_encoded'] = 3
        else:
            input_data['employment_stability_encoded'] = 4
        
        result = predict_with_lstm(input_data, models)
        
        st.markdown("---")
        st.markdown("### Assessment Results")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            if result['decision'] == "APPROVE":
                st.success(f"### Decision: {result['decision']}")
            elif result['decision'] == "APPROVE WITH CONDITIONS":
                st.warning(f"### Decision: {result['decision']}")
            elif result['decision'] == "MANUAL REVIEW":
                st.info(f"### Decision: {result['decision']}")
            else:
                st.error(f"### Decision: {result['decision']}")
            
            st.markdown(f"**Risk Score:** {result['probability']:.1%}")
            st.markdown(f"**Recommendation:** {result['recommendation']}")
            
            st.markdown("#### Key Risk Factors")
            risk_factors = []
            
            if loan_int_rate > 15:
                risk_factors.append(f"High interest rate ({loan_int_rate:.1f}%)")
            if loan_percent_income > 0.3:
                risk_factors.append(f"High debt-to-income ratio ({loan_percent_income:.1%})")
            if cb_person_default_on_file == 'Y':
                risk_factors.append("Previous default history")
            if person_emp_length < 2:
                risk_factors.append("Limited employment history")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- No significant risk factors identified")
        
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Probability", 'font': {'size': 24}},
                number = {'suffix': '%', 'font': {'size': 40}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#1f2937"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d1fae5'},
                        {'range': [30, 50], 'color': '#fed7aa'},
                        {'range': [50, 70], 'color': '#fecaca'},
                        {'range': [70, 100], 'color': '#fca5a5'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("### Detailed Credit Analysis Report")
        
        report_data = {
            'Metric': [
                'Debt-to-Income Ratio',
                'Credit Utilization',
                'Employment Stability Score',
                'Interest Coverage Ratio',
                'Loan Affordability Index'
            ],
            'Value': [
                f"{loan_percent_income:.1%}",
                f"{(loan_amnt / (person_income * 0.3)):.1%}",
                f"{min(100, person_emp_length * 10):.0f}/100",
                f"{person_income / (loan_amnt * loan_int_rate / 100):.2f}x",
                f"{person_income / loan_amnt:.2f}"
            ],
            'Status': [
                'Good' if loan_percent_income < 0.3 else 'Concern',
                'Good' if loan_amnt / (person_income * 0.3) < 0.5 else 'High',
                'Excellent' if person_emp_length > 5 else 'Fair',
                'Strong' if person_income / (loan_amnt * loan_int_rate / 100) > 3 else 'Weak',
                'Good' if person_income / loan_amnt > 3 else 'Low'
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        
        def highlight_status(val):
            if val == 'Good' or val == 'Strong' or val == 'Excellent':
                return 'background-color: #d1fae5'
            elif val == 'Fair':
                return 'background-color: #fed7aa'
            else:
                return 'background-color: #fecaca'
        
        styled_df = df_report.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

def render_data_exploration(df):
    st.title("Data Exploration & Analytics Center")
    
    df_processed = preprocess_data(df)
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Single Feature Distribution", "Correlation Analysis", "Bivariate Relationship Analysis", "Advanced Segmentation"]
    )
    
    if analysis_type == "Single Feature Distribution":
        st.markdown("### Single Feature Distribution Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        all_cols = numeric_cols + categorical_cols
        selected_col = st.selectbox("Select Column for Analysis", all_cols)
        
        if selected_col in numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                bin_count = st.slider("Number of bins", min_value=10, max_value=50, value=30)
                fig_hist = px.histogram(
                    df,
                    x=selected_col,
                    nbins=bin_count,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#3b82f6']
                )
                
                mean_val = df[selected_col].mean()
                median_val = df[selected_col].median()
                
                fig_hist.add_vline(x=mean_val, line_dash="dash", 
                                 line_color="red", annotation_text=f"Mean: {mean_val:.2f}")
                fig_hist.add_vline(x=median_val, line_dash="dash", 
                                 line_color="green", annotation_text=f"Median: {median_val:.2f}")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                show_outliers = st.checkbox("Show outliers", value=True)
                fig_box = px.box(
                    df,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}",
                    color_discrete_sequence=['#10b981'],
                    points="outliers" if show_outliers else None
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("#### Descriptive Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{df[selected_col].mean():.2f}")
                st.metric("Std Dev", f"{df[selected_col].std():.2f}")
            
            with col2:
                st.metric("Median", f"{df[selected_col].median():.2f}")
                st.metric("IQR", f"{df[selected_col].quantile(0.75) - df[selected_col].quantile(0.25):.2f}")
            
            with col3:
                st.metric("Min", f"{df[selected_col].min():.2f}")
                st.metric("Max", f"{df[selected_col].max():.2f}")
            
            with col4:
                st.metric("Skewness", f"{df[selected_col].skew():.2f}")
                st.metric("Kurtosis", f"{df[selected_col].kurtosis():.2f}")
        
        else:
            value_counts = df[selected_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {selected_col}",
                    labels={'x': selected_col, 'y': 'Count'},
                    color=value_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Proportion of {selected_col}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("#### Frequency Table")
            freq_df = pd.DataFrame({
                selected_col: value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(freq_df, use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        st.markdown("### Correlation Analysis")
        
        correlation_method = st.radio(
            "Select correlation method",
            ["Pearson", "Spearman", "Kendall"],
            horizontal=True
        )
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if correlation_method == "Pearson":
            corr_matrix = numeric_df.corr(method='pearson')
        elif correlation_method == "Spearman":
            corr_matrix = numeric_df.corr(method='spearman')
        else:
            corr_matrix = numeric_df.corr(method='kendall')
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            show_annotations = st.checkbox("Show values", value=True)
            color_scale = st.selectbox(
                "Color scale",
                ["RdBu", "Viridis", "Plasma", "Inferno"]
            )
        
        with col1:
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f' if show_annotations else None,
                title=f"{correlation_method} Correlation Heatmap",
                color_continuous_scale=color_scale,
                aspect='auto'
            )
            fig_corr.update_layout(height=700)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("#### Top Correlations with Loan Status")
        
        loan_status_corr = corr_matrix['loan_status'].abs().sort_values(ascending=False)[1:11]
        
        fig_top_corr = px.bar(
            x=loan_status_corr.values,
            y=loan_status_corr.index,
            orientation='h',
            title="Top 10 Features Correlated with Loan Status",
            labels={'x': 'Absolute Correlation', 'y': 'Feature'},
            color=loan_status_corr.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_top_corr, use_container_width=True)
    
    elif analysis_type == "Bivariate Relationship Analysis":
        st.markdown("### Bivariate Relationship Analysis")
        
        col1, col2 = st.columns(2)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        all_cols = numeric_cols + categorical_cols
        
        with col1:
            feature_x = st.selectbox("Select X-axis Feature", all_cols, key="feat_x")
        
        with col2:
            feature_y = st.selectbox("Select Y-axis Feature", all_cols, key="feat_y")
        
        if feature_x == feature_y:
            st.warning("Please select different features for X and Y axes.")
        else:
            color_by = st.selectbox(
                "Color by",
                ["None", "loan_status", "loan_grade", "loan_intent", "person_home_ownership"]
            )
            
            if feature_x in numeric_cols and feature_y in numeric_cols:
                fig = px.scatter(
                    df,
                    x=feature_x,
                    y=feature_y,
                    color=color_by if color_by != "None" else None,
                    title=f"Relationship between {feature_x} and {feature_y}",
                    opacity=0.6,
                    trendline="ols" if st.checkbox("Show trendline") else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
                corr_coef = df[[feature_x, feature_y]].corr().iloc[0, 1]
                st.info(f"Correlation Coefficient: {corr_coef:.3f}")
            
            elif feature_x in categorical_cols and feature_y in numeric_cols:
                plot_type = st.radio("Plot type", ["Box", "Violin"], horizontal=True)
                
                if plot_type == "Box":
                    fig = px.box(
                        df,
                        x=feature_x,
                        y=feature_y,
                        color=color_by if color_by != "None" else None,
                        title=f"Distribution of {feature_y} by {feature_x}"
                    )
                else:
                    fig = px.violin(
                        df,
                        x=feature_x,
                        y=feature_y,
                        color=color_by if color_by != "None" else None,
                        title=f"Distribution of {feature_y} by {feature_x}",
                        box=True
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            elif feature_x in numeric_cols and feature_y in categorical_cols:
                plot_type = st.radio("Plot type", ["Box", "Violin"], horizontal=True)
                
                if plot_type == "Box":
                    fig = px.box(
                        df,
                        y=feature_x,
                        x=feature_y,
                        color=color_by if color_by != "None" else None,
                        title=f"Distribution of {feature_x} by {feature_y}",
                        orientation='h'
                    )
                else:
                    fig = px.violin(
                        df,
                        y=feature_x,
                        x=feature_y,
                        color=color_by if color_by != "None" else None,
                        title=f"Distribution of {feature_x} by {feature_y}",
                        box=True,
                        orientation='h'
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                crosstab = pd.crosstab(df[feature_x], df[feature_y])
                
                visualization = st.radio(
                    "Visualization type",
                    ["Heatmap", "Stacked Bar", "Grouped Bar"],
                    horizontal=True
                )
                
                if visualization == "Heatmap":
                    fig = px.imshow(
                        crosstab,
                        text_auto=True,
                        title=f"Cross-tabulation: {feature_x} vs {feature_y}",
                        color_continuous_scale='Blues'
                    )
                elif visualization == "Stacked Bar":
                    fig = px.bar(
                        crosstab.T,
                        title=f"Stacked Bar: {feature_x} vs {feature_y}",
                        barmode='stack'
                    )
                else:
                    fig = px.bar(
                        crosstab.T,
                        title=f"Grouped Bar: {feature_x} vs {feature_y}",
                        barmode='group'
                    )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.markdown("### Advanced Segmentation Analysis")
        
        segmentation_by = st.selectbox(
            "Segment by",
            ["Risk Category", "Income Category", "Age Group", "Loan Grade"]
        )
        
        if segmentation_by == "Risk Category":
            segment_col = 'risk_category'
        elif segmentation_by == "Income Category":
            segment_col = 'income_category'
        elif segmentation_by == "Age Group":
            segment_col = 'age_group'
        else:
            segment_col = 'loan_grade'
        
        segment_stats = df_processed.groupby(segment_col).agg({
            'loan_status': ['count', 'mean'],
            'loan_amnt': ['mean', 'sum'],
            'person_income': 'mean',
            'loan_int_rate': 'mean'
        })
        
        segment_stats.columns = ['Count', 'Default_Rate', 'Avg_Loan_Amount', 
                                'Total_Exposure', 'Avg_Income', 'Avg_Interest_Rate']
        segment_stats = segment_stats.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_size = px.scatter(
                segment_stats,
                x='Avg_Income',
                y='Default_Rate',
                size='Total_Exposure',
                color=segment_col,
                title=f"Risk vs Income by {segmentation_by}",
                hover_data=['Count', 'Avg_Loan_Amount'],
                size_max=60
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        with col2:
            fig_sunburst = px.sunburst(
                df_processed,
                path=[segment_col, 'loan_intent'],
                values='loan_amnt',
                title=f"Loan Distribution by {segmentation_by} and Purpose"
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        st.markdown(f"#### Detailed Statistics by {segmentation_by}")
        
        styled_stats = segment_stats.style.format({
            'Count': '{:,.0f}',
            'Default_Rate': '{:.2%}',
            'Avg_Loan_Amount': '${:,.0f}',
            'Total_Exposure': '${:,.0f}',
            'Avg_Income': '${:,.0f}',
            'Avg_Interest_Rate': '{:.2f}%'
        })
        
        st.dataframe(styled_stats, use_container_width=True, hide_index=True)
        
        selected_segment = st.selectbox(
            f"Select {segmentation_by} for detailed view",
            segment_stats[segment_col].tolist()
        )
        
        segment_data = df_processed[df_processed[segment_col] == selected_segment]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_rate_segment = segment_data['loan_status'].mean()
            st.metric(
                "Segment Default Rate",
                f"{default_rate_segment:.2%}",
                delta=f"{(default_rate_segment - df['loan_status'].mean()):.2%} vs overall"
            )
        
        with col2:
            avg_loan_segment = segment_data['loan_amnt'].mean()
            st.metric(
                "Average Loan Amount",
                f"${avg_loan_segment:,.0f}",
                delta=f"${(avg_loan_segment - df['loan_amnt'].mean()):+,.0f} vs overall"
            )
        
        with col3:
            count_segment = len(segment_data)
            st.metric(
                "Segment Size",
                f"{count_segment:,}",
                delta=f"{(count_segment/len(df)):.1%} of total"
            )

def render_automation_settings():
    st.title("Business Process Automation Settings")
    
    st.markdown("""
    Configure automated decision-making rules and thresholds for the credit risk assessment system.
    """)
    
    st.markdown("### Decision Automation Rules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Automatic Approval Thresholds")
        auto_approve_threshold = st.slider(
            "Maximum Risk Score for Auto-Approval",
            min_value=0.0,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Applications with risk scores below this threshold will be automatically approved"
        )
        
        min_income_auto = st.number_input(
            "Minimum Income for Auto-Approval ($)",
            min_value=0,
            value=40000,
            step=5000
        )
        
        max_dti_auto = st.slider(
            "Maximum Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05
        )
    
    with col2:
        st.markdown("#### Automatic Rejection Thresholds")
        auto_reject_threshold = st.slider(
            "Minimum Risk Score for Auto-Rejection",
            min_value=0.5,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Applications with risk scores above this threshold will be automatically rejected"
        )
        
        prev_default_reject = st.checkbox(
            "Auto-reject if previous default",
            value=True
        )
        
        min_emp_length = st.number_input(
            "Minimum Employment Length (years)",
            min_value=0.0,
            value=1.0,
            step=0.5
        )
    
    st.markdown("---")
    st.markdown("### Notification & Escalation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Email Notifications")
        notify_high_risk = st.checkbox("Notify on high-risk applications", value=True)
        notify_large_loans = st.checkbox("Notify on large loan amounts", value=True)
        large_loan_threshold = st.number_input(
            "Large loan threshold ($)",
            min_value=0,
            value=50000,
            step=5000,
            disabled=not notify_large_loans
        )
    
    with col2:
        st.markdown("#### Escalation Rules")
        escalate_manual_review = st.checkbox("Escalate manual reviews after delay", value=True)
        escalation_hours = st.number_input(
            "Hours before escalation",
            min_value=1,
            value=24,
            step=1,
            disabled=not escalate_manual_review
        )
    
    st.markdown("---")
    st.markdown("### API Integration Settings")
    
    api_enabled = st.checkbox("Enable API for external systems", value=False)
    
    if api_enabled:
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password", value="**********************")
            webhook_url = st.text_input("Webhook URL", value="https://api.example.com/credit-decisions")
        with col2:
            rate_limit = st.number_input("API Rate Limit (requests/hour)", min_value=1, value=100)
            batch_processing = st.checkbox("Enable batch processing", value=True)
    
    st.markdown("---")
    if st.button("Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")
        
        st.markdown("### Current Configuration Summary")
        config_summary = {
            "Auto-Approval": {
                "Risk Threshold": f"≤ {auto_approve_threshold:.0%}",
                "Min Income": f"${min_income_auto:,}",
                "Max DTI": f"≤ {max_dti_auto:.0%}"
            },
            "Auto-Rejection": {
                "Risk Threshold": f"≥ {auto_reject_threshold:.0%}",
                "Previous Default": "Yes" if prev_default_reject else "No",
                "Min Employment": f"{min_emp_length} years"
            },
            "Notifications": {
                "High Risk Alerts": "Enabled" if notify_high_risk else "Disabled",
                "Large Loan Alerts": f"Enabled (>${large_loan_threshold:,})" if notify_large_loans else "Disabled"
            }
        }
        
        for category, settings in config_summary.items():
            st.markdown(f"**{category}:**")
            for key, value in settings.items():
                st.markdown(f"- {key}: {value}")

def main():
    with st.spinner("Loading system components..."):
        df = load_data()
        models = load_lstm_model()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        [
            "Main Dashboard",
            "Credit Analyst",
            "Data Exploration",
            "Automation Settings"
        ]
    )
    
    if page == "Main Dashboard":
        render_dashboard(df, models)
    elif page == "Credit Analyst":
        render_credit_assessment(models)
    elif page == "Data Exploration":
        render_data_exploration(df)
    else:
        render_automation_settings()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Credit Risk Analysis System**  
        Version 2.0 - AI Automation  
        © 2025 DatathonUI
        """
    )

if __name__ == "__main__":
    main()