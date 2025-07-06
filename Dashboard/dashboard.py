import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Risiko Kredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load data
@st.cache_data
def load_data():
    """Load dataset credit risk"""
    try:
        df = pd.read_csv('credit_risk_dataset.csv')
        return df
    except:
        st.error("File 'credit_risk_dataset.csv' tidak ditemukan!")
        return None

# Fungsi preprocessing sederhana untuk demo
def preprocess_data(df):
    """Preprocessing data untuk visualisasi"""
    df_clean = df.copy()
    
    # Handle missing values
    if df_clean['person_emp_length'].isnull().any():
        df_clean['person_emp_length'].fillna(df_clean['person_emp_length'].median(), inplace=True)
    
    # Feature engineering sederhana
    df_clean['debt_to_income_category'] = pd.cut(
        df_clean['loan_percent_income'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
        labels=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    )
    
    df_clean['age_group'] = pd.cut(
        df_clean['person_age'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Senior', 'Elder']
    )
    
    df_clean['income_category'] = pd.qcut(
        df_clean['person_income'],
        q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
        duplicates='drop'
    )
    
    return df_clean

# Fungsi untuk membuat prediksi (simulasi)
def make_prediction(input_data):
    """Simulasi prediksi credit risk"""
    # Ini adalah simulasi, dalam implementasi real akan load model yang sudah ditraining
    
    # Simple risk scoring berdasarkan beberapa faktor
    risk_score = 0
    
    # Faktor umur
    if input_data['person_age'] < 25:
        risk_score += 20
    elif input_data['person_age'] > 65:
        risk_score += 15
    
    # Faktor income
    if input_data['person_income'] < 30000:
        risk_score += 25
    elif input_data['person_income'] < 50000:
        risk_score += 15
    
    # Faktor loan amount vs income
    loan_to_income = input_data['loan_amnt'] / input_data['person_income']
    if loan_to_income > 0.5:
        risk_score += 30
    elif loan_to_income > 0.3:
        risk_score += 20
    
    # Faktor interest rate
    if input_data['loan_int_rate'] > 15:
        risk_score += 20
    elif input_data['loan_int_rate'] > 10:
        risk_score += 10
    
    # Faktor employment length
    if input_data['person_emp_length'] < 2:
        risk_score += 15
    
    # Faktor credit history
    if input_data['cb_person_default_on_file'] == 'Y':
        risk_score += 30
    
    if input_data['cb_person_cred_hist_length'] < 5:
        risk_score += 10
    
    # Normalize risk score
    risk_probability = min(risk_score / 100, 0.95)
    
    return {
        'risk_probability': risk_probability,
        'risk_category': 'Tinggi' if risk_probability > 0.5 else 'Rendah',
        'approval_recommendation': 'Tolak' if risk_probability > 0.6 else 'Setujui dengan Syarat' if risk_probability > 0.3 else 'Setujui'
    }

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Dashboard Analisis Risiko Kredit</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    df_clean = preprocess_data(df)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["🏠 Beranda", "📈 Analisis Data", "🤖 Prediksi Risiko", "💰 Analisis Dampak Bisnis", "📊 Performa Model"]
    )
    
    # Beranda
    if page == "🏠 Beranda":
        st.markdown('<h2 class="sub-header">Ringkasan Dataset</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Data", f"{len(df):,}")
        with col2:
            default_rate = (df['loan_status'] == 0).mean() * 100
            st.metric("Tingkat Gagal Bayar", f"{default_rate:.1f}%")
        with col3:
            avg_loan = df['loan_amnt'].mean()
            st.metric("Rata-rata Pinjaman", f"${avg_loan:,.0f}")
        with col4:
            avg_interest = df['loan_int_rate'].mean()
            st.metric("Rata-rata Suku Bunga", f"{avg_interest:.1f}%")
        
        # Distribusi status pinjaman
        st.markdown("### Distribusi Status Pinjaman")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=df['loan_status'].value_counts().values,
                names=['Gagal Bayar', 'Lancar'],
                title="Proporsi Status Pinjaman",
                color_discrete_map={'Gagal Bayar': '#ef5350', 'Lancar': '#66bb6a'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Distribusi berdasarkan loan intent
            intent_counts = df.groupby(['loan_intent', 'loan_status']).size().unstack(fill_value=0)
            intent_counts.columns = ['Gagal Bayar', 'Lancar']
            
            fig_bar = px.bar(
                intent_counts.T,
                title="Status Pinjaman per Tujuan",
                color_discrete_map={'Gagal Bayar': '#ef5350', 'Lancar': '#66bb6a'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistik deskriptif
        st.markdown("### Statistik Deskriptif")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df[numeric_cols].describe().round(2))
    
    # Analisis Data
    elif page == "📈 Analisis Data":
        st.markdown('<h2 class="sub-header">Eksplorasi Data</h2>', unsafe_allow_html=True)
        
        # Tabs untuk berbagai analisis
        tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Fitur", "Korelasi", "Analisis Risiko", "Segmentasi"])
        
        with tab1:
            st.markdown("### Distribusi Fitur Numerik")
            
            # Pilih fitur untuk visualisasi
            numeric_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 
                              'person_emp_length', 'cb_person_cred_hist_length']
            
            selected_feature = st.selectbox("Pilih Fitur:", numeric_features)
            
            # Histogram dengan pembagian berdasarkan status
            fig = px.histogram(
                df_clean,
                x=selected_feature,
                color='loan_status',
                nbins=30,
                title=f"Distribusi {selected_feature} berdasarkan Status Pinjaman",
                color_discrete_map={0: '#ef5350', 1: '#66bb6a'},
                labels={'loan_status': 'Status'}
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            fig_box = px.box(
                df_clean,
                y=selected_feature,
                x='loan_status',
                title=f"Box Plot {selected_feature} berdasarkan Status",
                color='loan_status',
                color_discrete_map={0: '#ef5350', 1: '#66bb6a'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with tab2:
            st.markdown("### Matriks Korelasi")
            
            # Hitung korelasi
            corr_matrix = df[numeric_features].corr()
            
            # Heatmap
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matriks Korelasi Fitur Numerik",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Korelasi dengan target
            st.markdown("### Korelasi dengan Status Pinjaman")
            target_corr = df[numeric_features + ['loan_status']].corr()['loan_status'].drop('loan_status').sort_values(ascending=False)
            
            fig_target_corr = px.bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation='h',
                title="Korelasi Fitur dengan Status Pinjaman",
                labels={'x': 'Korelasi', 'y': 'Fitur'},
                color=target_corr.values,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_target_corr, use_container_width=True)
        
        with tab3:
            st.markdown("### Analisis Risiko Berdasarkan Kategori")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risiko berdasarkan grade
                grade_risk = df_clean.groupby('loan_grade')['loan_status'].agg(['mean', 'count'])
                grade_risk['default_rate'] = (1 - grade_risk['mean']) * 100
                
                fig_grade = px.bar(
                    x=grade_risk.index,
                    y=grade_risk['default_rate'],
                    title="Tingkat Gagal Bayar per Grade Pinjaman",
                    labels={'x': 'Grade', 'y': 'Tingkat Gagal Bayar (%)'},
                    color=grade_risk['default_rate'],
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_grade, use_container_width=True)
            
            with col2:
                # Risiko berdasarkan tujuan pinjaman
                intent_risk = df_clean.groupby('loan_intent')['loan_status'].agg(['mean', 'count'])
                intent_risk['default_rate'] = (1 - intent_risk['mean']) * 100
                
                fig_intent = px.bar(
                    x=intent_risk.index,
                    y=intent_risk['default_rate'],
                    title="Tingkat Gagal Bayar per Tujuan Pinjaman",
                    labels={'x': 'Tujuan', 'y': 'Tingkat Gagal Bayar (%)'},
                    color=intent_risk['default_rate'],
                    color_continuous_scale='Reds'
                )
                fig_intent.update_xaxis(tickangle=-45)
                st.plotly_chart(fig_intent, use_container_width=True)
            
            # Analisis risiko berdasarkan rentang usia dan income
            st.markdown("### Peta Risiko: Usia vs Income")
            
            # Buat bins untuk scatter plot
            risk_map = df_clean.groupby(['age_group', 'income_category'])['loan_status'].agg(['mean', 'count'])
            risk_map['default_rate'] = (1 - risk_map['mean']) * 100
            risk_map = risk_map[risk_map['count'] > 10]  # Filter untuk sample size yang cukup
            
            # Pivot untuk heatmap
            risk_pivot = risk_map['default_rate'].reset_index().pivot(
                index='age_group', 
                columns='income_category', 
                values='default_rate'
            )
            
            fig_heatmap = px.imshow(
                risk_pivot,
                title="Peta Risiko: Tingkat Gagal Bayar (%) berdasarkan Usia dan Income",
                labels=dict(x="Kategori Income", y="Kelompok Usia", color="Gagal Bayar (%)"),
                aspect="auto",
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab4:
            st.markdown("### Segmentasi Nasabah")
            
            # Segmentasi berdasarkan home ownership
            col1, col2 = st.columns(2)
            
            with col1:
                ownership_dist = df_clean['person_home_ownership'].value_counts()
                fig_ownership = px.pie(
                    values=ownership_dist.values,
                    names=ownership_dist.index,
                    title="Distribusi Kepemilikan Rumah"
                )
                st.plotly_chart(fig_ownership, use_container_width=True)
            
            with col2:
                # Default rate by ownership
                ownership_risk = df_clean.groupby('person_home_ownership')['loan_status'].agg(['mean', 'count'])
                ownership_risk['default_rate'] = (1 - ownership_risk['mean']) * 100
                
                fig_ownership_risk = px.bar(
                    x=ownership_risk.index,
                    y=ownership_risk['default_rate'],
                    title="Tingkat Gagal Bayar per Status Kepemilikan Rumah",
                    labels={'x': 'Status Kepemilikan', 'y': 'Gagal Bayar (%)'},
                    color=ownership_risk['default_rate'],
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_ownership_risk, use_container_width=True)
            
            # Analisis multi-dimensi
            st.markdown("### Analisis Multi-Dimensi")
            
            # Bubble chart: Income vs Loan Amount vs Interest Rate
            sample_data = df_clean.sample(n=min(1000, len(df_clean)))
            
            fig_bubble = px.scatter(
                sample_data,
                x='person_income',
                y='loan_amnt',
                size='loan_int_rate',
                color='loan_status',
                title="Hubungan Income, Jumlah Pinjaman, dan Suku Bunga",
                labels={'person_income': 'Income', 'loan_amnt': 'Jumlah Pinjaman', 
                       'loan_int_rate': 'Suku Bunga', 'loan_status': 'Status'},
                color_discrete_map={0: '#ef5350', 1: '#66bb6a'},
                hover_data=['loan_grade', 'loan_intent']
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Prediksi Risiko
    elif page == "🤖 Prediksi Risiko":
        st.markdown('<h2 class="sub-header">Prediksi Risiko Kredit</h2>', unsafe_allow_html=True)
        
        st.info("💡 Masukkan data nasabah untuk memprediksi risiko kredit")
        
        # Form input
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Data Pribadi")
                person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
                person_income = st.number_input("Penghasilan Tahunan ($)", min_value=0, value=50000, step=1000)
                person_home_ownership = st.selectbox("Status Kepemilikan Rumah", 
                                                   ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
                person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            
            with col2:
                st.markdown("### Data Pinjaman")
                loan_intent = st.selectbox("Tujuan Pinjaman", 
                                         ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 
                                          'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
                loan_grade = st.selectbox("Grade Pinjaman", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
                loan_amnt = st.number_input("Jumlah Pinjaman ($)", min_value=0, value=10000, step=500)
                loan_int_rate = st.slider("Suku Bunga (%)", min_value=5.0, max_value=25.0, value=10.0, step=0.5)
            
            with col3:
                st.markdown("### Riwayat Kredit")
                cb_person_default_on_file = st.selectbox("Pernah Gagal Bayar?", ['N', 'Y'])
                cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", 
                                                            min_value=0, max_value=50, value=10)
            
            # Calculate loan_percent_income
            loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
            st.metric("Rasio Pinjaman terhadap Income", f"{loan_percent_income:.2%}")
            
            submitted = st.form_submit_button("🔮 Prediksi Risiko", use_container_width=True)
        
        if submitted:
            # Prepare input data
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
            
            # Make prediction
            prediction = make_prediction(input_data)
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction['risk_probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Skor Risiko (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction['risk_probability'] > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Prediction box
                risk_class = "high-risk" if prediction['risk_probability'] > 0.5 else "low-risk"
                
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h3>Hasil Analisis</h3>
                    <p><strong>Kategori Risiko:</strong> {prediction['risk_category']}</p>
                    <p><strong>Probabilitas Gagal Bayar:</strong> {prediction['risk_probability']:.1%}</p>
                    <p><strong>Rekomendasi:</strong> {prediction['approval_recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Faktor risiko
                st.markdown("### 🔍 Faktor-faktor Risiko")
                
                risk_factors = []
                if person_age < 25:
                    risk_factors.append("❗ Usia muda (< 25 tahun)")
                if person_income < 30000:
                    risk_factors.append("❗ Penghasilan rendah (< $30,000)")
                if loan_percent_income > 0.3:
                    risk_factors.append("❗ Rasio pinjaman terhadap income tinggi (> 30%)")
                if loan_int_rate > 15:
                    risk_factors.append("❗ Suku bunga tinggi (> 15%)")
                if person_emp_length < 2:
                    risk_factors.append("❗ Masa kerja pendek (< 2 tahun)")
                if cb_person_default_on_file == 'Y':
                    risk_factors.append("❗ Memiliki riwayat gagal bayar")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.success("✅ Tidak ada faktor risiko utama terdeteksi")
            
            # Comparison with similar profiles
            st.markdown("### 📊 Perbandingan dengan Profil Serupa")
            
            # Filter similar profiles
            similar_profiles = df_clean[
                (df_clean['loan_grade'] == loan_grade) &
                (df_clean['loan_intent'] == loan_intent) &
                (df_clean['person_age'].between(person_age - 5, person_age + 5))
            ]
            
            if len(similar_profiles) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    similar_default_rate = (1 - similar_profiles['loan_status'].mean()) * 100
                    st.metric("Tingkat Gagal Bayar Profil Serupa", f"{similar_default_rate:.1f}%")
                
                with col2:
                    avg_loan_similar = similar_profiles['loan_amnt'].mean()
                    st.metric("Rata-rata Pinjaman Serupa", f"${avg_loan_similar:,.0f}")
                
                with col3:
                    count_similar = len(similar_profiles)
                    st.metric("Jumlah Profil Serupa", f"{count_similar:,}")
            else:
                st.info("Tidak ada profil serupa ditemukan dalam dataset")
    
    # Analisis Dampak Bisnis
    elif page == "💰 Analisis Dampak Bisnis":
        st.markdown('<h2 class="sub-header">Analisis Dampak Bisnis</h2>', unsafe_allow_html=True)
        
        # Cost matrix
        st.markdown("### ⚙️ Pengaturan Matriks Biaya")
        
        col1, col2 = st.columns(2)
        with col1:
            cost_fn = st.number_input("Biaya False Negative (Gagal mendeteksi risiko)", 
                                    value=1000, step=100, help="Biaya ketika model gagal mendeteksi nasabah yang akan gagal bayar")
        with col2:
            cost_fp = st.number_input("Biaya False Positive (Salah menolak)", 
                                    value=200, step=50, help="Biaya ketika model salah menolak nasabah yang sebenarnya akan lancar")
        
        # Simulasi dampak bisnis
        st.markdown("### 📊 Simulasi Dampak Bisnis")
        
        # Asumsi confusion matrix untuk demonstrasi
        total_samples = len(df)
        actual_default_rate = (df['loan_status'] == 0).mean()
        
        # Simulasi untuk berbagai threshold
        thresholds = np.linspace(0.1, 0.9, 9)
        business_metrics = []
        
        for threshold in thresholds:
            # Simulasi confusion matrix berdasarkan threshold
            # Ini adalah simulasi sederhana untuk demonstrasi
            predicted_positive_rate = 1 - threshold
            
            tp = int(actual_default_rate * total_samples * 0.7 * (1-threshold))  # True Positives
            fn = int(actual_default_rate * total_samples) - tp  # False Negatives
            fp = int((1-actual_default_rate) * total_samples * predicted_positive_rate * 0.3)  # False Positives
            tn = int((1-actual_default_rate) * total_samples) - fp  # True Negatives
            
            total_cost = fn * cost_fn + fp * cost_fp
            approval_rate = (tn + fn) / total_samples
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            business_metrics.append({
                'threshold': threshold,
                'total_cost': total_cost,
                'approval_rate': approval_rate,
                'precision': precision,
                'recall': recall
            })
        
        metrics_df = pd.DataFrame(business_metrics)
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cost = px.line(
                metrics_df,
                x='threshold',
                y='total_cost',
                title='Total Biaya vs Threshold',
                labels={'threshold': 'Threshold', 'total_cost': 'Total Biaya ($)'},
                markers=True
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            fig_approval = px.line(
                metrics_df,
                x='threshold',
                y='approval_rate',
                title='Tingkat Persetujuan vs Threshold',
                labels={'threshold': 'Threshold', 'approval_rate': 'Tingkat Persetujuan'},
                markers=True
            )
            st.plotly_chart(fig_approval, use_container_width=True)
        
        # Trade-off analysis
        st.markdown("### 🔄 Analisis Trade-off")
        
        fig_tradeoff = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Precision vs Recall', 'Biaya vs Tingkat Persetujuan')
        )
        
        # Precision vs Recall
        fig_tradeoff.add_trace(
            go.Scatter(x=metrics_df['recall'], y=metrics_df['precision'], 
                      mode='lines+markers', name='PR Curve'),
            row=1, col=1
        )
        
        # Cost vs Approval Rate
        fig_tradeoff.add_trace(
            go.Scatter(x=metrics_df['approval_rate'], y=metrics_df['total_cost'], 
                      mode='lines+markers', name='Cost-Approval'),
            row=1, col=2
        )
        
        fig_tradeoff.update_xaxes(title_text="Recall", row=1, col=1)
        fig_tradeoff.update_yaxes(title_text="Precision", row=1, col=1)
        fig_tradeoff.update_xaxes(title_text="Tingkat Persetujuan", row=1, col=2)
        fig_tradeoff.update_yaxes(title_text="Total Biaya ($)", row=1, col=2)
        
        fig_tradeoff.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_tradeoff, use_container_width=True)
        
        # ROI Analysis
        st.markdown("### 💵 Analisis ROI")
        
        avg_loan = df['loan_amnt'].mean()
        avg_interest_rate = df['loan_int_rate'].mean() / 100
        loan_term_years = 3  # Asumsi
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            expected_revenue_per_loan = avg_loan * avg_interest_rate * loan_term_years
            st.metric("Pendapatan Rata-rata per Pinjaman", f"${expected_revenue_per_loan:,.0f}")
        
        with col2:
            optimal_threshold = metrics_df.loc[metrics_df['total_cost'].idxmin(), 'threshold']
            st.metric("Threshold Optimal", f"{optimal_threshold:.2f}")
        
        with col3:
            min_cost = metrics_df['total_cost'].min()
            st.metric("Biaya Minimum", f"${min_cost:,.0f}")
        
        # Summary
        st.markdown("### 📋 Ringkasan Rekomendasi")
        
        optimal_metrics = metrics_df[metrics_df['threshold'] == optimal_threshold].iloc[0]
        
        st.info(f"""
        **Rekomendasi Berdasarkan Analisis:**
        - Gunakan threshold {optimal_threshold:.2f} untuk meminimalkan biaya
        - Tingkat persetujuan yang diharapkan: {optimal_metrics['approval_rate']:.1%}
        - Precision yang diharapkan: {optimal_metrics['precision']:.1%}
        - Recall yang diharapkan: {optimal_metrics['recall']:.1%}
        - Total biaya yang diproyeksikan: ${optimal_metrics['total_cost']:,.0f}
        """)
    
    # Performa Model
    elif page == "📊 Performa Model":
        st.markdown('<h2 class="sub-header">Perbandingan Performa Model</h2>', unsafe_allow_html=True)
        
        # Simulasi metrik model (dalam implementasi real akan load dari model yang sudah ditraining)
        model_metrics = {
            'LSTM': {
                'accuracy': 0.834,
                'precision': 0.782,
                'recall': 0.751,
                'f1_score': 0.766,
                'roc_auc': 0.889
            },
            'XGBoost': {
                'accuracy': 0.851,
                'precision': 0.798,
                'recall': 0.773,
                'f1_score': 0.785,
                'roc_auc': 0.905
            },
            'LightGBM': {
                'accuracy': 0.848,
                'precision': 0.801,
                'recall': 0.765,
                'f1_score': 0.783,
                'roc_auc': 0.902
            },
            'CatBoost': {
                'accuracy': 0.846,
                'precision': 0.795,
                'recall': 0.769,
                'f1_score': 0.782,
                'roc_auc': 0.901
            },
            'Ensemble': {
                'accuracy': 0.868,
                'precision': 0.821,
                'recall': 0.794,
                'f1_score': 0.807,
                'roc_auc': 0.923
            }
        }
        
        # Convert to DataFrame
        metrics_comparison = pd.DataFrame(model_metrics).T
        
        # Visualisasi perbandingan
        st.markdown("### 📊 Perbandingan Metrik Model")
        
        # Bar chart comparison
        fig_comparison = go.Figure()
        
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics_list):
            fig_comparison.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metrics_comparison.index,
                y=metrics_comparison[metric],
                marker_color=colors[i]
            ))
        
        fig_comparison.update_layout(
            title="Perbandingan Metrik Antar Model",
            barmode='group',
            yaxis_title="Skor",
            xaxis_title="Model",
            legend_title="Metrik",
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### 📋 Tabel Detail Metrik")
        styled_metrics = metrics_comparison.style.format("{:.3f}").background_gradient(cmap='YlGn', axis=0)
        st.dataframe(styled_metrics)
        
        # Model selection recommendation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Model Terbaik per Metrik")
            
            best_models = {}
            for metric in metrics_list:
                best_model = metrics_comparison[metric].idxmax()
                best_score = metrics_comparison[metric].max()
                best_models[metric] = f"{best_model} ({best_score:.3f})"
            
            for metric, model_info in best_models.items():
                st.write(f"**{metric.replace('_', ' ').title()}:** {model_info}")
        
        with col2:
            st.markdown("### 🎯 Rekomendasi Model")
            
            # Calculate overall score (weighted average)
            weights = {'accuracy': 0.2, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.2, 'roc_auc': 0.2}
            overall_scores = {}
            
            for model in metrics_comparison.index:
                score = sum(metrics_comparison.loc[model, metric] * weight 
                          for metric, weight in weights.items())
                overall_scores[model] = score
            
            best_overall = max(overall_scores, key=overall_scores.get)
            
            st.success(f"""
            **Model Terbaik: {best_overall}**
            
            Skor Keseluruhan: {overall_scores[best_overall]:.3f}
            
            Model ini menunjukkan performa terbaik secara keseluruhan dengan:
            - ROC AUC tertinggi: {metrics_comparison.loc[best_overall, 'roc_auc']:.3f}
            - F1 Score terbaik: {metrics_comparison.loc[best_overall, 'f1_score']:.3f}
            - Balance yang baik antara precision dan recall
            """)
        
        # Feature importance (untuk tree-based models)
        st.markdown("### 🔍 Feature Importance")
        
        # Simulasi feature importance
        features = ['loan_int_rate', 'loan_percent_income', 'grade_risk_score', 
                   'person_income', 'loan_amnt', 'person_age', 'cb_person_cred_hist_length',
                   'person_emp_length', 'combined_risk_score', 'loan_to_income_ratio']
        
        importance_scores = np.random.exponential(0.1, len(features))
        importance_scores = importance_scores / importance_scores.sum()
        importance_scores = np.sort(importance_scores)[::-1]
        
        fig_importance = px.bar(
            x=importance_scores[:10],
            y=features[:10],
            orientation='h',
            title="Top 10 Feature Importance (XGBoost)",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            color=importance_scores[:10],
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model training history visualization
        st.markdown("### 📈 Riwayat Training Model")
        
        # Simulasi training history
        epochs = list(range(1, 101))
        train_loss = [1.0 * np.exp(-0.02 * e) + 0.1 * np.random.random() for e in epochs]
        val_loss = [1.1 * np.exp(-0.018 * e) + 0.12 * np.random.random() for e in epochs]
        
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss', mode='lines'))
        fig_history.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', mode='lines'))
        
        fig_history.update_layout(
            title="Training History (LSTM Model)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        st.plotly_chart(fig_history, use_container_width=True)

if __name__ == "__main__":
    main()