import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from io import BytesIO
import base64

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Beasiswa - Analisis Logistic Regression",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("🎓 Sistem Prediksi Beasiswa dengan Logistic Regression")
st.markdown("---")

# Sidebar untuk upload dataset
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV dataset", type=["csv"])
    
    st.header("⚙️ Konfigurasi Model")
    test_size = st.slider("Ukuran Data Testing (%)", 10, 40, 30)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    st.header("📊 Visualisasi")
    show_charts = st.checkbox("Tampilkan Visualisasi", value=True)
    
    if uploaded_file is not None:
        st.success(f"✅ File berhasil diunggah: {uploaded_file.name}")
    else:
        st.info("ℹ️ Gunakan dataset contoh atau unggah dataset Anda")
        use_sample = st.checkbox("Gunakan dataset contoh", value=True)

# Fungsi untuk load dataset
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Fungsi untuk preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # Encoding variabel kategorikal
    label_encoders = {}
    categorical_cols = ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    
    return df_processed, label_encoders

# Fungsi untuk training model
def train_model(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Fungsi untuk menampilkan matriks konfusi
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prediksi Tidak', 'Prediksi Diterima'],
                yticklabels=['Aktual Tidak', 'Aktual Diterima'],
                cbar_kws={'label': 'Jumlah'}, ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_ylabel('Aktual')
    ax.set_xlabel('Prediksi')
    return fig

# Fungsi untuk plot ROC Curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr_roc, tpr_roc)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.fill_between(fpr_roc, tpr_roc, alpha=0.3, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig, roc_auc

# Fungsi untuk plot feature importance
def plot_feature_importance(model, feature_names):
    coefficients = pd.DataFrame({
        'Fitur': feature_names,
        'Koefisien': model.coef_[0],
        'Pengaruh': ['Positif' if c > 0 else 'Negatif' for c in model.coef_[0]],
        'Magnitude': [abs(c) for c in model.coef_[0]]
    }).sort_values('Koefisien', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_coef = ['#e74c3c' if c < 0 else '#2ecc71' for c in coefficients['Koefisien']]
    bars_coef = ax.barh(range(len(coefficients)), coefficients['Koefisien'], 
                       color=colors_coef, edgecolor='black')
    ax.set_yticks(range(len(coefficients)))
    ax.set_yticklabels(coefficients['Fitur'])
    ax.set_xlabel('Koefisien Logistic Regression')
    ax.set_title('Pengaruh Fitur Terhadap Penerimaan', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Tambah nilai koefisien di bar
    for i, (bar, coef) in enumerate(zip(bars_coef, coefficients['Koefisien'])):
        x_pos = bar.get_width()
        align = 'left' if coef >= 0 else 'right'
        offset = 0.01 if coef >= 0 else -0.01
        ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f'{coef:.3f}', ha=align, va='center', fontsize=8)
    
    return fig, coefficients

# Fungsi untuk prediksi data baru
def predict_new_data(model, scaler, label_encoders, X_columns, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    categorical_cols = ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']
    for col in categorical_cols:
        if col in input_df.columns and col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])
            except:
                # If value not in encoder, use most frequent
                input_df[col] = 0
    
    # Ensure correct column order
    for col in X_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[X_columns]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

# Main application logic
def main():
    # Load dataset
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        # Use sample data path or show message
        try:
            df = pd.read_csv('dataset_beasiswa_2020_2024_1000.csv')
        except:
            st.warning("Silakan unggah dataset Anda atau pastikan file 'dataset_beasiswa_2020_2024_1000.csv' tersedia")
            st.stop()
    
    # Display dataset info
    with st.expander("📊 Informasi Dataset", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data", f"{df.shape[0]:,}")
        with col2:
            st.metric("Jumlah Fitur", df.shape[1])
        with col3:
            st.metric("Kolom Target", "Diterima_Beasiswa")
        
        col4, col5 = st.columns(2)
        with col4:
            st.subheader("5 Data Pertama")
            st.dataframe(df.head(), use_container_width=True)
        with col5:
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)
    
    # Preprocessing
    with st.spinner("🔄 Memproses data..."):
        df_processed, label_encoders = preprocess_data(df)
        
        # Pisahkan fitur dan target
        X = df_processed.drop(['PK', 'Diterima_Beasiswa', 'Tahun_Pendaftaran'], axis=1, errors='ignore')
        y = df_processed['Diterima_Beasiswa']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        # Standardisasi
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Training model
    with st.spinner("🤖 Melatih model Logistic Regression..."):
        model = train_model(X_train_scaled, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics calculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Display metrics
    st.markdown("---")
    st.header("📈 Evaluasi Model")
    
    # Metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}", delta=None)
    with col2:
        st.metric("Precision", f"{precision:.2%}", delta=None)
    with col3:
        st.metric("Recall", f"{recall:.2%}", delta=None)
    with col4:
        st.metric("F1-Score", f"{f1:.2%}", delta=None)
    with col5:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.metric("ROC-AUC", f"{roc_auc:.2%}", delta=None)
    
    # Confusion Matrix details
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(cm)
        st.pyplot(fig_cm)
        
        # Detailed confusion matrix
        st.info(f"""
        **Detail Confusion Matrix:**
        - **True Positives (TP):** {tp} - Prediksi benar sebagai Diterima
        - **False Positives (FP):** {fp} - Prediksi salah sebagai Diterima
        - **False Negatives (FN):** {fn} - Prediksi salah sebagai Tidak Diterima
        - **True Negatives (TN):** {tn} - Prediksi benar sebagai Tidak Diterima
        """)
    
    with col2:
        st.subheader("ROC Curve")
        fig_roc, roc_auc_value = plot_roc_curve(y_test, y_pred_proba)
        st.pyplot(fig_roc)
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Tidak Diterima', 'Diterima'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.header("🔍 Feature Importance")
    
    fig_importance, coefficients = plot_feature_importance(model, X.columns)
    st.pyplot(fig_importance)
    
    # Display top features
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("5 Fitur dengan Pengaruh Positif Terbesar")
        top_positive = coefficients[coefficients['Koefisien'] > 0].nlargest(5, 'Koefisien')
        st.dataframe(top_positive[['Fitur', 'Koefisien']].style.format({'Koefisien': '{:.4f}'}), 
                    use_container_width=True)
    
    with col2:
        st.subheader("5 Fitur dengan Pengaruh Negatif Terbesar")
        top_negative = coefficients[coefficients['Koefisien'] < 0].nsmallest(5, 'Koefisien')
        st.dataframe(top_negative[['Fitur', 'Koefisien']].style.format({'Koefisien': '{:.4f}'}), 
                    use_container_width=True)
    
    # Data distribution visualizations
    if show_charts:
        st.markdown("---")
        st.header("📊 Analisis Distribusi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pendapatan Orang Tua vs Penerimaan")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            accepted = df[df['Diterima_Beasiswa'] == 1]['Pendapatan_Orang_Tua']
            rejected = df[df['Diterima_Beasiswa'] == 0]['Pendapatan_Orang_Tua']
            ax1.hist([accepted, rejected], bins=15, label=['Diterima', 'Tidak Diterima'], 
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Pendapatan Orang Tua')
            ax1.set_ylabel('Frekuensi')
            ax1.legend()
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Prestasi Akademik vs Penerimaan")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            data_to_plot = [df[df['Diterima_Beasiswa'] == 0]['Prestasi_Akademik'],
                          df[df['Diterima_Beasiswa'] == 1]['Prestasi_Akademik']]
            box = ax2.boxplot(data_to_plot, patch_artist=True, 
                            labels=['Tidak Diterima', 'Diterima'])
            ax2.set_ylabel('Prestasi Akademik')
            ax2.grid(axis='y', alpha=0.3)
            
            colors_box = ['#e74c3c', '#2ecc71']
            for patch, color in zip(box['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            st.pyplot(fig2)
    
    # Prediction Interface
    st.markdown("---")
    st.header("🔮 Prediksi Data Baru")
    
    with st.form("prediction_form"):
        st.subheader("Masukkan Data Calon Penerima Beasiswa")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pendapatan = st.number_input("Pendapatan Orang Tua (juta)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            asal_sekolah = st.selectbox("Asal Sekolah", ["Negeri", "Swasta"])
            lokasi = st.selectbox("Lokasi Domisili", ["Kota", "Desa"])
        
        with col2:
            organisasi = st.number_input("Keikutsertaan Organisasi", min_value=0, max_value=10, value=2, step=1)
            pengalaman_sosial = st.number_input("Pengalaman Sosial (jam)", min_value=0, max_value=1000, value=300, step=50)
            gender = st.selectbox("Gender", ["L", "P"])
        
        with col3:
            disabilitas = st.selectbox("Status Disabilitas", ["Tidak", "Ya"])
            prestasi_akademik = st.number_input("Prestasi Akademik (skala 1-10)", min_value=1, max_value=10, value=6, step=1)
            prestasi_non_akademik = st.number_input("Prestasi Non-Akademik (skala 1-10)", min_value=0, max_value=10, value=2, step=1)
        
        submitted = st.form_submit_button("🚀 Prediksi Sekarang!")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Pendapatan_Orang_Tua': pendapatan,
                'Asal_Sekolah': asal_sekolah,
                'Lokasi_Domisili': lokasi,
                'Keikutsertaan_Organisasi': organisasi,
                'Pengalaman_Sosial': pengalaman_sosial,
                'Gender': gender,
                'Status_Disabilitas': disabilitas,
                'Prestasi_Akademik': prestasi_akademik,
                'Prestasi_Non_Akademik': prestasi_non_akademik
            }
            
            # Make prediction
            prediction, probability = predict_new_data(model, scaler, label_encoders, X.columns, input_data)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Hasil Prediksi")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.success(f"### ✅ DIPREDIKSI DITERIMA")
                else:
                    st.error(f"### ❌ DIPREDIKSI TIDAK DITERIMA")
                
                # Probability gauge
                prob_diterima = probability[1] * 100
                st.metric("Probabilitas Diterima", f"{prob_diterima:.1f}%")
                
                # Progress bar
                st.progress(int(prob_diterima))
            
            with col_result2:
                st.info(f"""
                **Detail Probabilitas:**
                - Probabilitas Diterima: **{probability[1]:.2%}**
                - Probabilitas Tidak Diterima: **{probability[0]:.2%}**
                
                **Interpretasi:**
                {f"**Tinggi** - Peluang diterima sangat besar" if probability[1] >= 0.7 else 
                  f"**Sedang** - Peluang diterima cukup baik" if probability[1] >= 0.5 else 
                  f"**Rendah** - Perlu perbaikan beberapa aspek" if probability[1] >= 0.3 else 
                  f"**Sangat Rendah** - Perlu perbaikan signifikan"}
                """)
    
    # Model summary and download
    st.markdown("---")
    st.header("📄 Ringkasan Model")
    
    summary_text = f"""
    SUMMARY REPORT - Logistic Regression for Scholarship Prediction
    {'='*60}
    Dataset: {df.shape[0]} records, {df.shape[1]} features
    Train-Test Split: {X_train.shape[0]}-{X_test.shape[0]} ({100-test_size}%-{test_size}%)
    
    PERFORMANCE METRICS:
    Accuracy:  {accuracy:.4f}
    Precision: {precision:.4f}
    Recall:    {recall:.4f}
    F1-Score:  {f1:.4f}
    ROC-AUC:   {roc_auc:.4f}
    
    CONFUSION MATRIX:
    True Negatives:  {tn}
    False Positives: {fp}
    False Negatives: {fn}
    True Positives:  {tp}
    
    TOP 5 FEATURES BY IMPORTANCE:
    {coefficients.nlargest(5, 'Magnitude')[['Fitur', 'Koefisien', 'Pengaruh']].to_string(index=False)}
    """
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Ringkasan Model", summary_text, height=300)
    
    with col2:
        st.subheader("📥 Download Hasil")
        
        # Create download link for summary
        b64 = base64.b64encode(summary_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="model_summary.txt">📄 Download Ringkasan Model</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Create download link for predictions
        predictions_df = pd.DataFrame({
            'Aktual': y_test.values,
            'Prediksi': y_pred,
            'Prob_Diterima': y_pred_proba
        })
        csv = predictions_df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()
        href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="prediksi_beasiswa.csv">📊 Download Hasil Prediksi</a>'
        st.markdown(href_csv, unsafe_allow_html=True)
        
        st.info("""
        **Rekomendasi Berdasarkan Model:**
        1. Fokus pada **Prestasi Akademik**
        2. Tingkatkan **pengalaman organisasi**
        3. Kumpulkan **prestasi non-akademik**
        4. **Pendapatan rendah** meningkatkan peluang
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Dibuat dengan ❤️ menggunakan Streamlit | Logistic Regression Model</p>
        <p>© 2024 Sistem Prediksi Beasiswa</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()