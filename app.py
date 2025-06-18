import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(
    page_title="Segmentasi Risiko Siber",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🛡️ Segmentasi Risiko Keamanan Siber")
st.markdown("Analisis aktivitas login untuk mendeteksi risiko menggunakan K-Means Clustering.")

# --- Memuat Model dan Preprocessor (dengan Caching) ---
# @st.cache_resource memastikan model hanya dimuat sekali
@st.cache_resource
def load_resources():
    """Memuat semua file model yang diperlukan."""
    model_path = 'model/'
    try:
        kmeans_model = joblib.load(model_path + 'kmeans.joblib')
        scaler = joblib.load(model_path + 'standard_scaler.joblib')
        label_encoders = joblib.load(model_path + 'label_encoders.joblib')
        feature_names = joblib.load(model_path + 'feature_names_for_model.joblib')
        categorical_cols = joblib.load(model_path + 'categorical_cols.joblib')
    except FileNotFoundError as e:
        st.error(f"Error memuat file model: {e}. Pastikan folder 'model' ada di direktori yang sama dengan app_streamlit.py dan berisi semua file .joblib.")
        return None, None, None, None, None
    return kmeans_model, scaler, label_encoders, feature_names, categorical_cols

kmeans_model, scaler, label_encoders, feature_names_for_model, categorical_cols = load_resources()

# Definisikan kolom numerik yang diharapkan scaler (dari log error sebelumnya)
numerical_cols_for_this_scaler = [
    'network_packet_size', 'login_attempts', 'session_duration',
    'ip_reputation_score', 'failed_logins', 'unusual_time_access'
]

# --- Label untuk Cluster ---
cluster_labels_map = {
    0: "Risiko Rendah (Aktivitas Normal)",
    1: "Risiko Sedang-Tinggi (Anomali Terdeteksi)",
    2: "Risiko Sedang (Penyelidikan Disarankan)",
    3: "Risiko Tinggi (Akses Tidak Biasa/Berbahaya)"
}

# --- Fungsi Preprocessing (Diadaptasi dari app.py Flask) ---
def preprocess_input_single_row(data_dict):
    """Fungsi ini mengambil satu baris data sebagai dict dan memprosesnya untuk prediksi."""
    processed_values = {}

    for col in feature_names_for_model:
        value_from_input = data_dict.get(col, None)

        if col in categorical_cols:
            le = label_encoders.get(col)
            if not le:
                raise ValueError(f"LabelEncoder untuk kolom '{col}' tidak ditemukan.")
            
            value_for_le_transform = None
            is_missing = pd.isna(value_from_input) or (isinstance(value_from_input, str) and value_from_input.lower() in ['tidak ada', 'none', '', 'nan'])

            if is_missing:
                value_for_le_transform = np.nan
            else:
                value_for_le_transform = str(value_from_input)
            
            try:
                processed_values[col] = le.transform([value_for_le_transform])[0]
            except ValueError:
                processed_values[col] = np.nan # Jika error, isi dengan NaN untuk diimputasi nanti

        else: # Kolom numerik
            num_val = pd.to_numeric(value_from_input, errors='coerce')
            processed_values[col] = num_val if not pd.isna(num_val) else 0.0 # Imputasi NaN dengan 0

    df_for_kmeans = pd.DataFrame([processed_values], columns=feature_names_for_model)

    # Paksa tipe data
    for col_name in feature_names_for_model:
        if col_name in categorical_cols:
            df_for_kmeans[col_name].fillna(0, inplace=True)
            df_for_kmeans[col_name] = df_for_kmeans[col_name].astype(int)
        else:
            df_for_kmeans[col_name].fillna(0.0, inplace=True)
            df_for_kmeans[col_name] = df_for_kmeans[col_name].astype(float)

    # Scaling hanya pada subset kolom numerik
    df_subset_to_scale = df_for_kmeans[numerical_cols_for_this_scaler].copy()
    try:
        scaled_values_subset = scaler.transform(df_subset_to_scale)
        for i, scaled_col_name in enumerate(numerical_cols_for_this_scaler):
            df_for_kmeans.loc[0, scaled_col_name] = scaled_values_subset[0, i]
    except Exception as e:
        raise ValueError(f"Error saat scaling data subset: {e}")
            
    return df_for_kmeans

# --- UI dengan Tabs ---
if kmeans_model: # Hanya tampilkan UI jika model berhasil dimuat
    tab1, tab2 = st.tabs(["Input Manual", "Upload File CSV"])

    # --- Tab 1: Input Manual ---
    with tab1:
        st.header("Prediksi Risiko Berdasarkan Input Manual")
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                network_packet_size = st.number_input("Ukuran Paket Jaringan", min_value=0, value=599, step=1)
                protocol_type = st.selectbox("Tipe Protokol", options=['TCP', 'UDP', 'ICMP'])
                login_attempts = st.number_input("Percobaan Login", min_value=0, value=4, step=1)

            with col2:
                session_duration = st.number_input("Durasi Sesi (detik)", min_value=0.0, value=492.98, format="%.2f")
                encryption_used = st.selectbox("Enkripsi Digunakan", options=['AES', 'DES', 'Tidak Ada'])
                ip_reputation_score = st.number_input("Skor Reputasi IP", min_value=0.0, max_value=1.0, value=0.60, format="%.3f")

            with col3:
                failed_logins = st.number_input("Login Gagal", min_value=0, value=1, step=1)
                browser_type = st.selectbox("Tipe Browser", options=['Chrome', 'Firefox', 'Safari', 'Edge', 'Unknown'])
                unusual_time_access = st.selectbox("Akses Waktu Tidak Biasa", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

            submitted = st.form_submit_button("Prediksi Risiko")

            if submitted:
                with st.spinner("Menganalisis data..."):
                    input_dict = {
                        'network_packet_size': network_packet_size,
                        'protocol_type': protocol_type,
                        'login_attempts': login_attempts,
                        'session_duration': session_duration,
                        'encryption_used': encryption_used,
                        'ip_reputation_score': ip_reputation_score,
                        'failed_logins': failed_logins,
                        'browser_type': browser_type,
                        'unusual_time_access': unusual_time_access
                    }
                    
                    try:
                        processed_df = preprocess_input_single_row(input_dict)
                        prediction = kmeans_model.predict(processed_df)
                        cluster_id = int(prediction[0])
                        cluster_label = cluster_labels_map.get(cluster_id, "Tidak diketahui")

                        st.success(f"**Prediksi Selesai!** Data masuk ke **Cluster {cluster_id}**.")
                        st.info(f"**Interpretasi Risiko:** {cluster_label}")
                        with st.expander("Lihat Detail Input yang Diproses"):
                            st.dataframe(processed_df)
                    except Exception as e:
                        st.error(f"Terjadi error saat prediksi: {e}")


    # --- Tab 2: Upload CSV ---
    with tab2:
        st.header("Analisis dari File CSV")
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

        if uploaded_file is not None:
            with st.spinner("Memproses file CSV..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Pratinjau Data dari CSV:**")
                    st.dataframe(df.head())

                    required_cols = set(feature_names_for_model)
                    if not required_cols.issubset(df.columns):
                        st.error(f"File CSV harus memiliki kolom berikut: {', '.join(required_cols)}")
                    else:
                        predictions = []
                        processed_rows = []
                        
                        for index, row in df.iterrows():
                            try:
                                processed_df_row = preprocess_input_single_row(row.to_dict())
                                pred = kmeans_model.predict(processed_df_row)
                                predictions.append(int(pred[0]))
                                processed_rows.append(row) # Simpan baris asli yang berhasil diproses
                            except Exception as e:
                                st.warning(f"Melewati baris {index+1} karena error: {e}")
                                predictions.append(None) # Tandai baris yang gagal
                        
                        df_results = pd.DataFrame(processed_rows).reset_index(drop=True)
                        df_results['cluster'] = [p for p in predictions if p is not None]

                        st.success(f"**Analisis Selesai!** Berhasil memproses {len(df_results)} dari {len(df)} baris.")

                        # --- Tampilan Metrik Evaluasi ---
                        if 'attack_detected' in df_results.columns:
                            st.subheader("Metrik Evaluasi (vs 'attack_detected')")
                            y_true = pd.to_numeric(df_results['attack_detected'], errors='coerce').fillna(0).astype(int)
                            y_pred_cluster = df_results['cluster']

                            # Petakan cluster ke label mayoritas
                            mapping_df = pd.DataFrame({'cluster': y_pred_cluster, 'attack_detected': y_true})
                            crosstab = pd.crosstab(mapping_df['cluster'], mapping_df['attack_detected'])
                            cluster_to_label_map = crosstab.idxmax(axis=1).to_dict()
                            y_pred_mapped = df_results['cluster'].map(cluster_to_label_map).fillna(0).astype(int)

                            # Hitung metrik
                            accuracy = accuracy_score(y_true, y_pred_mapped)
                            precision = precision_score(y_true, y_pred_mapped, zero_division=0)
                            recall = recall_score(y_true, y_pred_mapped, zero_division=0)
                            f1 = f1_score(y_true, y_pred_mapped, zero_division=0)

                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.metric("Akurasi", f"{accuracy:.2%}")
                            col_m2.metric("Presisi", f"{precision:.2%}")
                            col_m3.metric("Recall", f"{recall:.2%}")
                            col_m4.metric("F1-Score", f"{f1:.2%}")
                        else:
                            st.info("Kolom 'attack_detected' tidak ditemukan di CSV, metrik evaluasi tidak dapat dihitung.")

                        # --- Visualisasi Distribusi Cluster ---
                        st.subheader("Distribusi Data per Cluster")
                        cluster_counts = df_results['cluster'].value_counts().sort_index()
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        cluster_counts.plot(kind='bar', ax=ax, color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
                        
                        ax.set_title('Jumlah Data per Cluster Hasil Prediksi', fontsize=16)
                        ax.set_xlabel('ID Cluster', fontsize=12)
                        ax.set_ylabel('Jumlah Data', fontsize=12)
                        ax.set_xticklabels(cluster_counts.index, rotation=0)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        st.pyplot(fig)

                        # --- Opsi Unduh ---
                        st.subheader("Unduh Hasil Analisis")
                        
                        @st.cache_data
                        def convert_df_to_csv(df_to_convert):
                            return df_to_convert.to_csv(index=False).encode('utf-8')

                        csv_downloadable = convert_df_to_csv(df_results)
                        st.download_button(
                            label="Unduh CSV dengan Kolom Cluster",
                            data=csv_downloadable,
                            file_name="hasil_segmentasi_risiko.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Gagal memproses file: {e}")
