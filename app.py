import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Fungsi Utama
def main():
    st.set_page_config(
        page_title="Klasifikasi Penyakit Gagal Ginjal Kronis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
            .main { background-color: #f0f2f6; }
            h1, h2, h3 { color: #4CAF50; }
            .stButton>button { background-color: #4CAF50; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Klasifikasi Penyakit Gagal Ginjal Kronis")

    # Global variables for data sharing
    global new_df, Imp_features, X, y, model
    new_df, Imp_features, X, y, model = None, None, None, None, None

    try:
        data = pd.read_csv('data.csv')
        st.success("Dataset berhasil dimuat!")
    except FileNotFoundError:
        st.error("File 'data.csv' tidak ditemukan. Harap unggah dataset terlebih dahulu.")
        return

    # Sidebar Navigation
    page = st.sidebar.selectbox("Navigasi", ["Data", "Preprocessing", "Modeling", "Evaluasi"])

    # Halaman Data
    if page == "Data":
        st.header("Data Gagal Ginjal Kronis")

        st.subheader("Data Asli")
        st.dataframe(data, height=400)

        st.subheader("Dimensi Data")
        st.write(f"Jumlah Baris dan Kolom: {data.shape}")

        st.subheader("Informasi Data")
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    elif page == "Preprocessing":
        st.header("Penanganan Missing Value")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Data Asli")
            st.dataframe(data, width=1000, height=400)

        with col2:
            st.subheader("Total Missing Value per Kolom")
            missing_values = data.isnull().sum()
            st.dataframe(missing_values.rename("Jumlah Missing Value"))

        missing_proportions = data.isnull().mean()
        high_missing_cols = missing_proportions[missing_proportions > 0.2]
        low_missing_cols = missing_proportions[(missing_proportions > 0) & (missing_proportions <= 0.2)]

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Kolom dengan Missing Value Tinggi (>20%)")
            st.write(high_missing_cols if not high_missing_cols.empty else "Tidak ada.")

        with col4:
            st.subheader("Kolom dengan Missing Value Rendah (<=20%)")
            st.write(low_missing_cols if not low_missing_cols.empty else "Tidak ada.")
        
        # Menampilkan tipe data setiap kolom
        st.write("Tipe data setiap kolom:")
        st.write(data.dtypes)

        # Menampilkan data sebelum preprocessing
        st.write("Data sebelum preprocessing:")
        st.dataframe(data.head())

        # Menghapus kolom 'rbc' jika ada
        if 'rbc' in data.columns:
            data = data.drop('rbc', axis=1)

        imputer_mean = SimpleImputer(strategy='mean')
        imputer_mode = SimpleImputer(strategy='most_frequent')

        # Memperbaiki imputasi berdasarkan tipe data
        for col in high_missing_cols.index:
            if col in data.columns:
                if data[col].dtype in ['float64', 'int64']:  # Periksa tipe data numerik
                    sample_values = data[col].dropna().sample(data[col].isnull().sum(), replace=True).values
                    data.loc[data[col].isnull(), col] = sample_values
                else:
                    st.warning(f"Kolom {col} tidak numerik, dan tidak dapat diisi dengan imputasi numerik.")

        for col in low_missing_cols.index:
            if col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    data[col] = imputer_mean.fit_transform(data[[col]]).flatten()
                else:
                    data[col] = imputer_mode.fit_transform(data[[col]]).flatten()
                    
        # Pastikan untuk menangani kategori dengan benar
        for col in data.select_dtypes(include=['object']).columns:
            if col in ['wc', 'rc']:  # Tambahkan kolom ini pada pengecekan
                data[col] = imputer_mode.fit_transform(data[[col]]).flatten()  # Gunakan imputasi modus
            else:
                data[col] = data[col].astype(str)  # Ubah kolom kategori menjadi string

        # One-hot encoding untuk kolom kategorikal
        categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
        new_df = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        # Mengonversi kolom kategorikal menjadi numerik menggunakan LabelEncoder jika diperlukan
        label_encoder = LabelEncoder()
        binary_columns = ['htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
        for col in binary_columns:
            new_df[col] = label_encoder.fit_transform(new_df[col])

        # Pastikan data tidak memiliki nilai NaN
        numerical_columns = new_df.select_dtypes(include=['float64']).columns
        new_df[numerical_columns] = new_df[numerical_columns].fillna(0)

        st.subheader("Data Setelah One-Hot Encoding dan Penanganan Missing Value")
        st.dataframe(new_df)

        corr_matrix = new_df.corr()
        Dependent_corr = corr_matrix.get('classification', pd.Series())
        Imp_features = Dependent_corr[Dependent_corr.abs() > 0.4].index.tolist()
        if 'id' in Imp_features:
            Imp_features.remove('id')

        st.subheader("Fitur yang Dipilih Berdasarkan Korelasi")
        st.write(Imp_features if Imp_features else "Tidak ada fitur yang memenuhi syarat korelasi.")

        # Pastikan new_df dan Imp_features tidak None atau kosong
        if new_df is not None and Imp_features:
            # Pastikan kolom target 'classification' ada di dalam new_df
            if 'classification' in new_df.columns:
                X = new_df[Imp_features]  # Mengambil fitur penting berdasarkan korelasi
                y = new_df['classification']  # Kolom target
                
                # Tampilkan sampel data untuk validasi
                st.write("Contoh Data Fitur (X):")
                st.dataframe(X.head())

                st.write("Contoh Data Target (y):")
                st.dataframe(y.head())
            else:
                st.error("Kolom 'classification' tidak ditemukan dalam dataset.")
        else:
            st.error("Data preprocessing belum selesai atau tidak ada fitur penting yang terdeteksi.")

    # Halaman Modeling
    elif page == "Modeling":
        st.header("Modeling")

        if X is None or y is None:
            st.error("Harap lakukan preprocessing terlebih dahulu!")
            return

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Pilih Model
        model_type = st.selectbox("Pilih Model", ["Naive Bayes", "Decision Tree"])

        if model_type == "Naive Bayes":
            model = GaussianNB()
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()

        # Latih Model
        model.fit(X_train, y_train)
        st.success(f"Model {model_type} berhasil dilatih!")

        # Simpan Model untuk Evaluasi
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

    # Halaman Evaluasi
    elif page == "Evaluasi":
        st.header("Evaluasi Model")

        if 'model' not in st.session_state:
            st.error("Harap lakukan modeling terlebih dahulu!")
            return

        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("Hasil Evaluasi")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Akurasi:** {accuracy:.2f}")

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig)

# Jalankan aplikasi
if __name__ == "__main__":
    main()
