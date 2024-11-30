import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(
        page_title="Klasifikasi Penyakit Gagal Ginjal Kronis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Klasifikasi Penyakit Gagal Ginjal Kronis")

    try:
        data = pd.read_csv('data.csv')
        st.success("Dataset berhasil dimuat!")
    except FileNotFoundError:
        st.error("File 'data.csv' tidak ditemukan. Harap unggah dataset terlebih dahulu.")
        return

    st.write("Kolom yang ada dalam dataset:", data.columns)

    # Penanganan Missing Value
    st.header("Penanganan Missing Value")

    imputer_mean = SimpleImputer(strategy='mean')
    imputer_mode = SimpleImputer(strategy='most_frequent')

    # Mengisi missing values untuk kolom numerik
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        data[col] = imputer_mean.fit_transform(data[[col]]).flatten()

    # Mengisi missing values untuk kolom kategorikal
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = imputer_mode.fit_transform(data[[col]]).flatten()

    # Menghapus kolom ID
    data = data.drop(columns=['id'])

    # Memisahkan fitur dan target
    X = data.drop(columns=['classification'])
    y = data['classification']

    # One-Hot Encoding untuk kolom kategorikal
    categorical_columns = X.select_dtypes(include=['object']).columns
    valid_categorical_columns = [col for col in categorical_columns if col in X.columns]
    X = pd.get_dummies(X, columns=valid_categorical_columns, drop_first=True)

    st.write(f"Data setelah One-Hot Encoding:\n{X.head()}")

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

    # Evaluasi Model
    st.subheader("Evaluasi Model")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Akurasi:** {accuracy:.2f}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(conf_matrix)

    # Visualisasi Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
