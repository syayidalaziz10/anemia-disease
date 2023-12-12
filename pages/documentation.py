import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from streamlit_extras.switch_page_button import switch_page


def main():

    st.set_page_config(
        page_title="Deteksi Penyakit Dini Anemia Pada Ibu Hamil",
        page_icon="👩‍👦",
        initial_sidebar_state="collapsed")

    st.markdown(
        """
        <style>
            [data-testid="collapsedControl"] {
                display: none
            }
        </style>
        """, unsafe_allow_html=True,
    )

    if st.button("👈 Kembali ke Halaman Utama"):
        switch_page("app")

    # data processing
    df = pd.read_csv("dataset/anemia.csv")
    reduce = df.drop(['Nama', 'Goldar', 'Pengobatan'], axis=1)
    data = df.drop(['Nama', 'Goldar', 'Pengobatan'], axis=1)

    scaler = MinMaxScaler()
    data[['Umur', 'HB', 'TD', 'Nadi']] = scaler.fit_transform(
        data[['Umur', 'HB', 'TD', 'Nadi']])

    X = data.drop('Keterangan', axis=1)
    y = data['Keterangan']

    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    imbalance = pd.DataFrame(X_resampled, columns=X.columns)
    imbalance['Keterangan'] = y_resampled

    before = data['Keterangan'].value_counts()
    after = imbalance['Keterangan'].value_counts()

    # st.button("Documentation", key="documentation")

    # Handle the button click event for "Documentation"
    # Redirect to the "About" page
    # about_page()

    # correlation_data = reduce.corr(method='pearson', numeric_only=False)

    # end data processing

    # interface section

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["(1) Data Understanding", "(2) Analisis Data", "(3) Preprocessing Data", "(4) KNN Classifier", "(5) SVM Classifier", "(6) Naive Bayes"])

    with tab1:
        st.subheader("1.1 Latar Belakang")
        st.write(
            "Anemia merupakan salah satu faktor yang dapat mempengaruhi kematian pada ibu hamil, terjadinya anemia ini disebabkan oleh penurunan hemoglobin, hematokrit dan eritrosit [1]. Tingkat penurunan hemoglobin ini memiliki kriteria yang berbeda-beda bergantung pada usia, pada wanita hamil akan terjadi anemia jika kadar hb <11g/dl, dimana kondisi ini akan mempengaruhi daya angkut oksigen yang digunakan untuk organ vital dan janin pada ibu menjadi berkurang [2]. Data Mining memiliki peran penting dalam bidang kesehatan khususnya untuk melakukan diagnosa penyakit sacara dini dengan cara mempelajari dari pola data penyakit yang ada [4]. Supervised learning merupakan suatu model matematis yang digunakan untuk melakukan prediksi sesuai dengan data yang telah dipelajari sebelumnya [4]. Penelitian ini akan melakukan klasifikasi pada data penyakit anemia pada ibu hamil, data ini diklasifikasikan menjadi 2 bagian yaitu anemia dan non anemia.")

    with tab2:
        st.subheader("2. Analisis Data")
        st.image("assets/diagram-ipo.png")

    with tab3:
        st.subheader("3.1 Dataset")
        st.write(df)

        st.subheader("3.2 Data Reduce")
        st.write(reduce)

        st.subheader("3.3 Data Normalization")
        st.dataframe(data, width=None, height=None)

        st.subheader("3.4 Balancing Data")
        st.write("Data sebelum di balancing: ", before)
        st.write("Data setelah di balancing: ", after)
    with tab4:
        st.subheader("3.5 Split Data")
        st.write("Data input X: ", X.head())
        st.write("Data input y: ", y.head())

    with tab5:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    with tab6:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    # st.subheader("2.5 Correlation Data")
    # st.write(
    #     "Berikut merupakan tabel correlasi data pada setiap fitur: ", correlation_data

    # end interface section


if __name__ == '__main__':
    main()
