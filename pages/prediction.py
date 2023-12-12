import streamlit as st
import pandas as pd
import time
import joblib
from sklearn.metrics import DistanceMetric
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

    st.image("assets/get-started.png")
    st.header("Prediksi Anemia pada Ibu Hamil")
    st.write("Masukkan beberapa isian yang digunakan untuk melakukan deteksi dini pada penyakit anemia pada ibu hamil. Pastikan isian yang anda masukkan benar!")

    col1, col2 = st.columns(2)
    with col1:
        umur = st.text_input('Umur')
        col3, col4 = st.columns(2)
        with col3:
            mm = st.text_input('Tekanan Darah (mm)')
        with col4:
            g = st.text_input('Tekanan Darah (g)')

    with col2:
        hb = st.text_input('Hemoglobin')
        nadi = st.text_input('Nadi')

    if st.button("Prediksi Sekarang 👉"):
        if umur and mm and g and hb and nadi:
            try:
                umur = int(umur)
                mm = int(mm)
                g = int(g)
                hb = float(hb)
                nadi = int(nadi)
            except ValueError:
                time.sleep(0.5)
                st.toast('Teks harus berisikan angka', icon='🤧')

            td = mm/g
            result = get_predict(umur, hb, td, nadi)

            if result == 0:
                st.success("Non Anemia")
            else:
                st.error("Anemia")

        else:
            time.sleep(.5)
            st.toast('Masukkan teks terlebih dahulu', icon='🤧')


def get_predict(umur, hb, td, nadi):

    df = pd.DataFrame({
        'Umur': [umur],
        'HB': [hb],
        'TD': [td],
        'Nadi': [nadi],
    })

    model = joblib.load('model/knn-model')
    scaler = joblib.load('model/minmax-model')

    minmax = scaler.transform(df[['Umur', 'HB', 'TD', 'Nadi']])
    prediction = model.predict(minmax)

    return prediction


if __name__ == '__main__':
    main()
