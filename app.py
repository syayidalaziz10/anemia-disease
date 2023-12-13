import streamlit as st
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
    st.image("assets/get-started.png")

    st.header("Selamat Datang 👋")
    st.write(
        "Aplikasi deteksi dini Anemia pada Ibu Hamil ini merupakan salah satu contoh implementasi data mining pada bidang kesehatan. Dengan menggunakan data yang ada sistem ini akan dapat melakukan prediksi pada data berikutnya yang sesuai dengan penyakit anemia, sehingga dapat melakukan prediksi yang tepat setelah sistem melakukan analisis pada data baru tersebut. Hasil akhir pada aplikasi ini akan memberikan deteksi dini pada ibu hamil tersebut sedang mengalami anemia atau tidak.")
    col1, col2 = st.columns([3, 11])

    with col1:
        if st.button("Dokumentasi 📖"):
            switch_page("documentation")

    with col2:
        if st.button("Mulai prediksi :point_right:", key="start"):
            switch_page("prediction")


if __name__ == '__main__':
    main()
