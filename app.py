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

    st.header("Home Page")
    st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
    col1, col2 = st.columns([3, 11])

    with col1:
        if st.button("Dokumentasi 📖"):
            switch_page("documentation")

    with col2:
        if st.button("Mulai prediksi :point_right:", key="start"):
            switch_page("prediction")


if __name__ == '__main__':
    main()
