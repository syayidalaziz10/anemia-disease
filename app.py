import streamlit as st


def show():

    st.image("assets/get-started.png")

    st.header("Home Page")
    st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")

    # tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

    # with tab1:
    #     st.header("A cat")
    #     st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

    # with tab2:
    #     st.header("A dog")
    #     st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    # with tab3:
    #     st.header("An owl")
    #     st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    # st.button("Documentation", key="documentation")

    # Handle the button click event for "Documentation"
    # Redirect to the "About" page
    # about_page()

    col1, col2 = st.columns([2, 8])

    with col1:
        st.button("Documentation")

    with col2:
        st.button("Mulai prediksi :point_right:", key="start")


if __name__ == '__main__':
    show()
