# third-party imports
import streamlit as st
from streamlit_chat import message

# project imports
from utils.file import list_models
from utils.file import ingest_files
from utils.llm import create_model
from utils.rag import create_prompt
from utils.rag import create_chain


def select_chat_model():
    with st.session_state['loading_model'], st.spinner('Loading model...'):

        if st.session_state['model_name'] \
                and st.session_state['model_temperature']:

            st.session_state['model'] = create_model(
                model=st.session_state['model_name'],
                temperature=st.session_state['model_temperature'])


def process_uploaded_files():
    st.session_state['messages'] = []
    st.session_state['user_input'] = ''

    with st.session_state['loading_retriever'], st.spinner('Loading documents...'):
        st.session_state['retriever'] = ingest_files(st.session_state['file_uploader'])


def process_user_prompt():
    if st.session_state['user_input'] and \
            len(st.session_state['user_input'].strip()) > 0:

        user_text = st.session_state['user_input'].strip()

        with st.session_state['loading_response'], st.spinner('Loading response...'):
            agent_text = st.session_state['chain'].invoke(user_text)

        st.session_state['messages'].append((user_text, True))
        st.session_state['messages'].append((agent_text, False))

        st.session_state['user_input'] = ''


def main():

    if len(st.session_state) == 0:
        st.session_state['messages'] = []
        st.session_state['model'] = None
        st.session_state['retriever'] = None

    st.sidebar.subheader('Configuration')

    st.sidebar.radio(
        'Model',
        index=None,
        key='model_name',
        options=list_models(),
        on_change=select_chat_model)

    st.sidebar.selectbox(
        'Temperature',
        index=None,
        key='model_temperature',
        on_change=select_chat_model,
        options=[.7, .75, .8, .85, 0.9, 0.95, 1.0],
        placeholder='Select the model temperature...')

    st.sidebar.file_uploader(
        'Upload File',
        type=['pdf', 'csv'],
        key='file_uploader',
        accept_multiple_files=True,
        on_change=process_uploaded_files)

    st.header('Chat with your local files!')
    st.caption('A RAG application using Langchain, Llama 2 and streamlit.')

    st.session_state['loading_model'] = st.empty()
    st.session_state['loading_retriever'] = st.empty()

    if st.session_state['model'] and st.session_state['retriever']:

        st.session_state['chain'] = create_chain(
            prompt=create_prompt(),
            model=st.session_state['model'],
            retriever=st.session_state['retriever'])

        for i, (msg, is_user) in enumerate(st.session_state['messages']):
            message(msg, is_user=is_user, key=str(i))

        st.session_state['loading_response'] = st.empty()

        st.text_input(
            label='Chat',
            key='user_input',
            on_change=process_user_prompt,
            placeholder='Ask a question to your files')


if __name__ == '__main__':
    main()
