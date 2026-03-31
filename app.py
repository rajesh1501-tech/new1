from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_together import Together
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import time

st.set_page_config(page_title="AttroneyGPT")
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.image("logo.png")

st.markdown(
    """
    <style>
    div[data-baseweb="input"] input {
            border-color: #000000;
        }
    margin-top: 0 !important;
div.stButton > button:first-child {
    background-color: #808080;
    color:white;
}
div.stButton > button:active {
    background-color: #808080;
    color : white;
}

   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)


def reset_conversation():
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
)
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

TOGETHER_AI_API = os.environ["TOGETHER_AI"]
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_AI_API,
)

# Prompt to condense question + history into a standalone question
condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
               "formulate a standalone question which can be understood without the chat history. "
               "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Main QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, "
               "your primary objective is to provide accurate and concise information based on the human's questions. "
               "Do not generate your own questions and answers. You will adhere strictly to the instructions provided, "
               "offering relevant context from the knowledge base while avoiding unnecessary details. "
               "Your responses will be brief, to the point, and in compliance with the established format. "
               "If a question falls outside the given context, you will refrain from utilizing the chat history and "
               "instead rely on your own knowledge base to generate an appropriate response. "
               "You will prioritize the human's query and refrain from posing additional questions. "
               "The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.\n\n"
               "CONTEXT: {context}[/INST]"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_contextualized_question(input_dict: dict) -> str:
    """Reformulate the question using chat history when available."""
    if input_dict.get("chat_history"):
        condense_chain = condense_question_prompt | llm | StrOutputParser()
        return condense_chain.invoke(input_dict)
    return input_dict["input"]


# LCEL RAG chain (replaces create_history_aware_retriever + create_retrieval_chain)
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(db_retriever.invoke(get_contextualized_question(x)))
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_history


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

for message in st.session_state.messages:
    role = message.get("role")
    content = message.get("content")
    with st.chat_message(role, avatar="user.svg" if role == "human" else "attorney.svg"):
        st.write(content)

input_prompt = st.chat_input("message LAWGpt.....")

if input_prompt:
    with st.chat_message("human", avatar="user.svg"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "human", "content": input_prompt})

    with st.chat_message("bot", avatar="attorney.svg"):
        with st.spinner("Thinking..."):
            answer = conversational_rag_chain.invoke(
                {"input": input_prompt},
                config={"configurable": {"session_id": "lawgpt_session"}},
            )

        message_placeholder = st.empty()
        full_response = "⚠️ **_Note: This offers basic legal advice and is not a complete substitute for consulting a human attorney_** \n\n\n"

        for chunk in answer:
            full_response += chunk
            time.sleep(0.02)
            message_placeholder.markdown(full_response + " ▌")

        message_placeholder.markdown(full_response)
        st.button("Reset All Chat 🗑️", on_click=reset_conversation)

    st.session_state.messages.append({"role": "ai", "content": answer, "avatar": "attorney.svg"})
