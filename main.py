import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

CAMINHO_DB = "db"

prompt_template = """
Responda a pergunta do usuário:
{question}

com base nessas informações abaixo:

{knowledge_base}

Formate sua resposta em Markdown sempre que possível.
"""

st.set_page_config(page_title="Chat RAG Simples", page_icon="💬")
st.title("💬 Agente FAQ do site da UFABC")

# Inicializa o histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# CSS para scroll horizontal (recomendação da documentação)
HSCROLL_STYLE = """
<style>
.scrollable-response {
    overflow-x: auto;
    padding-bottom: 1rem;
}
</style>
"""
st.markdown(HSCROLL_STYLE, unsafe_allow_html=True)

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(
            f"<div class='scrollable-response'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

# Campo de entrada estilo chat
question = st.chat_input("Digite sua pergunta...")

if question:
    # Adiciona ao histórico
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)

    # Carregar banco vetorial
    emb = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=emb)

    # Busca por similaridade
    results = db.similarity_search_with_relevance_scores(question, k=4)

    if len(results) == 0 or results[0][1] < 0.7:
        resposta = "Desculpe, não encontrei informações relevantes para sua pergunta."
    else:
        # Junta textos
        text_results = [r[0].page_content for r in results]
        knowledge_base = "\n\n----\n\n".join(text_results)

        # Prompt
        prompt = prompt_template.format(
            question=question,
            knowledge_base=knowledge_base
        )

        modelo = ChatOllama(model="gpt-oss:20b")
        resposta = modelo.invoke(prompt).content

    # Adiciona ao histórico
    st.session_state.messages.append({"role": "assistant", "content": resposta})

    # Exibe resposta com scroll horizontal
    with st.chat_message("assistant"):
        st.markdown(
            f"<div class='scrollable-response'>{resposta}</div>",
            unsafe_allow_html=True
        )
