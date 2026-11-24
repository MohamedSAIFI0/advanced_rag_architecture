import re
import streamlit as st
try:
    # Versions récentes de LangChain
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama
except ImportError:
    # Anciennes versions de LangChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import Ollama

# ---------------------- CONFIG STREAMLIT ----------------------
st.set_page_config(page_title="PDF-RAG (Student Version)", page_icon="📘")
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
st.title("Local PDF-RAG – Student Version")

# Mode de comparaison : RAG classique vs RAG amélioré
advanced_mode = st.sidebar.checkbox("RAG avancé (Query Expansion + Reranking)", value=True)

# ---------------------- SESSION STATE ----------------------
if "db" not in st.session_state:
    st.session_state.db = None
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(model="llama3.2")
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ---------------------- TODO 1 : QUERY EXPANSION ----------------------
def expand_query(query, llm):
    """
    Utilise le LLM pour générer 3 reformulations de la question.
    Retourne une liste de requêtes (question originale + reformulations).
    """
    prompt = f"""Generate 3 alternative questions for the question below.
Each alternative must keep the same meaning but use different words.
Write one question per line, with no numbering, no bullet points and no extra text.

Question: {query}

Alternative questions:"""

    try:
        response = llm(prompt)
    except Exception:
        return [query]

    reformulations = []
    for line in response.split("\n"):
        # Nettoyer numérotation / puces éventuelles (1. , 1) , - , * )
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", line).strip()
        if len(line) > 5 and line.endswith("?") and line.lower() != query.lower():
            reformulations.append(line)

    # Question originale + au maximum 3 reformulations
    return [query] + reformulations[:3]


# ---------------------- TODO 2 : RERANKING ----------------------
def rerank_chunks(query, docs, llm, top_k=5):
    """
    Score chaque chunk (0 à 10) selon sa pertinence par rapport à la question,
    puis retourne les top_k chunks les plus pertinents.
    """
    # Supprimer les doublons (un même chunk peut ressortir pour plusieurs requêtes)
    unique_docs = []
    seen = set()
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    scored_docs = []
    for doc in unique_docs:
        prompt = f"""You are a relevance judge.
Rate how relevant the passage is for answering the question, on a scale from 0 to 10.
0 = completely irrelevant, 10 = contains the exact answer.
Answer with ONLY a single number.

Question: {query}

Passage:
{doc.page_content}

Score:"""
        try:
            response = llm(prompt)
            match = re.search(r"\d+(?:\.\d+)?", response)
            score = float(match.group()) if match else 0.0
            score = min(max(score, 0.0), 10.0)
        except Exception:
            score = 0.0
        scored_docs.append((score, doc))

    # Tri décroissant par score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


# ---------------------- PDF PROCESSING ----------------------
if uploaded_file is not None and st.session_state.processed_file != uploaded_file.name:
    with st.spinner("Processing PDF..."):
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        st.session_state.db = Chroma.from_documents(texts, embeddings)
        st.session_state.processed_file = uploaded_file.name

        st.success("PDF processed successfully!")

# ---------------------- QUESTION / ANSWER ----------------------
query = st.chat_input("Ask a question about the PDF:")

if query and st.session_state.db:
    with st.spinner("Generating answer..."):

        retriever = st.session_state.db.as_retriever(search_kwargs={"k": 5})

        if advanced_mode:
            # ---------- TODO 3 : USE QUERY EXPANSION ----------
            expanded_queries = expand_query(query, st.session_state.llm)

            # ---------- MULTI-SEARCH ----------
            all_docs = []
            for q in expanded_queries:
                docs = retriever.get_relevant_documents(q)
                all_docs.extend(docs)

            # ---------- TODO 4 : RERANK DOCUMENTS ----------
            selected_docs = rerank_chunks(
                query,
                all_docs,
                st.session_state.llm,
                top_k=5
            )
        else:
            # RAG classique : une seule recherche, pas de reranking
            expanded_queries = [query]
            selected_docs = retriever.get_relevant_documents(query)

        # ---------- FINAL PROMPT ----------
        context = "\n\n".join([doc.page_content for doc in selected_docs])

        prompt = f"""
You must answer ONLY using the context below.
If the answer is not in the context, say: "I don't know based on the provided document."

Context:
{context}

Question:
{query}
"""

        answer = st.session_state.llm(prompt)

        st.write("### Answer")
        st.write(answer)

        if advanced_mode:
            st.write("### Reformulations utilisées")
            for q in expanded_queries:
                st.write(f"- {q}")

        st.write("### Sources")
        for doc in selected_docs:
            st.write(f"- {doc.metadata['source']} (page {doc.metadata.get('page', '?')})")