
# ============================================================
# PROJET : ENSET-ChatBot
# OBJECTIF : Mise en place d'un système RAG
# Technologies : Streamlit, LangChain, Ollama, Chroma, HuggingFace
# ============================================================


# ============================================================
# 1. IMPORTATION DES BIBLIOTHEQUES
# ============================================================

import streamlit as st
import os
import tempfile
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_ollama import ChatOllama

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 2. CONFIGURATION DE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="ENSET-ChatBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ENSET-ChatBot")
st.subheader("Your intelligent PDF assistant powered by RAG")

st.sidebar.title("📄 Document Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# ============================================================
# 3. INITIALISATION DE LA SESSION
# ============================================================

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None


# ============================================================
# 4. CHARGEMENT DU DOCUMENT PDF
# ============================================================

def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


# ============================================================
# 5. DECOUPAGE DU DOCUMENT (CHUNKING)
# ============================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# ============================================================
# 6. GENERATION DES EMBEDDINGS
# ============================================================

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


# ============================================================
# 7. CREATION DE LA BASE VECTORIELLE CHROMA
# ============================================================

def create_vector_database(chunks, embeddings):

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="enset_chatbot"
    )

    return db


# ============================================================
# 8. CREATION DE LA CHAINE RAG
# ============================================================

def create_qa_chain(db):

    # Initialisation du modele LLM
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.2
    )

    # Creation du retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Définition du prompt
    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            Tu es ENSET-ChatBot, un assistant pédagogique intelligent.

            Ta mission est de répondre aux questions des étudiants
            en te basant exclusivement sur les informations présentes
            dans le contexte fourni par la base documentaire.

            RÈGLES STRICTES :

            1. Source unique :
            Utilise uniquement les informations du CONTEXTE.
            N'utilise jamais tes connaissances générales.

            2. Interdiction d'inventer :
            Ne génère aucune information qui n'est pas explicitement
            présente ou directement déductible du contexte.

            3. Gestion des informations absentes :
            Si le contexte ne contient pas la réponse à la question,
            réponds exactement :
            "Désolé, je ne trouve pas cette information
            dans la base documentaire."

            4. Réponses partielles :
            Si le contexte contient seulement une partie de la réponse,
            réponds uniquement avec les informations disponibles.
            Précise que les autres informations ne sont pas disponibles
            dans la base documentaire.

            5. Questions hors sujet :
            Si la question ne concerne pas les informations disponibles
            dans la base documentaire, indique que tu ne peux pas
            répondre à cette question.

            6. Clarté pédagogique :
            Formule des réponses claires, précises et faciles à comprendre.

            7. Respect du contexte :
            Ne considère jamais une supposition comme un fait.
            En cas de doute, préfère ne pas répondre.

            CONTEXTE :
            {context}
            """
        ),

        ("human", "{input}")

    ])

    # Combinaison des documents avec le LLM
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Creation de la chaine de retrieval
    qa_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return qa_chain


# ============================================================
# 9. TRAITEMENT DU PDF
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    # Identification du fichier
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Eviter de retraiter le meme document
    if file_hash != st.session_state.processed_file_hash:

        try:

            with st.spinner("Processing PDF..."):

                # Sauvegarde temporaire du PDF
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(pdf_bytes)

                    file_path = temp_file.name

                try:

                    # Chargement du PDF
                    documents = load_pdf(file_path)

                    # Decoupage en chunks
                    chunks = split_documents(documents)

                    if not chunks:
                        raise ValueError(
                            "No extractable text found in this PDF."
                        )

                    # Generation des embeddings
                    embeddings = create_embeddings()

                    # Creation de la base vectorielle
                    db = create_vector_database(
                        chunks,
                        embeddings
                    )

                    # Creation de la chaine RAG
                    qa_chain = create_qa_chain(db)

                    # Sauvegarde dans la session
                    st.session_state.qa_chain = qa_chain

                    st.session_state.processed_file_hash = file_hash

                    # Reinitialisation de l'historique
                    st.session_state.messages = []

                finally:

                    # Suppression du fichier temporaire
                    if os.path.exists(file_path):
                        os.remove(file_path)

            st.success(
                f"PDF processed successfully! "
                f"{len(chunks)} chunks created."
            )

        except Exception as e:

            st.error(f"Error while processing PDF: {e}")

            st.session_state.qa_chain = None
            st.session_state.processed_file_hash = None


# ============================================================
# 10. AFFICHAGE DE L'HISTORIQUE
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# 11. INTERFACE DE QUESTIONS / REPONSES
# ============================================================

query = st.chat_input(
    "Ask a question about your PDF..."
)


# ============================================================
# 12. GENERATION DE LA REPONSE
# ============================================================

if query:

    if st.session_state.qa_chain is not None:

        # Afficher la question
        with st.chat_message("user"):

            st.markdown(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        try:

            with st.chat_message("assistant"):

                with st.spinner("Generating response..."):

                    # Execution de la chaine RAG
                    result = st.session_state.qa_chain.invoke(
                        {"input": query}
                    )

                    answer = result["answer"]

                    # Affichage de la reponse
                    st.markdown("### 💡 Answer")

                    st.write(answer)

                    # Affichage des sources
                    st.markdown("### 📚 Sources")

                    source_documents = result["context"]

                    if source_documents:

                        for doc in source_documents:

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            source = doc.metadata.get(
                                "source",
                                "Unknown"
                            )

                            if isinstance(page, int):
                                page = page + 1

                            st.write(
                                f"- **{os.path.basename(source)}** "
                                f"(Page {page})"
                            )

                    else:

                        st.info("No sources found.")

            # Sauvegarde de la reponse
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(
                f"Error while generating response: {e}"
            )

    else:

        st.warning(
            "Please upload and process a PDF document first."
        )


# ============================================================
# 13. SIDEBAR : INFORMATIONS
# ============================================================

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ System Information")

st.sidebar.write("LLM: Llama 3.2")
st.sidebar.write("Embeddings: MiniLM")
st.sidebar.write("Vector Database: Chroma")
st.sidebar.write("Framework: LangChain")

if st.sidebar.button("🗑️ Clear conversation"):

    st.session_state.messages = []

    st.rerun()
