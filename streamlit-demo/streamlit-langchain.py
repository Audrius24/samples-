import os
import streamlit as st
from dotenv import load_dotenv
import bs4

from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
token = os.getenv("SECRET")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# === Load source 1: Local text file ===
def load_file_source(path):
    try:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = "📄 Local File"
        print("✅ Loaded local file.")
        return docs
    except Exception as e:
        print(f"⚠️ Failed to load local file: {e}")
        return []

# === Load source 2: Wikipedia ===
def load_wikipedia_source():
    print("🔄 Scraping Wikipedia...")
    url = "https://en.wikipedia.org/wiki/Gargždai"
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="mw-parser-output"))
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = "🌐 Wikipedia"
    print("✅ Loaded Wikipedia.")
    return docs

# === Load source 3: VLE (Lithuanian Encyclopedia) ===
def load_extra_web_source():
    print("🔄 Scraping VLE...")
    url = "https://www.vle.lt/straipsnis/gargzdai/"
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(name="main"))  # scrape main tag
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = "🌍 VLE Encyclopedia"
    print("✅ Loaded VLE source.")
    return docs

# === Load all sources ===
file_path = "C:/Users/justj/samples/streamlit-demo/gargzdai.txt"
file_docs = load_file_source(file_path)
wiki_docs = load_wikipedia_source()
vle_docs = load_extra_web_source()

# Combine documents
all_docs = file_docs + wiki_docs + vle_docs

# === Text splitting ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(all_docs)

# === Embedding and Vector Store ===
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    ),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Streamlit UI ===
st.title("📍 Gargždai RAG Chatbot (3 Sources)")

def generate_response(query):
    llm = ChatOpenAI(
        base_url=endpoint,
        temperature=0.7,
        api_key=token,
        model=model,
    )

    fetched_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in fetched_docs])

    prompt = f"""You are a helpful assistant answering questions about Gargždai, Lithuania.
Use the context below to answer the user's question. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    return response, fetched_docs

# === Input Form ===
with st.form("input_form"):
    user_input = st.text_area("Ask me about Gargždai:", height=150)
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            answer, sources = generate_response(user_input)
            st.success(answer)

            st.subheader("📚 Sources")
            for i, doc in enumerate(sources, 1):
                source_name = doc.metadata.get("source", "Unknown Source")
                with st.expander(f"Source {i} - {source_name}"):
                    st.write(doc.page_content)
