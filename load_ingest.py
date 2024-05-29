import os
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

llm = ChatOpenAI(
    base_url="http://host.docker.internal:1234/v1",
    api_key="lm-studio",
    temperature=0.6
)

transformer_path = "./faiss"


def load_documents(clone_url):
    if not os.path.exists(transformer_path):
        loader = GitLoader(
            clone_url=clone_url,
            repo_path="./docs/repo",
            branch="master"
        )
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(documents=raw_documents)
        print(f"Split into {len(documents)} chunks")
        return documents
    else:
        return None


def ingest_docs(question, documents) -> str:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") #all-MiniLM-L6-v2, all-mpnet-base-v2
    #documents = load_documents(clone_url)
    if documents:
        print(f"vectorsore started")
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectorstore.save_local(transformer_path)
        print(f"vectorsore ended")

    my_vectorstore = FAISS.load_local(transformer_path, embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=my_vectorstore.as_retriever(), chain_type="stuff")
    response = qa.invoke({"query": question})
    print('Vectorstore loaded')
    return response["result"]
