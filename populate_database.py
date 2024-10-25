from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    # Cria (ou altera) o data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


## Função para carregar os documentos.
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# #Teste do loader
# documents = load_documents()
# print(documents[0])

# Função para realizer a divisão do texto (Chunks)
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)

    # Teste para ver se o retorno é do tipo correto
    if isinstance(split_docs[0], Document):
        print("Os chunks são objetos do tipo Document.")
    else:
        print("⚠️ Aviso: Os chunks não são do tipo Document!")


    return split_docs

    # Teste do Split
    documents = load_documents()
    chunks = split_documents(documents)
    print(chunks[0])


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calcular o Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Adiciona ou Altera os documents.
    existing_items = db.get(include=[])  # IDs já estão inclusos por padrão
    existing_ids = set(existing_items["ids"])
    print(f"Número de documentos no DB: {len(existing_ids)}")

    # Só adiciona IDs que não estão no db.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Novos documentos adiconados: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ Não tem novos documentos para adicionar")


def calculate_chunk_ids(chunks):
    # Cria IDs como "data\Inativar_Usuário_SISTEMA_B_KB0053251.pdf:0:0"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Se o ID for igual ao último, incrementa o index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calcula o ID do chunk.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id


        # Adiciona o ID ao page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


if __name__ == "__main__":
    main()
