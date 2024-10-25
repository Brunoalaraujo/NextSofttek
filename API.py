from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

# Parâmetros gerais
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Traduza a pergunta com base apenas no seguinte contexto:

{context}

---

Baseie-se apenas no contexto acima para responder à pergunta: {question}
"""

# Inicialize o FastAPI
app = FastAPI()


# Defina o modelo de entrada para a consulta
class QueryRequest(BaseModel):
    query_text: str


# Defina a rota para a consulta
@app.post("/query/")
async def query_rag(request: QueryRequest):
    query_text = request.query_text

    # Inicialize o banco de dados
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Execute a busca no banco de dados
    results = db.similarity_search_with_score(query_text, k=3)
    if not results:
        raise HTTPException(status_code=404, detail="Nenhum contexto relevante encontrado.")

    # Prepare o contexto para o prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoque o modelo Ollama
    model = Ollama(model="llama3.2:latest")
    response_text = model.invoke(prompt)

    # Prepare as fontes de origem
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return {"response": response_text, "sources": sources}
