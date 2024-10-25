# Projeto RAG com Llama 3.2 e Chroma

Este projeto implementa uma API baseada em RAG (Retrieval-Augmented Generation) utilizando o modelo Llama 3.2-3B da Ollama e Chroma como banco de dados vetorial. A API permite realizar consultas em documentos PDF armazenados, retornando respostas baseadas no contexto extraído desses documentos.

## Estrutura do Projeto

- **data/**: Pasta para armazenar os documentos PDF a serem processados.
- **llama/**: Pasta para armazenar o modelo Llama 3.2-3B.

## Requisitos

1. **Python** 3.8+
2. **Ollama** instalado na máquina
3. Pacotes listados em `requirements.txt`

## Instalação

1. **Clone o repositório e instale as dependências**:

    ```bash
    git clone https://github.com/Brunoalaraujo/NextSofttek.git
    cd NextSofttek
    pip install -r requirements.txt
    ```

2. **Crie as pastas necessárias**:

    Na raiz do projeto, crie as pastas `data` e `llama`:

    ```bash
    mkdir data llama
    ```

3. **Baixe o modelo Llama 3.2-3B com o Ollama**:

    - Certifique-se de que o Ollama está instalado (siga as instruções de instalação no site da Ollama, caso necessário).
    - Navegue até a pasta `llama` e execute o comando para baixar o modelo:

    ```bash
    cd llama
    ollama run llama3.2
    ```

4. **Adicione documentos à pasta `data`**:

    Insira seus arquivos PDF na pasta `data` para que possam ser processados e indexados pela API.

5. **Execute o script `populate_database.py` para carregar os documentos**:

    ```bash
    python populate_database.py
    ```

    Este script processa os PDFs na pasta `data`, segmentando e armazenando os textos no banco de dados Chroma para consultas futuras.

## Utilização da API

### Execução

Para iniciar a API, execute o seguinte comando:

```bash
uvicorn main:app --reload
```

API estará disponível em [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Endpoints

#### POST /query
Recebe uma consulta em texto e retorna uma resposta baseada no conteúdo dos PDFs carregados.

**Exemplo de Requisição:**

- **Método:** POST
- **URL:** [http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)

**Corpo da requisição (JSON):**

```json
  {
    "query_text": "Como posso alterar o acesso de um usuário no sistema B?"
  }
```

**Resposta(JSON):**
```json
{
  "response": "Resposta baseada no contexto dos documentos.",
  "sources": ["KB0010536", "KB0053251"]
}
```