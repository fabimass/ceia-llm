from typing import TypedDict, List
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

class AgentState(TypedDict):
    question: str
    context: List[str]
    detector: List[str]
    agent: str
    llm: str
    complete: bool


class AgentRag:    
    def __init__(self, index): 

        # Embeddings
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Conexión con base de datos
        self.vstore = PineconeVectorStore(
            embedding=self.embed_model,
            index_name=index,
            pinecone_api_key=os.getenv("API_KEY_PINECONE")
        )
       
    def search(self, state: AgentState):
        if "context" in state:
            context_list = state["context"]
        else:
            context_list = []
        
        context = ""
        results = self.vstore.similarity_search(state["question"], k=3)
        for res in results:
            context += res.page_content
        
        return { "context": context_list + [context] } 
    

class AgentLlm:    
    def __init__(self):
        # Crea el cliente de GROQ
        self.client = Groq(api_key=os.environ.get("API_KEY_GROQ"))

    def generate(self, state: AgentState):
        input = state["question"]
        context = ""
        for ctx in state["context"]:
            context += ctx
            
        query = f"""
        Contesta la siguiente pregunta en base al contexto provisto.

        Pregunta: {input}

        Contexto: {context}
        """

        # Genera la respuesta del chatbot utilizando el modelo LLaMA 3
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="llama3-8b-8192",
        )
        response = chat_completion.choices[0].message.content

        return {"llm": response}