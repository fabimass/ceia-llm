import streamlit as st
import os
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Carga la clave de API de GROQ desde las variables de entorno
groq_api_key = os.environ.get("API_KEY_GROQ")

# Crea el cliente de GROQ
client = Groq(
    api_key=groq_api_key,
)

# Embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Conexión con base de datos
pinecone = PineconeVectorStore(
    embedding=embed_model,
    index_name="ceia",
    pinecone_api_key=os.getenv("API_KEY_PINECONE")
)

# Función para obtener el contexto de la query
def search_vstore(query, vstore):
    context = ""
    results = vstore.similarity_search(query, k=3)
    for res in results:
        context += res.page_content
    return context

# Inicializa el historial de conversación en el estado de la sesión
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def generate_response(input_text, context):
    query = f"""
    Contesta la siguiente pregunta en base al contexto provisto.

    Pregunta: {input_text}

    Contexto: {context}
    """

    # Agrega el mensaje del usuario al historial de conversación
    st.session_state.conversation_history.append({"role": "user", "content": query})

    # Genera la respuesta del chatbot utilizando el modelo LLaMA 3 y el historial de la conversación
    chat_completion = client.chat.completions.create(
        messages=st.session_state.conversation_history,
        model="llama3-8b-8192",
    )
    response = chat_completion.choices[0].message.content

    # Agrega la respuesta del chatbot al historial de conversación
    st.session_state.conversation_history.append({"role": "assistant", "content": response})

    return response

# Configuración de la interfaz de Streamlit
st.title("Chatbot con LLaMA 3")
st.subheader("¡Hazme una pregunta!")

user_input = st.text_input("Usuario:", "")

if user_input:
    context = search_vstore(user_input, pinecone)
    response = generate_response(user_input, context)
    st.write(f"**Chatbot**: {response}")
