import streamlit as st
import re
from langgraph.graph import StateGraph
from agent import AgentRag, AgentLlm, AgentState

agent_1 = AgentRag("fabian-cv")
agent_2 = AgentRag("mock-cv")
llm = AgentLlm()

# Esta funcion detecta patrones en la pregunta del usuario
def detector(state: AgentState):
    persons_detected = []
    patterns = {
        "mock": r'\brichard\b.*\?',
        "fabian": r'\bfabian\b.*\?'
    }
    
    for name, pattern in patterns.items():
        match = bool(re.search(pattern, state["question"], re.IGNORECASE))
        if match:
            persons_detected.append(name)

    # Si no se detecto ningun patron, se va a utilizar el cv del alumno
    if len(persons_detected) == 0:
        persons_detected.append("fabian")
        
    return {"detector": persons_detected, "context": []}

# Esta funcion chequea si tiene que seguir iterando o no
def orchestrator(state: AgentState):
    print(state)
    if len(state["context"]) == len(state["detector"]):
        return { "complete": True } 
    else:
        return { "complete": False } 
    
# Esta funcion indica a que nodo ir desde el orchestrator
def next_node(state: AgentState):
    if state["complete"]:
        return "llm"
    else:
        return state["detector"][len(state["context"])]
    

builder = StateGraph(AgentState)
builder.add_node("detector_node", detector)
builder.add_node("orchestrator_node", orchestrator)
builder.add_node("agent_1", agent_1.search)
builder.add_node("agent_2", agent_2.search)
builder.add_node("llm_node", llm.generate)
builder.add_edge("detector_node", "orchestrator_node")
builder.add_conditional_edges(
    "orchestrator_node", 
    next_node, 
    {"fabian": "agent_1", "mock": "agent_2", "llm": "llm_node"}
)
builder.add_edge("agent_1", "orchestrator_node")
builder.add_edge("agent_2", "orchestrator_node")
builder.set_entry_point("detector_node")
graph = builder.compile()


# Configuración de la interfaz de Streamlit
st.title("Chatbot con LLaMA 3")
st.subheader("¡Hazme una pregunta!")

user_input = st.text_input("Usuario:", "")

if user_input:
    output = graph.invoke({"question": user_input})
    print(output)
    st.write(f"**Chatbot**: {output["llm"]}")
