import gradio as gr

import uuid
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from agents.main_agent import main_node, security_node
from agents.user_profile_agent import user_profile_node
from agents.vacancy_agent import vacancy_node
from agents.state import AgentState
from langchain_core.messages import HumanMessage

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("main_agent", main_node)
    workflow.add_node("security_node", security_node)
    workflow.add_node("user_profile_agent", user_profile_node)
    workflow.add_node("vacancy_agent", vacancy_node)
    
    workflow.set_entry_point("main_agent")

    def router(state: AgentState):
        return state.get("next_agent", "END")

    workflow.add_conditional_edges("main_agent", router)
    workflow.add_conditional_edges("security_node", router, {
        "main_agent": "main_agent",
        "END": END
    })
    workflow.add_conditional_edges("user_profile_agent", router)
    workflow.add_conditional_edges("vacancy_agent", router, {
        "main_agent": "main_agent",
        "END": END
    })
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)



graph = build_graph()

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

async def predict(message, history):

    inputs = {"messages": [HumanMessage(content=message)]}
    
    final_content = ""
    
    async for chunk in graph.astream(inputs, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if "messages" in update and update["messages"]:
                last_msg = update["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    final_content = last_msg.content

    if not final_content:
        final_content = "Обработка завершена. Чем я могу еще помочь?"
        
    return final_content

demo = gr.ChatInterface(
    fn=predict,
    title="Data Science Career AI",
    description="Загрузите резюме (ссылку на hh.ru) или спросите о рынке труда DS.",
    examples=[
        "Привет! Помоги проанализировать рынок DS.",
        "Подбери вакансии по моему резюме: [ссылка]",
        "Как мне стать Senior Data Scientist за год?"
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)