from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import  add_messages # Ensure this is correctly used or messages are plain lists
from typing import Annotated, List, Optional,Dict
from langgraph.checkpoint.sqlite import SqliteSaver
from ContextStore.cs import ImagePointer,DocumentPointer
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from ContextStore.cs import ContextStore
from typing_extensions import TypedDict
import sqlite3



class State(TypedDict):
    messages : Annotated[List,add_messages]


class ManagerAgent:
    def __init__(self,checkpoint_path: str, model: str, store: ContextStore, state : Optional[Dict] = State) -> None:
        self.connection = sqlite3.connect(checkpoint_path,check_same_thread=False)
        self.model = ChatGoogleGenerativeAI(model=model)
        self.agent_state = state
        self.graph = self._build_graph_()
        self.CS = store

    def _build_graph_(self):
        checkpoint = SqliteSaver(self.connection)
        graph_builder = StateGraph(self.agent_state)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile(checkpointer=checkpoint)
        return graph
    def chatbot(self, state: ContextStore):
        return {"messages": [self.model.invoke(state["messages"])]}
    
    def handle_and_save_input(self, request_data: dict):
        # if it has prompt, if it has images if it has docs etc
        prompt = request_data.get("prompt","Prompt not found")
        images = request_data.get("images",[])
        docs = request_data.get("docs",[])
        
        payload = {}

        if images:
            for image in images:
                image_pointer = ImagePointer(caption=image["caption"],image_data=image["image_data"])
                self.CS.img_manager.add_image(image_pointer)

        if docs:
            for doc in docs:
                doc_pointer = DocumentPointer(caption=doc["caption"],doc_data=doc["doc_data"])
                self.CS.doc_mngr.add_doc(doc_pointer)

        payload["messages"] = [{"role":"user","content":prompt}]

        return payload
        
    def run_prompt(self, prompt: str):
        
        payload = self.handle_and_save_input(prompt)
        response = self.graph.invoke(payload, {"configurable": {"thread_id": "2"}})
        return response



# agent = ManagerAgent("checkpoint.sqlite","models/gemini-2.0-flash",ContextStore("1",[],[]))







