from IPython.display import Image, display
from ContentAgent.html_ag import ContentAgent

content_agent = ContentAgent()


try:
    display(Image(content_agent.graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass