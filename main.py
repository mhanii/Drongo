from Agents.ContentAgent.html_ag import HtmlAgent
from dotenv import load_dotenv

load_dotenv()

html_agent = HtmlAgent()

html_agent.run("Create a landing page for a new product", "The page should be modern and sleek", "")

# html_agent.get_graph()
