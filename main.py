import flask
from flask import request, session

from agents.manager import ManagerAgent

from dotenv import load_dotenv
import os
load_dotenv()

app = flask.Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session

# Global store for agents (for demo; use Redis or DB for production)
agent_store = {}

def get_session_id():
    # Use Flask's session or generate a new one
    if "session_id" not in session:
        import uuid
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

@app.route("/prompt", methods=["POST"])
def handle_prompt():
    session_id = get_session_id()
    agent = agent_store.get(session_id)
    if agent is None:
        # Create a new ManagerAgent for this session
        agent = ManagerAgent(
            checkpoint_path="data/checkpoint.sqlite",
            model="models/gemini-2.5-pro",
            store=None  # or your ContextStore instance
        )
        agent_store[session_id] = agent
    input_data = request.json
    response = agent.run_prompt(input_data)

    print(response)
    # print(request.json.get("text"))
    return flask.jsonify({"message": "ManagerAgent instance ready for this session (placeholder)",
                          "agent_response": "This should be the response"})



if __name__ == "__main__":
    app.run(debug=True)
