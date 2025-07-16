import datetime

LOG_FILE = 'logs.txt'

def log_html_agent_event(html: str, validator_response: str, evaluator_response: str, document_structure: str = None):
    """
    Log the HTML agent's output, validator response, evaluator response, and optionally document_structure to logs.txt with a timestamp.
    """
    return
    timestamp = datetime.datetime.now().isoformat()
    log_entry = f"\n--- HTML AGENT EVENT [{timestamp}] ---\n"
    if document_structure:
        log_entry += f"Document Structure:\n{document_structure}\n"
    log_entry += (
        f"Generated HTML:\n{html}\n"
        f"Validator Response:\n{validator_response}\n"
        f"Evaluator Response:\n{evaluator_response}\n"
        f"--- END EVENT ---\n"
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def log_content_agent_event(description: str, style_guidelines: str, context: str, result: str, status: str, document_structure: str = None):
    """
    Log the ContentAgent's content generation event, including description, style guidelines, context, result, status, and optionally document_structure.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_entry = f"\n--- CONTENT AGENT EVENT [{timestamp}] ---\n"
    if document_structure:
        log_entry += f"Document Structure:\n{document_structure}\n"
    log_entry += (
        f"Description: {description}\n"
        f"Style Guidelines: {style_guidelines}\n"
        f"Context: {context}\n"
        f"Status: {status}\n"
        f"Result:\n{result}\n"
        f"--- END EVENT ---\n"
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry) 