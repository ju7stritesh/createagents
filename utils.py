import json
import datetime

def save_chat_history(history, filename):
    """Save chat history to a JSON file."""
    with open(f'chat_history_{filename}.json', 'w') as f:
        json.dump(history, f, indent=2)

def load_chat_history(filename):
    """Load chat history from a JSON file."""
    try:
        with open(f'chat_history_{filename}.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def format_chat_message(role, content, sources=None):
    """Format a chat message with timestamp and sources."""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.datetime.now().isoformat()
    }
    if sources:
        message['sources'] = sources
    return message 