import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4o-mini"
openai = OpenAI()

# System message for the assistant
system_message = (
    "You are a helpful assistant that counts how many times specific alphabets "
    "appear in a given sentence. Provide accurate counts."
)

# Alphabet counting function
def count_alphabets(sentence, letters):
    """Count occurrences of specified letters in a sentence."""
    counts = {}
    for letter in letters:
        counts[letter] = sentence.lower().count(letter.lower())
        print(sentence)
    return counts

# Define the tool for alphabet counting
alphabet_count_function = {
    "name": "count_alphabets",
    "description": (
        "Count how many times specific alphabets appear in a sentence. Use this for queries like "
        "'How many a's and b's are in this sentence?'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sentence": {
                "type": "string",
                "description": "The sentence to analyze.",
            },
            "letters": {
                "type": "array",
                "items": {"type": "string", "maxLength": 1},
                "description": "The alphabets to count (e.g., ['a', 'b', 'c']).",
            },
        },
        "required": ["sentence", "letters"],
        "additionalProperties": False
    }
}

# Tools array
tools = [{"type": "function", "function": alphabet_count_function}]

# Chat function
def chat(message, history):
    messages = [
        {"role": "system", "content": system_message}
    ] + history + [
        {"role": "user", "content": message}
    ]

    response = openai.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, sentence, letters = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

# Handle tool calls
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    sentence = arguments.get('sentence')
    letters = arguments.get('letters')
    counts = count_alphabets(sentence, letters)
    response = {
        "role": "tool",
        "content": json.dumps({"sentence": sentence, "counts": counts}),
        "tool_call_id": tool_call.id
    }
    return response, sentence, letters

# Launch Gradio chat interface
gr.ChatInterface(fn=chat, type="messages").launch()
