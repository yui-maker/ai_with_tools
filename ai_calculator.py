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
    "You are a helpful calculator assistant. "
    "Provide accurate answers for mathematical queries. "
    "If the query is not mathematical, politely say you can't handle it."
)

# Calculator function
def calculate_expression(expression):
    """Evaluate a mathematical expression and return the result."""
    try:
        # Use eval safely for mathematical expressions
        result = eval(expression, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Define the tool for the calculator
calculator_function = {
    "name": "calculate_expression",
    "description": "Evaluate mathematical expressions. Use this for any calculation, e.g., 'What is 5 + 7?'.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g., '5 + 7' or '2 * (3 + 4)'",
            },
        },
        "required": ["expression"],
        "additionalProperties": False
    }
}

# Tools array
tools = [{"type": "function", "function": calculator_function}]

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
        response, expression = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

# Handle tool calls
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    expression = arguments.get('expression')
    result = calculate_expression(expression)
    response = {
        "role": "tool",
        "content": json.dumps({"expression": expression, "result": result}),
        "tool_call_id": tool_call.id
    }
    return response, expression

# Launch Gradio chat interface
gr.ChatInterface(fn=chat, type="messages").launch()
