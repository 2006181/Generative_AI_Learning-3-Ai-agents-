from langchain.tools import tool


@tool#decorator for creating a tool
def get_greeting(name : str) -> str:
    """Generate a greeting message for a user"""

    return f"Hello {name}, Welcomer to the AI world"

result = get_greeting.invoke({"name": "Rahul"})

print(result)
