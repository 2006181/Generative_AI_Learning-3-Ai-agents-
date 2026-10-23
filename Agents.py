from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print

#Now lets create some tools

#weather tool

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return (
            f"Weather in {data['name']}:\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Condition: {description}\n"
            f"Humidity: {humidity}%"
        )

    except requests.RequestException as e:
        return f"Weather API error: {e}"
    except (KeyError, IndexError, ValueError):
        return "Could not parse weather data."



# Tavily news tool
tavily = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="news",
    time_range="day"
)

@tool
def get_news(city: str) -> str:
    """Get latest news about a city with article links."""

    result = tavily.invoke({
        "query": f"Latest news in {city}"
    })

    news_list = result.get("results", [])

    if not news_list:
        return "No news found."

    formatted_news = []

    for i, news in enumerate(news_list, start=1):
        title = news.get("title", "No title")
        summary = news.get("content", "No summary")
        link = news.get("url", "No link available")

        formatted_news.append(
            f"News {i}: {title}\n"
            f"Summary: {summary}\n"
            f"Read more: {link}"
        )

    return "\n\n".join(formatted_news)



llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    timeout=30,
    max_retries=1
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tool = llm.bind_tools([get_weather, get_news])


# Agent loop - Human in the Loop

messages = []

print("City Intelligence System")
print("Type 'exit' to exit")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            denied = False

            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]

                confirm = input(
                    f"Agent wants to call {tool_name}. "
                    "Approve? (yes/no): "
                ).strip().lower()

                if confirm not in ("yes", "y"):
                    print(f"Tool call '{tool_name}' denied.")

                    messages.append(ToolMessage(
                        content="The user denied this tool call. "
                                "Do not execute it.",
                        tool_call_id=tool_call["id"]
                    ))

                    denied = True
                    break

                # Execute only after approval
                tool_result = tools[tool_name].invoke(
                    tool_call["args"]
                )

                print(f"{tool_name} executed successfully.")

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))

            if denied:
                break

            continue

        else:
            content = result.content

            if isinstance(content, list):
                content = "\n\n".join(
                    item["text"]
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )

            print(f"\n{content}\n")
            break

