"""Here, we define the chat generator that we will use to generate responses to user queries."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI

load_dotenv()


def get_chat_generator(model_name: str) -> AzureChatOpenAI:
    if model_name == "gpt-35-turbo-instruct":
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif model_name == "gpt-4.1-mini":
        return AzureChatOpenAI(
            model_name="gpt-4.1-mini",
            temperature=0,
            azure_endpoint=os.getenv("GPT_41_MINI_ENDPOINT"),
            api_key=os.getenv("GPT_41_MINI_API_KEY"),
            api_version="2024-12-01-preview",
        )

    else:
        raise ValueError(f"{model_name} is not a valid chat generator name")
