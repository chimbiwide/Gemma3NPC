import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# env variables and Deepseek chat
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
ds_api = os.getenv("DS_API")

client = AsyncOpenAI(
    base_url="https://api.novita.ai/openai",
    api_key=ds_api,
)


async def respond(
    system_prompt: str, user_prompt: str, max_tokens: int = 15000
) -> tuple:
    model = "deepseek/deepseek-v3.2"
    stream = False
    max_tokens = 15000
    system = system_prompt
    user = user_prompt
    temperature = 1.3
    top_p = 0.95
    min_p = 0
    presence_penalty = 0
    frequency_penalty = 0
    repetition_penalty = 1.1
    response_format = {"type": "json_object"}
    chat_completion_res = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        response_format=response_format,
        extra_body={
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "thinking": {"type": "enabled"},
        },
    )

    thinking: str = chat_completion_res.choices[0].message.reasoning_content
    response: str = chat_completion_res.choices[0].message.content
    # return f"{thinking} \n\n {response}"
    return (thinking, response)


if __name__ == "__main__":
    print(respond("be a good boy, use english when you think", "Hi DeepSeek"))
