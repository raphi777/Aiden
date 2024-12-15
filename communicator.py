import json
import os
from dotenv import load_dotenv
from openai import OpenAI


def call_openai(prompt: str):
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant for generating Q&A couples out of given study material."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    print(response)
    return response.choices[0].message.content
