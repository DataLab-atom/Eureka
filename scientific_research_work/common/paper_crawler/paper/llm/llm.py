from openai import OpenAI
import os
from pathlib import Path
from dotenv import dotenv_values

config = dotenv_values(os.path.join(Path(os.path.abspath(__file__)).parents[5], '.env'))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

class LLM:
    def __init__(self, system_prompt, model="gpt-4.1-mini", temperature=0):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    def run(self, prompt):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": str(prompt)}
                ],
                response_format={ "type": "json_object" },
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"
