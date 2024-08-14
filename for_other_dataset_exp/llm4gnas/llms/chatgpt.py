import json

import requests

from llm4gnas.llms.llm_base import LLMBase
import logging


class ChatGPT(LLMBase):
    def __init__(self, API_KEY: str, model="gpt-3.5-turbo"):
        self.key = API_KEY  # input your openai_api_key
        self.model = model  # choice your base model. such:"gpt-4","gpt-3.5-turbo"
        super().__init__()

    def response(self, system_content: str, prompt: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key
        }
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # check status
            res = response.json()
        except (requests.HTTPError, json.JSONDecodeError) as err:
            logging.error(f"JSON parsing error:{err}")
            raise err
        except Exception as err:
            logging.error(f"JSON parsing error:{err}")
            raise err

        result_value = res['choices'][0]['message']['content']
        # print(res_temp)
        return result_value
