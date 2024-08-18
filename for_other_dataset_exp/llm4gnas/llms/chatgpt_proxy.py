import json

import requests

from for_other_dataset_exp.llm4gnas.llms.llm_base import LLMBase


class ChatGPTProxy(LLMBase):
    def __init__(self, API_KEY: str, model="gpt-3.5-turbo", temperature=0):
        self.key = API_KEY  # input your openai_api_key
        self.model = model  # choice your base model. such:"gpt-4","gpt-3.5-turbo"
        self.temperature = temperature  # choice your temperature. such:0,0.5,1
        super().__init__()

    def response(self, system_content: str, prompt: str):
        url = "https://api.openai-sb.com/v1/chat/completions"
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
            "temperature": self.temperature
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # check status
            res = response.json()
            result_value = res['choices'][0]['message']['content']
            return result_value
        except (requests.HTTPError, json.JSONDecodeError) as err:
            print("JSON parsing error:", err)
        except Exception as err:
            print("Other exceptions:", err)

        # except requests.exceptions.RequestException as err:
        #     print("Request error occurred:", err)
        #     return None

