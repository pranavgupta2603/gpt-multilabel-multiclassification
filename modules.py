import base64
import io
from pydantic import BaseModel

class Response(BaseModel):
    final_answer: list[str]

class LLM:
    
    def __init__(self, client, sys_prompt=None):
        self.client = client
        self.sys_prompt = sys_prompt
        if self.sys_prompt:
            self.sys_prompt = {"role": "developer", "content": self._preprocess_content(self.sys_prompt)}
            
    def _preprocess_content(self, text):
        text = text.split("\u2060")
        text = "".join(text)
        return text
    
    def get_description(self, model_choice, content):
        messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        
        if self.sys_prompt:
            messages.insert(0, self.sys_prompt)
        
        response = self.client.beta.chat.completions.parse(
            model=model_choice,
            messages=messages,
            response_format=Response,
            max_tokens=300,
            temperature=0.01,
            top_p=0.0,
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

    def text_content(self, text):
        return {"type": "text", "text": self._preprocess_content(text)}

    def image_content(self, uploaded_file):
        encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        cont = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
            }
        return cont