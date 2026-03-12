import os
import openai
import ollama
from groq import Groq

class Agents:
    def __init__(self, provider: str = "openai", model: str = "gpt-4-turbo-preview"):
        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = openai.OpenAI(api_key=os.environ["OPENAI"])
        elif provider == "groq":
            self.client = Groq(api_key=os.environ["GROQ"])

    def chat(self, messages, temperature=0.3):
        if self.provider == "openai":
            return self._chat_openai(messages, temperature)
        elif self.provider == "ollama":
            return self._chat_ollama(messages)
        elif self.provider == "groq":
            return self._chat_groq(messages, temperature)
        else:
            raise ValueError("Proveedor no soportado")

    def _chat_openai(self, messages, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content

    def _chat_ollama(self, messages):

        response = ollama.chat(
            model=self.model,
            messages=messages
        )

        return response["message"]["content"]
    
    def _chat_groq(self, messages, temperature):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content