import openai
from app.config import get_settings


class AsyncModelClient:
    def __init__(self):
        self.settings = get_settings()

        self.client = openai.AsyncOpenAI(
            base_url=self.settings.llm_address,
            api_key=self.settings.llm_api_key
        )

    async def __get_completion(self, prompt, temperature=0.7, system_prompt=None, stream=False):
        try:
            system_content = system_prompt or "You are a helpful chat assistant for humans on Russian and English languages."
            completion = await self.client.chat.completions.create(
                model=self.settings.llm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                stream=stream
            )
            return completion
        except Exception as e:
            return f"Error to connect to llm server llama.cpp: {e}"

    async def generate(self, prompt, temperature=0.7, system_prompt=None):
        try:
            completion = await self.__get_completion(prompt, 
                                                     temperature, 
                                                     stream=False,
                                                     system_prompt=system_prompt)
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error to connect to llm server llama.cpp: {e}"

    async def generate_stream(self, prompt, temperature=0.7, system_prompt=None):
        try:
            stream = await self.__get_completion(prompt, 
                                                 temperature, 
                                                 stream=True,
                                                 system_prompt=system_prompt)
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
                if chunk.choices[0].finish_reason:
                    break
        except Exception as e:
            yield f"Error to connect to llm server llama.cpp: {e}"


class AsyncEmbModelClient:
    def __init__(self):
        self.settings = get_settings()

        self.client = openai.AsyncOpenAI(
            base_url=self.settings.emb_address,
            api_key=self.settings.emb_api_key
        )

    async def embedding(self, content):
        try:
            response = await self.client.embeddings.create(
                model=self.settings.emb_model_name,
                input=content,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            return f"Error to connect to embedding server llama.cpp: {e}"


