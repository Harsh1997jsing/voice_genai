from openai import AsyncOpenAI
from .config import settings

class OpenAIClient():
    def __init__(self, model="gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model

    async def complete(self, prompt: str, system=None, **kwargs):
        if isinstance(system, (tuple, list)):
            system = " ".join(str(part) for part in system if part is not None)
        if isinstance(prompt, (tuple, list)):
            prompt = " ".join(str(part) for part in prompt if part is not None)

        print(f"OpenAIClient: Completing with model '{self.model}'")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        return response.choices[0].message.content
    


class OpenAIEmbeddingClient:
    def __init__(self, model="text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model

    async def embed(self, texts):
        is_single_input = isinstance(texts, str)
        inputs = [texts] if is_single_input else texts

        response = await self.client.embeddings.create(
            model=self.model,
            input=inputs
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if is_single_input else embeddings
