from .llm_client import AsyncEmbModelClient, AsyncModelClient
from .qdrant_base import QdrantService
from app.config import get_settings


class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.llm = AsyncModelClient()
        self.qdrand_base = QdrantService(AsyncEmbModelClient())

    async def chat(self, user_id: str, message: str):
        memories = await self.qdrand_base.search_memory(user_id, message)
        memory_context = "\n".join(memories) if memories else "Nothing"


        rag_docs = await self.qdrand_base.search_rag(message)
        rag_context = "\n".join(rag_docs) if rag_docs else "Nothing"
        
        prompt = f"""
        Ты полезный ассистент.

        ---ПАМЯТЬ (Прошлые диалоги)---    
        {memory_context}

        ---БАЗА ЗНАНИЙ (Документы)---
        {rag_context}

        ---ПОЛЬЗОВАТЕЛЬСКИЙ ЗАПРОС---
        {message}

        Ответь на вопрос, используя контекст выше
        """

        response = await self.llm.generate(prompt)

        await self.qdrand_base.save_memory(
            user_id,
            f"User: {message}\nAI: {response}"
        )

        return response
