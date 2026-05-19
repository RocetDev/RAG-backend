from qdrant_client import AsyncQdrantClient
from qdrant_client import models
import uuid
from .llm_client import AsyncEmbModelClient
from app.config import get_settings


class QdrantService:
    def __init__(self, embedder: AsyncEmbModelClient):
        self.settings = get_settings()
        self.client = AsyncQdrantClient(self.settings.qdrant_address)
        self.embedder = embedder
        self._initialized = False
        
    async def initialize(self):
        if self._initialized:
            return

        base_name = self.settings.qdrant_base_colname
        memory_name= self.settings.qdrant_memory_colname

        if not await self.client.collection_exists(base_name):
            await self.client.create_collection(
                collection_name=base_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size = self.settings.dense_vec_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )

        if not await self.client.collection_exists(memory_name):
            await self.client.create_collection(
                collection_name=memory_name,
                vectors_config = models.VectorParams(
                    size = self.settings.dense_vec_size,
                    distance=models.Distance.COSINE
                )
            )

            await self.client.create_payload_index(
                collection_name=memory_name,
                field_name="user_id",
                field_schema="keyword"
            )

        self._initialized = True

    async def save_memory(self, user_id: str, text: str):
        vector = await self.embedder.embedding(text)
        await self.client.upsert(
            collection_name=self.settings.qdrant_memory_colname,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "text": text
                    }
                )
            ]
        )

    async def search_memory(self, user_id: str, text: str):
        search_result = await self.client.query_points(
            collection_name=self.settings.qdrant_memory_colname,
            query = await self.embedder.embedding(text),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=self.settings.max_memory_context
        )

        search_result = [data.payload['text'] for data in search_result.points]
        return search_result

    async def save_documents(self, documents: list[dict]):
        points = []
        for doc in documents:
            text = doc['text']
            dense_vector = await self.embedder.embedding(text)
            sparse_vector = models.Document(text=text, model="Qdrant/bm25")

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector
                    },
                    payload=doc
                )
            )

        await self.client.upsert(
            collection_name=self.settings.qdrant_base_colname,
            points=points
        )

    async def search_rag(self, text: str):
        search_result = await self.client.query_points(
            collection_name=self.settings.qdrant_base_colname,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            prefetch=[
                models.Prefetch(
                    query = await self.embedder.embedding(text),
                    using='dense'
                ),
                models.Prefetch(
                    query= models.Document(text=text, model="Qdrant/bm25"),
                    using='sparse'
                )
            ],
            query_filter=None,
            limit=self.settings.max_doc_context
        )

        metadata = [point.payload["text"] for point in search_result.points]
        return metadata

