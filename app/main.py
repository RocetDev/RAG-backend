import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
import uvicorn

from app.services.chat import ChatService
from app.models import ChatRequest, ChatResponse, SaveDocRequest


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)
agent = ChatService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Init app...")
    await agent.qdrand_base.initialize()
    logger.info("Qdrant was initialized!")
    yield
    logger.info('App is stoped')


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Запрос: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Ответ: {request.method} {request.url.path} - Status: {response.status_code}")
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await agent.chat(
            user_id=request.user_id,
            message=request.message
        )

        logger.info(f"Succefull response to user {request.user_id}")
        return ChatResponse(message=response)
    except Exception as e:
        logger.error(f"Error in chat_endpoint for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save/doсs")
async def save_doc_endpoint(request: SaveDocRequest):
    try:
        await agent.qdrand_base.save_documents(documents=request.documents)

        logger.info(f"Succefull loading of {len(request.documents)} documents")
        return {"status": "OK", "saved": len(request.documents)}
    except Exception as e:
        logger.error(f"Error in save_doc_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)