import os

import sqlbot_xpack
from alembic.config import Config
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.routing import APIRoute
# MCP 功能已禁用
# from fastapi.staticfiles import StaticFiles
# from fastapi_mcp import FastApiMCP
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

from alembic import command
from apps.api import api_router
from common.utils.embedding_threads import fill_empty_table_and_ds_embeddings
from apps.system.crud.aimodel_manage import async_model_info
from apps.system.crud.assistant import init_dynamic_cors
from apps.system.middleware.auth import TokenMiddleware
from common.core.config import settings
from common.core.security_check import enforce_security_check
from common.core.response_middleware import ResponseMiddleware, exception_handler
from common.core.sqlbot_cache import init_sqlbot_cache
from common.utils.embedding_threads import fill_empty_terminology_embeddings, fill_empty_data_training_embeddings
from common.utils.utils import SQLBotLogUtil


def run_migrations():
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


def init_terminology_embedding_data():
    fill_empty_terminology_embeddings()


def init_data_training_embedding_data():
    fill_empty_data_training_embeddings()


def init_table_and_ds_embedding():
    fill_empty_table_and_ds_embeddings()


def preload_embedding_model():
    """预加载 embedding 模型到内存"""
    try:
        from apps.ai_model.embedding import EmbeddingModelCache
        SQLBotLogUtil.info("🔄 开始预加载 embedding 模型...")
        start_time = __import__('time').time()
        
        # 预加载模型
        model = EmbeddingModelCache.get_model()
        
        # 进行一次预热推理
        _ = model.embed_query("测试")
        
        elapsed = __import__('time').time() - start_time
        SQLBotLogUtil.info(f"Embedding 模型预加载完成，耗时: {elapsed:.2f}秒")
    except Exception as e:
        SQLBotLogUtil.error(f"Embedding 模型预加载失败: {str(e)}")
        # 不阻断应用启动，允许后续懒加载


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 首先进行安全检查
#     这是一个防守性编程措施，确保：

# 生产环境安全 - 防止使用默认密钥的应用被部署
# 开发者友好 - 如果密钥不安全，启动时立即失败
    SQLBotLogUtil.info("🔒 执行安全检查...")
    enforce_security_check(settings.SECRET_KEY, strict_mode=True)
    
    run_migrations()
    init_sqlbot_cache()
    init_dynamic_cors(app)
    preload_embedding_model()  # 预加载 embedding 模型
    init_terminology_embedding_data()
    init_data_training_embedding_data()
    init_table_and_ds_embedding()
    SQLBotLogUtil.info("✅ SQLBot 初始化完成")
    await sqlbot_xpack.core.clean_xpack_cache()
    await async_model_info()  # 异步加密已有模型的密钥和地址
    yield
    SQLBotLogUtil.info("SQLBot 应用关闭")


def custom_generate_unique_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags and len(route.tags) > 0 else ""
    return f"{tag}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan
)

# MCP 功能已禁用
# mcp_app = FastAPI()
# # mcp server, images path
# images_path = settings.MCP_IMAGE_PATH
# os.makedirs(images_path, exist_ok=True)
# mcp_app.mount("/images", StaticFiles(directory=images_path), name="images")
# 
# mcp = FastApiMCP(
#     app,
#     name="SQLBot MCP Server",
#     description="SQLBot MCP Server",
#     describe_all_responses=True,
#     describe_full_response_schema=True,
#     include_operations=["get_datasource_list", "get_model_list", "mcp_question", "mcp_start", "mcp_assistant"]
# )
# 
# mcp.mount(mcp_app)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(TokenMiddleware)
app.add_middleware(ResponseMiddleware)
app.include_router(api_router, prefix=settings.API_V1_STR)

# Register exception handlers
app.add_exception_handler(StarletteHTTPException, exception_handler.http_exception_handler)
app.add_exception_handler(Exception, exception_handler.global_exception_handler)

# MCP 功能已禁用
# mcp.setup_server()

sqlbot_xpack.init_fastapi_app(app)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # uvicorn.run("main:mcp_app", host="0.0.0.0", port=8001) # mcp server
