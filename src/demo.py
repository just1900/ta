
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage

from agents.sql_workflow import SQLOptimizationWorkflow
from mcp import StdioServerParameters
from agno.tools.mcp import MCPTools
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from agno.knowledge.text import TextKnowledgeBase
from pathlib import Path
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.openai import OpenAIEmbedder

load_dotenv()
nest_asyncio.apply()

knowledge_base = TextKnowledgeBase(
    path=Path("docs/llms-full.txt"),
    # Use LanceDB as the vector database and store embeddings in the `tidb_docs` table
    vector_db=PgVector(
        db_url=os.getenv("PGVECTOR_DB_URL"),
        table_name="tidb",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(
            id="text-embedding-3-small"
        )
    )
)

async def run_workflow() -> None:

    tidb_server_params = StdioServerParameters(
        command="uv",
        args=["--directory", os.getenv("TIDB_MCP_SERVER_DIR"), "run", "src/main.py"],
        env={
            "TIDB_HOST": os.getenv("TIDB_HOST"),
            "TIDB_PORT": os.getenv("TIDB_PORT"),
            "TIDB_USERNAME": os.getenv("TIDB_USERNAME"),
            "TIDB_PASSWORD": os.getenv("TIDB_PASSWORD"),
            "TIDB_DATABASE": os.getenv("TIDB_DATABASE")
        }
    )

    # Metrics server for querying Prometheus metrics
    o11y_server_params = StdioServerParameters(
        command="mcp-grafana",
        env={
            "GRAFANA_URL": os.getenv("GRAFANA_URL"),
            "GRAFANA_API_KEY": os.getenv("GRAFANA_API_KEY"),
        }
    )
    async with MCPTools(server_params=tidb_server_params,include_tools=["db_execute","db_query","show_create_table"]) as tidb_tools, MCPTools(server_params=o11y_server_params,include_tools=["list_datasources","query_prometheus","list_prometheus_metric_metadata","list_prometheus_metric_names","list_prometheus_label_names","list_prometheus_label_values","query_loki_logs","list_loki_label_names","list_loki_label_values","get_datasource_by_name"]) as o11y_tools:
        # Initialize the workflow
        workflow = SQLOptimizationWorkflow(
            workflow_id="sql_optimization_workflow",
            storage=SqliteStorage(
                table_name="sql_optimization_workflow",
                db_file="tmp/agno_workflows.db",
            ),
            debug_mode=True,
        )
        workflow.setup(kb=knowledge_base,tidb_tools=tidb_tools,o11y_tools=o11y_tools)

        app = Playground(
            workflows=[
                workflow,
            ]
        ).get_app()

        serve_playground_app(app)


if __name__ == "__main__":
    # load knowledge base, only used in first run
    # knowledge_base.load(recreate=True)
    asyncio.run(run_workflow())
