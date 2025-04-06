import os
import asyncio
from datetime import datetime
from textwrap import dedent
from typing import Dict, Iterator, List, Optional, AsyncIterator
import json

from agno.agent import Agent
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.models.openai import OpenAILike, OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.storage.sqlite import SqliteStorage
from agno.tools.mcp import MCPTools
from agno.tools.thinking import ThinkingTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import RunEvent, RunResponse, Workflow
from agno.utils.log import logger
from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agno.knowledge.agent import AgentKnowledge
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.openai import OpenAIEmbedder
from pathlib import Path
import nest_asyncio

load_dotenv()
nest_asyncio.apply()

class SlowQueryInfo(BaseModel):
    """Information about a slow query extracted from slowlog data."""
    db: Optional[str] = Field(None, description="Database name.")
    query_time: float = Field(..., description="Total execution time of the query in seconds.")
    sql_text: Optional[str] = Field(None, description="The actual SQL query text.")
    index_names: Optional[str] = Field(None, description="Names of indexes used in the query.")
    plan: Optional[str] = Field(None, description="Execution plan of the query.")
    time: Optional[int] = Field(None, description="Time when the query was executed.")


class SlowQueryAnalysisResults(BaseModel):
    """Analysis results from the Slowlog Analyzer agent."""
    time_range: str = Field(..., description="Time range of the analysis.")
    top_queries: List[SlowQueryInfo] = Field(..., description="List of top slow queries identified.")
    pattern_analysis: Optional[str] = Field(None, description="Analysis of query patterns and trends.")
    recommendations: Optional[str] = Field(None, description="Initial recommendations for query optimization.")


class OptimizationResults(BaseModel):
    """Results from the SQL Optimizer agent."""
    query_digest: str = Field(..., description="The digest of the optimized query.")
    original_query_time: float = Field(..., description="Original execution time before optimization.")
    optimized_query_time: Optional[float] = Field(None, description="Execution time after optimization.")
    optimization_recommendations: List[str] = Field(..., description="List of optimization recommendations.")
    applied_optimizations: List[str] = Field(default_factory=list, description="List of optimizations that were applied.")
    before_after_comparison: Optional[str] = Field(None, description="Comparison of performance before and after optimization.")
    optimized_plan: Optional[str] = Field(None, description="The optimized execution plan if available.")

class SQLOptimizationWorkflow(Workflow):
    """Workflow that pipelines Slowlog Analyzer and SQL Optimizer agents for TiDB query optimization."""

    description: str = dedent("""\
    An intelligent SQL optimization workflow for TiDB that automatically analyzes slow query logs,
    identifies performance bottlenecks, and generates optimization recommendations.
    This workflow combines a Slowlog Analyzer agent that identifies problematic query patterns
    with a SQL Optimizer agent that provides concrete optimization recommendations.
    """)

    tidb_tools: MCPTools = None
    o11y_tools: MCPTools = None
    kb: AgentKnowledge = None

    # Slowlog Analyzer Agent: Analyzes slow query logs to identify top problematic queries
    slowlog_analyzer: Agent = Agent(
        name="Slowlog Analyzer",
        model=OpenAIChat(
            id="o3-mini",
        ),
        # reasoning=True,
        tools=[],  # Will be set at runtime with tidb_tools
        description=dedent("""\
        You are a TiDB slow query log analysis expert.
        You excel at identifying patterns in slow query logs, clustering similar queries,
        and extracting the most important slow queries that need attention.
        """),
        instructions=dedent("""\
        Your task is to analyze slow query logs from TiDB's information_schema.cluster_slow_query table to identify the most problematic queries that need optimization.

        ## Key Steps
        1. Check the schema of the cluster_slow_query table via <show_create_table> to understand the columns

        2. Query cluster_slow_query to get slow query data
           - Filter by time range
           - Sort by query_time to find the slowest queries

        3. Identify the top slow queries worth optimizing based on:
           - Query execution time
           - Memory usage
           - Wait time
           - Execution count (look for frequently occurring patterns with the same query_digest)
           - Plan variability (check if the same query has multiple different execution plans)

        4. Group similar queries by query_digest and analyze patterns
           - Look for queries that appear multiple times with high cumulative cost
           - Identify common tables or operations in problematic queries

        5. Examine execution plans to identify optimization opportunities
           - Look for full table scans, missing indexes, and suboptimal join methods
           - Pay attention to the number of rows being processed vs. returned

        """),
        response_model=SlowQueryAnalysisResults,
        use_json_mode=True,
        debug_mode=True,
        show_tool_calls=True,
        markdown=True,
    )

    # SQL Optimizer Agent: Generates optimization recommendations for slow queries
    sql_optimizer: Agent = Agent(
        model=OpenAIChat(
            id="gpt-4o",
        ),
        tools=[],  # Tools will be added at runtime
        description=dedent("""\
        You are a TiDB Database Administrator (DBA) and performance tuning expert.
        You excel at analyzing slow query logs, identifying performance bottlenecks,
        and providing concrete, actionable optimization recommendations.
        """),
        instructions=dedent("""\
        Follow the ReAct (Reasoning + Acting) framework to solve problems:

        1. Thought: First, clearly think through the problem and plan your approach.
           Reason step-by-step about what information you need and how to get it.

        2. Action: Execute your planned actions using available tools:
           - Query the database to gather necessary information
             * Table schemas, indexes, and statistics
             * Table sizes and cardinality
             * Actual execution plans
           - Fetch relevant metrics from the monitoring system
             * CPU/Memory utilization during query execution
             * Storage I/O patterns
             * Transaction throughput and latency
             * Connection pool status
           - Search for relevant TiDB optimization techniques

        3. Observation: Analyze the results of your actions and compare it with the original execution plan.

        4. Repeat steps 1-3 until you have enough information to provide recommendations

        ## Metrics to Consider
        When optimizing queries, consider fetching these metrics for deeper insights:
        - Query execution metrics: QPS, latency percentiles, execution durations
        - Resource utilization: CPU, memory, storage I/O for TiDB and TiKV nodes
        - Transaction metrics: Transaction per second, commit latency
        - Table-specific metrics: DML operations, scan operations, lock contention
        - Index usage statistics: Hit ratios, inefficient indexes

        ## Optimization Approaches
        Consider these optimization techniques based on your findings:
        - Schema improvements (indexing strategies, column types, partitioning)
        - Query rewrites (simplification, join optimizations, subquery elimination)
        - Parameter tuning (memory allocations, work_mem, execution parallelism)
        - Caching strategies (prepare statements, connection pooling)
        - Table statistics updates
        - Plan binding for stabilizing execution plans with high variability

        Throughout the process:
        - Analyze slow query information to identify bottlenecks
        - Gather context information methodically including relevant metrics
        - Generate clear optimization recommendations with expected impact
        - Execute optimizations when approved and evaluate results
        """),
        expected_output=dedent("""\
        # SQL Optimization Report

        ## Analysis Time
        {current_datetime}

        ## Query Analysis
        {query_analysis_with_identified_issues}

        ## Related Metrics
        {relevant_metrics_collected}

        ## Optimization Recommendations
        {prioritized_list_of_recommendations}

        ## Expected Impact
        {expected_impact_of_recommendations}

        ## Execution Results
        {execution_results_if_applied}

        ## Before/After Comparison
        {performance_comparison_if_available}
        {detailed_execution_plan_before_and_after_optimization}

        ## References
        {references_to_tidb_documentation_and_best_practices}
        """),
        markdown=True,
        show_tool_calls=True,
        # reasoning=True,

        debug_mode=True,
        add_datetime_to_instructions=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self,kb: AgentKnowledge,tidb_tools: MCPTools,o11y_tools: MCPTools):
        self.kb = kb
        self.tidb_tools = tidb_tools
        self.o11y_tools = o11y_tools

    def run(
        self,
        time_range: str = "24h",
        top_n: int = 5,
        auto_apply_optimizations: bool = False,
        use_cached_analysis: bool = True,
    ) -> Iterator[RunResponse]:
        """Run the SQL Optimization Workflow."""
        logger.info(f"Starting SQL Optimization Workflow for time range: {time_range}")

        # if self.kb is not None:
        #     self.slowlog_analyzer.knowledge = self.kb
        #     self.sql_optimizer.knowledge = self.kb

        # Set the tools for the slowlog analyzer
        self.slowlog_analyzer.tools = [self.tidb_tools]

        # Check cache for analysis results
        analysis_results = None
        if use_cached_analysis:
            cached_analysis = self.get_cached_analysis(time_range)
            if cached_analysis:
                yield RunResponse(
                    content=f"Using cached analysis results for time range: {time_range}",
                    event=RunEvent.reasoning_started
                )
                analysis_results = cached_analysis

        if not analysis_results:
            # Run slowlog analyzer and get analysis results
            yield RunResponse(
                content=f"Analyzing slow queries for time range: {time_range}",
                event=RunEvent.reasoning_started
            )

            # Run the slowlog analyzer
            slowlog_message = f"""
            Current time is {datetime.now().astimezone()}.
            Please analyze the top {top_n} slowest queries from the past {time_range} by querying the
            information_schema.cluster_slow_query table.

            Look for patterns and group similar queries to provide a comprehensive analysis.
            Focus on queries with the highest execution time, memory usage, and key processing counts.
            """
            slowlog_response = self.slowlog_analyzer.run(slowlog_message)
            yield slowlog_response

            if not slowlog_response or not isinstance(slowlog_response.content, SlowQueryAnalysisResults):
                yield RunResponse(
                    content="Failed to analyze slow queries.",
                    event=RunEvent.workflow_completed
                )
                return

            analysis_results = slowlog_response.content
            self.add_analysis_to_cache(time_range, analysis_results)

        # Process each slow query with the SQL optimizer
        optimization_results = []

        for query_info in analysis_results.top_queries[:top_n]:
            yield RunResponse(
                content=f"Optimizing query with digest: {query_info.query_digest}",
                event=RunEvent.reasoning_started
            )

            # Configure tools for SQL optimizer
            self.sql_optimizer.tools = [self.tidb_tools, self.o11y_tools, DuckDuckGoTools()]

            # Prepare input for SQL optimizer
            optimizer_input = {
                "query_info": query_info.model_dump(),
                "auto_apply": auto_apply_optimizations
            }

            # Run the SQL optimizer
            optimizer_message = f"""
            Please analyze and optimize the following slow query:

            {json.dumps(optimizer_input, indent=2)}

            Provide detailed optimization recommendations and explain the expected impact of each recommendation.
            Fetch relevant metrics from the monitoring system if needed to understand the query's performance context.
            {'' if not auto_apply_optimizations else 'Apply the optimizations if they are safe to apply.'}
            """

            optimizer_response = self.sql_optimizer.run(optimizer_message)
            if optimizer_response:
                yield optimizer_response

            # Parse optimization results if available
            if optimizer_response and hasattr(optimizer_response, 'content'):
                optimization_result = OptimizationResults(
                    query_digest=query_info.query_digest,
                    original_query_time=query_info.query_time,
                    optimization_recommendations=["Recommendations extracted from optimizer output"]
                )
                optimization_results.append(optimization_result)

        # Generate final summary
        summary = f"""
        # SQL Optimization Workflow Summary

        ## Analysis Time
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        ## Analysis Scope
        - Time Range: {time_range}
        - Top Queries Analyzed: {len(optimization_results)}

        ## Optimization Results
        {self.format_optimization_results(optimization_results)}

        ## Next Steps
        - Review and apply the recommended optimizations
        - Monitor query performance after applying optimizations
        - Consider scheduling regular optimization runs
        """

        yield RunResponse(content="summary", event=RunEvent.workflow_completed)


    def get_cached_analysis(self, time_range: str) -> Optional[SlowQueryAnalysisResults]:
        """Get cached analysis results for the given time range."""
        logger.info("Checking if cached analysis results exist")
        analysis = self.session_state.get("analysis_results", {}).get(time_range)
        return (
            SlowQueryAnalysisResults.model_validate(analysis)
            if analysis and isinstance(analysis, dict)
            else None
        )

    def add_analysis_to_cache(self, time_range: str, analysis_results: SlowQueryAnalysisResults):
        """Add analysis results to cache."""
        logger.info(f"Saving analysis results for time range: {time_range}")
        self.session_state.setdefault("analysis_results", {})
        self.session_state["analysis_results"][time_range] = analysis_results.model_dump()

    def format_optimization_results(self, optimization_results: List[OptimizationResults]) -> str:
        """Format optimization results into a markdown table."""
        if not optimization_results:
            return "No optimization results available."

        result = "| Query Digest | Original Time | Optimized Time | Recommendations |\n"
        result += "|-------------|--------------|---------------|----------------|\n"

        for opt in optimization_results:
            digest = opt.query_digest[:10] + "..." if len(opt.query_digest) > 13 else opt.query_digest
            orig_time = f"{opt.original_query_time:.4f}s"
            opt_time = f"{opt.optimized_query_time:.4f}s" if opt.optimized_query_time else "N/A"
            recs = ", ".join(opt.optimization_recommendations[:2]) + ("..." if len(opt.optimization_recommendations) > 2 else "")

            result += f"| {digest} | {orig_time} | {opt_time} | {recs} |\n"

        return result


# Helper function to run the workflow
async def run_sql_optimization_workflow(
    time_range: str = "1h",
    top_n: int = 10,
    auto_apply: bool = False,
    use_cache: bool = True
) -> None:
    """Run the SQL Optimization Workflow with the given parameters."""
    # Create a unique session ID based on parameters
    session_id = f"sql-optimization-{time_range}-{top_n}-{datetime.now().strftime('%Y%m%d')}"

    logger.info("setup tools")
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

    async with MCPTools(server_params=tidb_server_params,include_tools=["db_execute","db_query","show_create_table"]) as tidb_tools, MCPTools(server_params=o11y_server_params,include_tools=["list_datasources","query_prometheus","list_prometheus_metric_metadata","list_prometheus_metric_names","list_prometheus_label_names","list_prometheus_label_values","query_loki_logs","list_loki_label_names","list_loki_label_values","get_datasource_by_name"]) as o11y_tools:
        # Initialize the workflow
        workflow = SQLOptimizationWorkflow(
            session_id=session_id,
            storage=SqliteStorage(
                table_name="sql_optimization_workflows",
                db_file="tmp/agno_workflows.db",
            ),
            debug_mode=True,
        )
        workflow.setup(kb=knowledge_base,tidb_tools=tidb_tools,o11y_tools=o11y_tools)

        # run workflow
        for response in workflow.run(
            time_range=time_range,
            top_n=top_n,
            auto_apply_optimizations=auto_apply,
            use_cached_analysis=use_cache,
        ):
            if hasattr(response, 'content'):
                print(response.content)

# Run the workflow if the script is executed directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SQL Optimization Workflow")
    parser.add_argument("--time-range", type=str, default="24h", help="Time range to analyze (e.g., '24h', '7d')")
    parser.add_argument("--top-n", type=int, default=2, help="Number of top slow queries to analyze")
    parser.add_argument("--auto-apply", action="store_true", help="Automatically apply optimizations")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached analysis results")

    args = parser.parse_args()

    asyncio.run(run_sql_optimization_workflow(
      time_range=args.time_range,
      top_n=args.top_n,
      auto_apply=args.auto_apply,
      use_cache=not args.no_cache
    ))

