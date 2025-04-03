import os
import asyncio
from datetime import datetime
import time
from textwrap import dedent
from typing import Dict, Iterator, List, Optional, AsyncIterator
import json

from agno.agent import Agent
from agno.models.openai import OpenAILike, OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.tools.mcp import MCPTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import RunEvent, RunResponse, Workflow
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class SlowQueryInfo(BaseModel):
    """Information about a slow query extracted from slowlog data."""
    query_digest: str = Field(..., description="The digest (fingerprint) of the query.")
    plan_digest: Optional[str] = Field(None, description="The digest of the execution plan.")
    query_time: float = Field(..., description="Total execution time of the query in seconds.")
    db: Optional[str] = Field(None, description="Database name.")
    process_keys: Optional[int] = Field(None, description="Number of keys processed during execution.")
    total_keys: Optional[int] = Field(None, description="Total number of keys involved.")
    mem_max: Optional[int] = Field(None, description="Maximum memory usage during execution in bytes.")
    wait_time: Optional[float] = Field(None, description="Total time spent waiting during execution.")
    backoff_time: Optional[float] = Field(None, description="Time spent in backoff during execution.")
    result_rows: Optional[int] = Field(None, description="Number of rows in the query result.")
    index_names: Optional[str] = Field(None, description="Names of indexes used in the query.")
    plan: Optional[str] = Field(None, description="decoded execution plan of the query.")
    timestamp: Optional[int] = Field(None, description="Timestamp when the query was executed.")
    execution_count: Optional[int] = Field(None, description="Number of times this query pattern was executed.")
    sql_text: Optional[str] = Field(None, description="The actual SQL query text if available.")


class SlowQueryAnalysisResults(BaseModel):
    """Analysis results from the Slowlog Analyzer agent."""
    time_range: str = Field(..., description="Time range of the analysis.")
    top_queries: List[SlowQueryInfo] = Field(..., description="List of top slow queries identified.")
    pattern_analysis: Optional[str] = Field(None, description="Analysis of query patterns and trends.")
    overall_statistics: Optional[str] = Field(None, description="Overall statistics of slow queries in the time range.")
    # recommendations: Optional[str] = Field(None, description="Initial recommendations for query optimization.")


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

    # Slowlog Analyzer Agent: Analyzes slow query logs to identify top problematic queries
    slowlog_analyzer: Agent = Agent(
        model=OpenAILike(
                base_url="https://api.novita.ai/v3/openai",
                api_key=os.getenv("NOVITA_API_KEY"),
                id="deepseek/deepseek-v3-0324"
            ),
        tools=[],  # Tools will be added at runtime
        description=dedent("""\
        You are a TiDB slow query log analysis expert.
        You excel at identifying patterns in slow query logs, clustering similar queries,
        and extracting the most important slow queries that need attention.
        """),
        instructions=dedent("""\
        1. Retrieve slow query logs for the specified time range with LogQL '{container="slowlog"}', **keep in mind that do not change the LogQL query**.
        2. Analyze patterns and characteristics in the slow query logs.
        3. Identify and filter the Top N slow queries most worth optimizing, you could write code to do the analysis if needed.
        4. Provide preliminary diagnostic analysis for each slow query

        Your analysis should focus on:
        - Queries with long execution times (high Query_time)
        - Frequently executed queries (high Count)
        - Queries with high memory usage (high Mem_max)
        - Queries that scan or process large amounts of data (high Process_keys, Total_keys)
        - Queries involving long wait times (high Wait_time, Backoff_time)

        Focus on producing accurate structured data for the top slow queries with all the required fields.
        """),
        response_model=SlowQueryAnalysisResults,
        use_json_mode=True,
        markdown=True,
        # reasoning=True,
        show_tool_calls=True,
        debug_mode=True,
        add_datetime_to_instructions=True,
    )

    # SQL Optimizer Agent: Generates optimization recommendations for slow queries
    sql_optimizer: Agent = Agent(
        model=OpenAILike(
            base_url="https://api.novita.ai/v3/openai",
            api_key=os.getenv("NOVITA_API_KEY"),
            id="deepseek/deepseek-v3-0324",
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
           - Examine table schemas, statistics, execution plans
           - Search for relevant TiDB optimization techniques

        3. Observation: Analyze the results of your actions and compare it with the original execution plan.

        4. Repeat steps 1-3 until you have enough information to provide recommendations

        Throughout the process:
        - Analyze slow query information to identify bottlenecks
        - Gather context information methodically
        - Use web search for up-to-date TiDB optimization techniques
        - Generate clear optimization recommendations
        - Execute optimizations when approved and evaluate results

        """),
        expected_output=dedent("""\
        # SQL Optimization Report

        ## Analysis Time
        {current_datetime}

        ## Query Analysis
        {query_analysis_with_identified_issues}

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

    async def start(
        self,
        time_range: str = "24h",
        top_n: int = 5,
        auto_apply_optimizations: bool = False,
        use_cached_analysis: bool = True,
    ) -> AsyncIterator[RunResponse]:
        """Run the SQL Optimization Workflow."""
        logger.info(f"Starting SQL Optimization Workflow for time range: {time_range}")
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

        async with MCPTools(server_params=tidb_server_params) as tidb_tools, MCPTools(server_params=o11y_server_params) as o11y_tools:

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

                # Configure tools for slowlog analyzer
                self.slowlog_analyzer.tools = [o11y_tools]

                # Run the slowlog analyzer
                slowlog_message = f"Current time is {datetime.now().astimezone()}, Analyze the top {top_n} slowest queries from the past {time_range} and provide structured output."
                slowlog_response = await self.slowlog_analyzer.arun(slowlog_message)
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
                self.sql_optimizer.tools = [tidb_tools]

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
                {'' if not auto_apply_optimizations else 'Apply the optimizations if they are safe to apply.'}
                """

                optimizer_response = await self.sql_optimizer.arun(optimizer_message)
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

            yield RunResponse(content=summary, event=RunEvent.workflow_completed)


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

    # Initialize the workflow
    workflow = SQLOptimizationWorkflow(
        session_id=session_id,
        storage=SqliteStorage(
            table_name="sql_optimization_workflows",
            db_file="tmp/agno_workflows.db",
        ),
        debug_mode=True,
    )

    # run workflow
    async for response in workflow.start(
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
    load_dotenv()
    # set timezone to UTC+8 for datetime lib
    os.environ["TZ"] = "Asia/Shanghai"
    time.tzset()

    parser = argparse.ArgumentParser(description="Run SQL Optimization Workflow")
    parser.add_argument("--time-range", type=str, default="12h", help="Time range to analyze (e.g., '24h', '7d')")
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

