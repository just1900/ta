"""Slowlog Analyzer Agent for TiDB."""

import os
import asyncio
from textwrap import dedent
from typing import Optional
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat, OpenAILike
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters


class SlowlogAnalyzerAgent:
    """TiDB Slow Query Log Analyzer Agent for analyzing patterns in slow query logs."""

    @staticmethod
    def create_agent(
        name: Optional[str] = None,
        storage: Optional[SqliteAgentStorage] = None,
        add_history: bool = True,
        num_history_responses: int = 5,
    ) -> Agent:
        """
        Create Slow Query Log Analyzer Agent.

        Args:
            name: Custom name for the agent
            storage: Storage instance for preserving agent state
            add_history: Whether to add conversation history
            num_history_responses: Number of history responses to include

        Returns:
            Configured Agent instance
        """
        return Agent(
            name=name or "TiDB Slow Query Log Analyzer",
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
                1. Retrieve slow query log statistics for the specified time range with <list_datasources> and <query_loki_logs> with LogQL '{container="slowlog"}',
                2. Analyze patterns and characteristics in the slow query logs
                3. Identify and filter the Top N slow queries most worth optimizing
                4. Provide preliminary diagnostic analysis for each slow query

                Your analysis should focus on:
                - Queries with long execution times (high Query_time)
                - Frequently executed queries (high Count)
                - Queries with high memory usage (high Mem_max)
                - Queries that scan or process large amounts of data (high Process_keys, Total_keys)
                - Queries involving long wait times (high Wait_time, Backoff_time)

                Always proceed until you have the final report.
            """),
            expected_output=dedent("""\
                # TiDB Slow Query Analysis Report

                ## Analysis Time
                {current_datetime}

                ## Analysis Scope
                - Time Range: {time_range}
                - Filter Conditions: {filter_conditions}

                ## Slow Query Statistics Overview
                {overall_statistics}

                ## Top {N} Slow Queries
                {table_of_top_queries_with_metrics}

                ## Query Pattern Analysis
                {pattern_analysis}

                ## Next Steps Recommendations
                {recommendations_for_further_analysis}
            """),
            storage=storage,
            markdown=True,
            reasoning=True,
            debug_mode=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            add_history_to_messages=add_history,
            num_history_responses=num_history_responses,
        )


async def run_slowlog_analyzer(message: str) -> None:
    """Run Slow Query Log Analyzer Agent."""
    # Configure TiDB slow query log related MCP server parameters
    slowlog_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-tidb-slowlog"],
    )

    o11y_server_params = StdioServerParameters(
            command="mcp-grafana",
            env={
                "GRAFANA_URL": os.getenv("GRAFANA_URL"),
                "GRAFANA_API_KEY": os.getenv("GRAFANA_API_KEY"),
            }
        )


    # Create Agent and load MCP tools
    async with (
        # MCPTools(server_params=slowlog_server_params) as slowlog_tools,
        MCPTools(server_params=o11y_server_params) as metrics_tools
    ):
        agent = SlowlogAnalyzerAgent.create_agent()
        agent.tools = [metrics_tools]

        # Run the agent to analyze slow queries
        await agent.aprint_response(message, stream=True)

# add main function
if __name__ == "__main__":
    load_dotenv()
    asyncio.run(run_slowlog_analyzer("Show me the slow query logs for last 1 days"))
