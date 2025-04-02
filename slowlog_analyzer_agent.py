import asyncio
import os
from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters


class SlowlogAnalyzerAgent:
    """TiDB Slow Query Log Analyzer Agent for analyzing patterns in slow query logs"""

    @staticmethod
    def create_agent():
        """Create Slow Query Log Analyzer Agent"""
        return Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[],  # Tools will be added at runtime
            description=dedent("""\
                You are a TiDB slow query log analysis expert.
                You excel at identifying patterns in slow query logs, clustering similar queries,
                and extracting the most important slow queries that need attention.
            """),
            instructions=dedent("""\
                1. Retrieve slow query log statistics for the specified time range from the MCP server
                2. Analyze patterns and characteristics in the slow query logs
                3. Identify and filter the Top N slow queries most worth optimizing
                4. Provide preliminary diagnostic analysis for each slow query

                Your analysis should focus on:
                - Queries with long execution times (high Query_time)
                - Frequently executed queries (high Count)
                - Queries with high memory usage (high Mem_max)
                - Queries that scan or process large amounts of data (high Process_keys, Total_keys)
                - Queries involving long wait times (high Wait_time, Backoff_time)
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
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )


async def run_slowlog_analyzer(message: str) -> None:
    """Run Slow Query Log Analyzer Agent"""

    # Configure TiDB slow query log related MCP server parameters
    slowlog_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-tidb-slowlog"],
    )

    metrics_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-prometheus-metrics"],
    )

    # Create Agent and load MCP tools
    async with (
        MCPTools(server_params=slowlog_server_params) as slowlog_tools,
        MCPTools(server_params=metrics_server_params) as metrics_tools
    ):
        agent = SlowlogAnalyzerAgent.create_agent()
        agent.tools = [slowlog_tools, metrics_tools]

        await agent.aprint_response(message, stream=True)


# Usage examples
if __name__ == "__main__":
    # Analyze slow query logs from the last 24 hours
    asyncio.run(
        run_slowlog_analyzer(
            "Analyze slow query logs from the last 24 hours and identify the top 10 slowest queries with preliminary analysis"
        )
    )

    # Other possible example usages:
    # asyncio.run(
    #     run_slowlog_analyzer(
    #         "Analyze the most frequently executed slow queries between 2023-06-01 and 2023-06-07, grouped by database"
    #     )
    # )
