import asyncio
import os
from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters


class SQLOptimizerAgent:
    """TiDB SQL Optimization Agent for analyzing and optimizing slow queries"""

    @staticmethod
    def create_agent():
        """Create SQL Optimization Agent"""
        return Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[],  # Tools will be added at runtime
            description=dedent("""\
                You are a TiDB Database Administrator (DBA) and performance tuning expert.
                You excel at analyzing slow query logs, identifying performance bottlenecks,
                and providing concrete, actionable optimization recommendations.
            """),
            instructions=dedent("""\
                1. Analyze the provided TiDB slow query information to identify potential bottlenecks
                2. Gather necessary context information (table schemas, statistics, execution plans, etc.)
                3. Generate optimization recommendations (such as index creation, statistics updates, SQL rewrites)
                4. Execute optimizations after approval and evaluate results

                Your analysis should focus on:
                - Missing indexes
                - Outdated statistics
                - Inefficient query patterns and execution plans
                - Operations with high memory usage
                - Operations scanning large numbers of keys or rows
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
            """),
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )


async def run_sql_optimizer(message: str) -> None:
    """Run SQL Optimization Agent"""

    # Configure TiDB related MCP server parameters
    slowlog_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-tidb-slowlog"],
    )

    schema_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-tidb-schema"],
    )

    stats_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-tidb-stats"],
    )

    metrics_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-prometheus-metrics"],
    )

    execution_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-tidb-execution"],
    )

    # Create Agent and load MCP tools
    async with (
        MCPTools(server_params=slowlog_server_params) as slowlog_tools,
        MCPTools(server_params=schema_server_params) as schema_tools,
        MCPTools(server_params=stats_server_params) as stats_tools,
        MCPTools(server_params=metrics_server_params) as metrics_tools,
        MCPTools(server_params=execution_server_params) as execution_tools
    ):
        agent = SQLOptimizerAgent.create_agent()
        agent.tools = [slowlog_tools, schema_tools, stats_tools, metrics_tools, execution_tools]

        await agent.aprint_response(message, stream=True)


# Usage examples
if __name__ == "__main__":
    # Example for optimizing a specific query
    asyncio.run(
        run_sql_optimizer(
            "Please analyze and optimize the performance of this query: SELECT * FROM orders WHERE create_date > '2023-01-01' AND status = 'processing'"
        )
    )

    # Other possible example usages:
    # asyncio.run(
    #     run_sql_optimizer(
    #         "Analyze the 5 slowest queries from the past 24 hours and provide optimization recommendations"
    #     )
    # )
