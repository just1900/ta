import asyncio
import os
from datetime import datetime
from textwrap import dedent
from dotenv import load_dotenv  # Import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.tools.mcp import MCPTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.agent.sqlite import SqliteAgentStorage
from mcp import StdioServerParameters

# Load environment variables from .env file
load_dotenv()

class SQLOptimizerAgent:
    """TiDB SQL Optimization Agent for analyzing and optimizing slow queries"""

    @staticmethod
    def create_agent(name=None, storage=None, add_history=True, num_history_responses=5):
        """
        Create SQL Optimization Agent

        Args:
            name: Custom name for the agent
            storage: Storage instance for preserving agent state
            add_history: Whether to add conversation history
            num_history_responses: Number of history responses to include

        Returns:
            Configured Agent instance
        """
        return Agent(
            name=name or "TiDB SQL Optimizer",
            model=OpenAILike(
                base_url="https://api.novita.ai/v3/openai",
                api_key=os.getenv("NOVITA_API_KEY"),
                id="deepseek/deepseek-v3-0324"
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

                3. Observation: Analyze the results of your actions

                4. Repeat steps 1-3 until you have enough information to provide recommendations

                Throughout the process:
                - Analyze slow query information to identify bottlenecks
                - Gather context information methodically
                - Use web search for up-to-date TiDB optimization techniques
                - Generate clear optimization recommendations
                - Execute optimizations when approved and evaluate results

                Focus your analysis on:
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

                ## References
                {references_to_tidb_documentation_and_best_practices}
            """),
            storage=storage,
            markdown=True,
            show_tool_calls=True,
            debug_mode=True,
            add_datetime_to_instructions=True,
            add_history_to_messages=add_history,
            num_history_responses=num_history_responses,
        )


async def run_sql_optimizer(message: str) -> None:
    """Run SQL Optimization Agent"""

    # Configure TiDB related MCP server parameters using environment variables
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

    # Configure specialized servers as separate MCP servers
    # metrics_server_params = StdioServerParameters(
    #     command="mcp-grafana",
    #     args=[],
    #     env={
    #         "GRAFANA_URL": os.getenv("GRAFANA_URL"),
    #         "GRAFANA_API_KEY": os.getenv("GRAFANA_API_KEY")
    #     }
    # )

    # Create web search tools
    web_search_tools = DuckDuckGoTools()

    # Create Agent and load MCP tools
    async with (
        MCPTools(server_params=tidb_server_params) as tidb_tools,
        # MCPTools(server_params=metrics_server_params) as metrics_tools
    ):
        agent = SQLOptimizerAgent.create_agent()
        # Add web search tools along with MCP tools
        agent.tools = [tidb_tools, web_search_tools]

        # Process the message, enriching it with a prompt to utilize web search when relevant
        enriched_message = f"""
{message}

As you analyze this query, please follow the ReAct framework:

1. Thought: Start by reasoning about what information you need to understand the query performance issues.
2. Action: Take specific actions like examining table schemas, statistics, or execution plans.
3. Observation: Analyze what you learned from each action.
4. Repeat until you can provide comprehensive optimization recommendations.

If you need additional information about TiDB optimization best practices, query syntax,
or recent performance tuning techniques, please use web search to find the most up-to-date information.
"""
        await agent.aprint_response(enriched_message, stream=True)


# Usage examples
if __name__ == "__main__":
    # Example for optimizing a specific query
    asyncio.run(
        run_sql_optimizer(
            "Please analyze and optimize the performance of this query: SELECT u.name, u.city, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count, (SELECT SUM(od.quantity) FROM order_details od JOIN orders o ON od.order_id = o.id WHERE o.user_id = u.id) as total_quantity FROM users u WHERE u.city = 'New York';"
        )
    )

    # Other possible example usages:
    # asyncio.run(
    #     run_sql_optimizer(
    #         "Analyze the 5 slowest queries from the past 24 hours and provide optimization recommendations"
    #     )
    # )
