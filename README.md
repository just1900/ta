# TiDB SQL Optimization Agent System

An AI Agent-based system for analyzing and optimizing slow queries in TiDB databases. This system automates the analysis of slow query logs in TiDB databases, identifies queries that need optimization, and generates optimization recommendations or automatically applies optimizations.

## System Architecture

The system consists of three main components:

1. **Slow Query Log Analyzer Agent (SlowlogAnalyzerAgent)**: Analyzes slow query logs, identifies patterns and key performance bottleneck queries
2. **SQL Optimization Agent (SQLOptimizerAgent)**: Generates optimization recommendations based on analysis results and optionally executes optimizations
3. **TiDB Optimization System (TiDBOptimizerSystem)**: Coordinates the above two agents to implement a complete analysis and optimization workflow

The system uses [Agno](https://docs.agno.com/agents/) as the Agent development framework, combined with MCP (Model Context Protocol) servers to interact with TiDB-related components.

## Features

- Automatically analyzes slow query logs and identifies key problem queries
- Gathers and analyzes table structures, statistics, and execution plans
- Generates specific, actionable optimization recommendations (such as index creation, statistics updates, SQL rewrites)
- Optionally executes optimization operations automatically and evaluates their effectiveness
- Generates detailed analysis and optimization reports

## Installation and Dependencies

```bash
# Install required dependencies
pip install agno-ai mcp openai

# Install MCP server related components
npm install -g uvx
```

## Usage

### 1. Running the Slow Query Analyzer

```python
from slowlog_analyzer_agent import run_slowlog_analyzer
import asyncio

# Analyze slow query logs from the last 24 hours
asyncio.run(
    run_slowlog_analyzer(
        "Analyze slow query logs from the last 24 hours and identify the top 10 slowest queries with preliminary analysis"
    )
)
```

### 2. Running the SQL Optimizer

```python
from sql_optimizer_agent import run_sql_optimizer
import asyncio

# Optimize a specific query
asyncio.run(
    run_sql_optimizer(
        "Please analyze and optimize the performance of this query: SELECT * FROM orders WHERE create_date > '2023-01-01' AND status = 'processing'"
    )
)
```

### 3. Running the Complete Optimization Workflow

```python
from tidb_optimizer_system import TiDBOptimizerSystem
import asyncio

# Run the complete analysis and optimization workflow
asyncio.run(
    TiDBOptimizerSystem.analyze_and_optimize(
        time_range_hours=24,  # Analyze the past 24 hours
        top_n=5,              # Analyze the top 5 problem queries
        auto_optimize=False   # Set to True to automatically execute optimizations
    )
)
```

## MCP Server Requirements

The system requires the following MCP server components:

1. `mcp-server-tidb-slowlog`: For retrieving and analyzing TiDB slow query logs
2. `mcp-server-tidb-schema`: For retrieving table structure information
3. `mcp-server-tidb-stats`: For retrieving statistics information
4. `mcp-server-prometheus-metrics`: For retrieving Prometheus metric data
5. `mcp-server-tidb-execution`: For executing SQL queries and optimization operations

## Notes

- Automatic optimization functionality (`auto_optimize=True`) should be used with caution; it's recommended to verify in a test environment before applying to production
- Optimization recommendations may require manual review, especially recommendations involving table structure changes
- The system depends on the availability and correct configuration of MCP servers
