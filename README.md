# TiDB SQL Optimization Agent

An intelligent AI agent system for analyzing and optimizing slow queries in TiDB clusters. This project aims to automate the process of SQL query optimization using AI-driven analysis and recommendations.

## Overview

The TiDB SQL Optimization Agent is designed to address the challenges of managing slow SQL queries in large-scale TiDB deployments. It automatically analyzes slow query logs, identifies performance bottlenecks, and provides optimization suggestions or automatic optimizations when possible.

### Key Features

- Automated slow query log analysis
- AI-driven SQL optimization recommendations
- Integration with TiDB observability data
- Automatic optimization execution (with safeguards)
- Performance impact analysis and rollback capabilities
- Structured optimization reports

## Architecture

The system consists of several key components:

1. **AI Agent**: Implements the ReAct (Reasoning + Acting) paradigm for intelligent decision-making
2. **MCP Server**: Model Context Protocol server that provides standardized interfaces to various data sources
3. **Slow Query Analysis**: Processes and analyzes slow query logs from S3 storage
4. **Optimization Engine**: Generates and validates SQL optimization suggestions

## Prerequisites

- Python 3.10 or higher
- TiDB cluster with slow query logging enabled
- Access to required data sources (S3, Prometheus metrics, etc.)
- Necessary permissions for SQL optimization operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tidb-optimizer.git
cd tidb-optimizer
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
uv pip install -e .
```

## Configuration

Create a `.env` file from example

## Usage

```bash
uv run src/demo.py
```

## Project Structure

```
.
├── src/              # Source code
├── sql/              # SQL-related resources
├── tmp/              # Temporary files
├── design.md         # Detailed design documentation
├── pyproject.toml    # Project configuration and dependencies
├── uv.lock           # Lock file for dependency versions
└── README.md         # This file
```

## Dependencies

Key dependencies include:
- mcp >= 0.3.0
- openai >= 1.70.0
- fastapi >= 0.103.1
- sqlalchemy >= 2.0.40
- agno >= 1.2.6
- [See pyproject.toml for complete list]

## Development

[Development instructions will be added based on contribution guidelines]
