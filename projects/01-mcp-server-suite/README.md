# MCP Server Suite

**Production-ready Model Context Protocol servers enabling LLM tool use.**

## Overview

This project implements 8 specialized MCP servers that provide LLMs with structured access to external systems and data sources. Each server handles a specific domain, maintaining clear separation of concerns and independent deployment.

## Architecture

### Why Multiple Servers?

Instead of a monolithic server with all tools, we chose domain-specific servers for:
- **Independent deployment** - Update one server without affecting others
- **Clear boundaries** - Single responsibility per server
- **Resource isolation** - File operations don't impact API calls
- **Security isolation** - Different permission scopes per server

See [ADR-001](./docs/adr-001-multi-server-architecture.md) for detailed decision rationale.

## Server Inventory

### 1. GitHub MCP Server
**Purpose:** Interact with GitHub repositories via REST API  
**Tools:** Repository info, PR management, issue tracking, file operations  
**Key Features:**
- Rate limiting with exponential backoff
- Response caching for read operations
- Comprehensive error handling

[Documentation](./servers/github-mcp-server/README.md)

### 2. Filesystem MCP Server
**Purpose:** File system operations with security controls  
**Tools:** Read/write files, directory operations, CSV/Excel, compression  
**Key Features:**
- Path traversal protection
- File size limits
- Audit logging for destructive operations

[Documentation](./servers/filesystem-mcp-server/README.md)

### 3. Git MCP Server
**Purpose:** Local git repository operations  
**Tools:** Status, add, commit, push, pull, branch management  
**Key Features:**
- Subprocess-based git commands
- Working directory validation

[Documentation](./servers/git-mcp-server/README.md)

### 4. Email MCP Server
**Purpose:** Email sending and task management  
**Tools:** SMTP email, Todoist task creation  
**Key Features:**
- Template support
- Attachment handling
- Task integration

[Documentation](./servers/email-mcp-server/README.md)

### 5. Web MCP Server
**Purpose:** Web search and content retrieval  
**Tools:** Brave API search, webpage fetching/parsing  
**Key Features:**
- BeautifulSoup content extraction
- Search result ranking

[Documentation](./servers/web-mcp-server/README.md)

### 6. Data Format MCP Server
**Purpose:** Parse and generate structured data  
**Tools:** JSON/XML parsing, validation, generation  
**Key Features:**
- Schema validation
- Format conversion

[Documentation](./servers/data-format-mcp-server/README.md)

### 7. OS Info MCP Server
**Purpose:** System monitoring and metrics  
**Tools:** CPU usage, memory usage, system info  
**Key Features:**
- Real-time metrics
- Cross-platform support

[Documentation](./servers/os-info-mcp-server/README.md)

### 8. Prize Draw MCP Server
**Purpose:** Prize draw discovery and entry mechanics (search, parse, submit, log)
**Tools:** `search_draws`, `parse_entry_page`, `submit_entry`, `check_log`
**Key Features:**
- Pluggable source config (static/mock and RSS)
- Dry-run mode and safety refusals for personal data / purchase-required draws
- JSONL entry log for duplicate avoidance
- No LLM/provider logic - reasoning is left to the orchestrating client

[Documentation](./servers/prize-draw-mcp-server/README.md)

## Requirements

### MCP Version Pin

Every server's `requirements.txt` under `projects/01-mcp-server-suite/servers/*/requirements.txt` **must** include a line pinning the MCP SDK to the 1.x line:

```
mcp<2.0.0
```

The upper bound is deliberate: the 2.x line of the `mcp` package introduces breaking API changes that these servers have not been migrated to, so an open range (e.g. `mcp` or `mcp>=1.0`) would pull in incompatible versions at install time. Pin the upper bound explicitly rather than relying on transitive constraints.

CI enforces this. The Python workflow at [`.github/workflows/python-ci.yml`](../../.github/workflows/python-ci.yml) iterates over each `projects/01-mcp-server-suite/servers/*/requirements.txt` and fails the build if the pin is missing, with:

```
::error::FAIL: <file> does not pin mcp<2.0.0
```

If you add a new server under `projects/01-mcp-server-suite/servers/`, add the `mcp<2.0.0` line to its `requirements.txt` before opening a PR. Do not loosen or remove the pin to make CI pass — the check is intentional.

## Quick Start

### Try It Out (5 Minutes)

Get the GitHub MCP server running:

```bash
# 1. Clone and navigate
git clone https://github.com/leonarduk/ai-systems-lab.git
cd ai-systems-lab/projects/01-mcp-server-suite/servers/github-mcp-server

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your GitHub token
export GITHUB_TOKEN="your_github_token_here"  # Windows: set GITHUB_TOKEN=your_token

# 5. Run the server
python server.py
```

**Get a GitHub token:** https://github.com/settings/tokens (need `repo` scope)

### Full Configuration

For all servers, create `.env` file in project root:

```bash
# GitHub API
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Brave Search API (2000 free queries/month)
# Get key at: https://brave.com/search/api/
BRAVE_API_KEY=BSAxxxxxxxxxxxxxxxxxxxx

# Email/SMTP (Gmail example)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password  # Get from Google Account settings

# Todoist
TODOIST_EMAIL=your-unique-email@todoist.com
```

### Running Other Servers

```bash
# Filesystem server (no API key needed)
cd servers/filesystem-mcp-server
python server.py

# Web search server (needs Brave API key)
cd servers/web_mcp_server
python server.py

# See individual server READMEs for specific requirements
```

### Testing It Works

Each server includes a simple test client:

```bash
# Test the GitHub server
cd servers/github-mcp-server
python -c "import server; print('Server loaded successfully')"

# Or run the test suite
pytest tests/ -v
```

## Development

### Testing

```bash
# Run tests for a specific server
cd servers/github-mcp-server
pytest tests/ -v --cov=server

# Run all tests
pytest servers/*/tests/ -v
```

### Code Quality

```bash
# Linting
flake8 servers/
black servers/ --check

# Security scanning
bandit -r servers/
```

## Project Structure

```
01-mcp-server-suite/
├── README.md                    # This file
├── docs/                        # Architecture & design docs
│   ├── adr-001-multi-server-architecture.md
│   ├── api-reference.md
│   └── troubleshooting.md
├── servers/                     # Server implementations
│   ├── github-mcp-server/
│   ├── filesystem-mcp-server/
│   ├── git-mcp-server/
│   ├── email-mcp-server/
│   ├── web-mcp-server/
│   ├── data-format-mcp-server/
│   ├── os-info-mcp-server/
│   └── prize-draw-mcp-server/
└── shared/                      # Common utilities (future)
    ├── auth.py
    ├── logging.py
    └── retry.py
```

## Technology Stack

- **Language:** Python 3.11+
- **Framework:** MCP Protocol (Model Context Protocol)
- **APIs:** GitHub REST API, Brave Search API, SMTP
- **Libraries:** 
  - `mcp` - MCP protocol implementation (pinned to `<2.0.0`, see [Requirements](#mcp-version-pin))
  - `requests` - HTTP client
  - `pandas`, `openpyxl` - Data processing
  - `beautifulsoup4` - HTML parsing
  - `psutil` - System monitoring

## Status

**Current Phase:** Production-ready, actively maintained

**Known Issues:**
- Rate limiting needs tuning for high-volume usage
- Cache invalidation strategy needs refinement
- Some servers lack comprehensive integration tests

**Roadmap:**
- [ ] Add Redis-based caching layer
- [ ] Implement circuit breaker pattern
- [ ] Add OpenTelemetry instrumentation
- [ ] Create shared utility library
- [ ] Add Terraform deployment configs

## Contributing

This is a personal portfolio project, but suggestions and feedback are welcome via issues.

## License

MIT License - see [LICENSE](../../LICENSE) for details

## Contact

**LinkedIn:** [linkedin.com/in/leonarduk](https://www.linkedin.com/in/leonarduk)

---

*Part of the [AI Systems Lab](../../README.md) portfolio*
