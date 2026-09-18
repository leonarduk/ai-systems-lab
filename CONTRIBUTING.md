# Contributing to AI Systems Lab

Thank you for your interest in this project! 

## About This Repository

This is a **personal portfolio repository** documenting my learning journey into AI systems engineering. While I'm not actively seeking code contributions, I welcome:

- 🐛 Bug reports
- 💡 Suggestions for improvements
- 🤔 Questions about implementation decisions
- 📚 Resources or references that might be helpful

## Reporting Issues

If you spot a bug or have a suggestion:

1. Check if an issue already exists
2. If not, [open a new issue](../../issues/new)
3. Provide context:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce (if applicable)

## Suggesting Improvements

I'm particularly interested in feedback on:

- **Architecture decisions**: Are there better patterns I should consider?
- **Security issues**: Especially in the filesystem and API integrations
- **Code quality**: Best practices I might have missed
- **Documentation**: Is anything unclear or missing?

## Questions & Discussion

Feel free to:
- Open an issue for technical questions
- Share resources or articles relevant to the projects
- Point out learning resources you found helpful

## What I'm NOT Looking For

Since this is a learning/portfolio project:
- ❌ Pull requests with major rewrites
- ❌ Requests to add new features
- ❌ Style/formatting nitpicks (unless they impact functionality)

## MCP Server Suite: Version Pin Requirement

If you're touching anything under `projects/01-mcp-server-suite/servers/`, note that every server's `requirements.txt` must pin `mcp<2.0.0` — CI enforces this and will fail with `::error::FAIL: <file> does not pin mcp<2.0.0` otherwise. See the [MCP Version Pin section in the suite README](./projects/01-mcp-server-suite/README.md#mcp-version-pin) for the rationale and the exact file glob.

## Automated PR Review

Every pull request is automatically reviewed by Claude, DeepSeek, and GPT (see
`.github/workflows/*-pr-review.yml`). Each posts an advisory review comment with
an APPROVE / REQUEST CHANGES verdict; a provider can be disabled repo-wide via
its `ENABLE_<PROVIDER>_REVIEW` repository variable. Add the `Deep Review
Required` label to a PR to opt DeepSeek into a stronger model with a larger
token budget. Approved PRs may get non-blocking follow-ups auto-filed as
issues labeled `ai-suggested`.

## Repository Automation

This repository uses the shared
[`cicaid-devtools`](https://github.com/leonarduk/cicaid) package for local
issue, pull-request, and review automation. Install the development dependencies
from the repository root:

```bash
python -m pip install -r requirements-dev.txt
```

Run `cicaid --help` to see all available commands. Common tasks include:

```bash
cicaid sync-issues
cicaid work-on-issue 81
cicaid local-review
cicaid commit-and-push
cicaid publish-pr
```

The package is pinned in `requirements-dev.txt`, and Python CI installs it and
runs a CLI smoke check. Repository automation should be added to the shared
package rather than copied into a local `scripts/developer_tools/` directory.

## Code of Conduct

Please be respectful and constructive. This is a learning repository, not a production system. Comments like "you should just use library X" without context aren't helpful - explain the trade-offs!

## Questions?

Feel free to reach out via:
- **GitHub Issues**: For technical questions about the code
- **LinkedIn**: [linkedin.com/in/leonarduk](https://www.linkedin.com/in/leonarduk) for general discussion

---

## For Those Building Similar Projects

If you're also learning AI systems engineering and found something useful here:

1. ⭐ Star the repository if you found it helpful
2. Feel free to use code/patterns with attribution
3. Share what you learned differently - I'm learning too!

## License

All code in this repository is available under the MIT License. See [LICENSE](./LICENSE) for details.

---

*This is a personal learning project. The goal is demonstrating growth and capability, not building a production framework.*
