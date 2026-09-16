## Contact
[redacted]
[redacted]
(LinkedIn)
github.com/leonarduk (Portfolio)
medium.com/@steveleonard11
(Blog)
Top Skills
E-Learning
Technical Writing
Large Language Models (LLM)
## Languages
French (Professional Working)
Spanish (Limited Working)
English (Native or Bilingual)
German (Professional Working)
## Certifications
FRM, Financial Risk Manager
AWS Certified: Data Engineer –
Associate
Goethe-Zertifikat B2 (upper
intermediate German)
AWS Certified Developer –
Associate
Stephen Leonard
Senior Java Engineer | Financial Services | Python, AWS Gen AI Pro
& FRM | AI & LLM Integration
London, England, United Kingdom
## Summary
Senior software engineer with 20+ years in financial services,
primarily building Java-based systems — distributed services, risk
platforms, and high-throughput data processing — with growing
depth in applied AI and LLM integration. Most recently at JPMorgan,
I designed and built a natural-language query system over
financial databases end to end: an LLM-powered internal chatbot with
enterprise authentication, backed by a three-tool MCP server I built
for discovery, context retrieval and auditable SQL generation against
a Sybase IQ warehouse — the kind of work where solid backend
engineering and modern AI capabilities reinforce each other, rather
than AI being bolted on. My
background spans Swiss Re, Credit Suisse, Morgan Stanley, and
JPMorgan across risk platforms, distributed services, and cloud
infrastructure — mostly Java, with Python where it's the right tool for
data pipelines and tooling. I hold an FRM alongside AWS Solutions
Architect Professional, Security Specialty, Developer Associate, and
Data Engineer Associate certifications — a combination that gives
me both engineering depth and financial-domain intuition. I'm looking
for a senior Java engineering role in financial services or a related
regulated environment — Python-led roles are fine too, and I'd lean
toward a polyglot environment over a single-language shop where
possible. My background means I think carefully about auditability,
observability and guardrails — which matters more in production AI
systems than most job specs acknowledge. Dual UK/Irish citizen.
Based in London.
## Experience
JPMorgan Chase & Co.
4 years 7 months
Lead Software Engineer VP
September 2022 - May 2026 (3 years 9 months)

Greater London
• Querying risk data required users to learn a complex Sybase IQ schema.
Built an LLM‑driven chatbot — a Python UI running in production on a
private Cloud Foundry instance — on the firm’s LLMSuite platform, which
abstracted Azure‑hosted LLMs behind an internal API, integrating enterprise
SSO and a three‑tool MCP server for discovery, context retrieval and
auditable SQL generation. Prototyped the MCP server in Python, then
rebuilt it in Java/Spring Boot for production; handed over for final
permission tuning and QA.
• Under pressure from senior management to improve AI tool adoption, led
the LLM adoption initiative across ~20 engineers — producing documentation,
prompt libraries and Copilot guidance — while personally building MCP tools
and context-handling approaches shared across wider teams.
• Inherited a hard three-month deadline to replace an obsolete chassis library
across multiple repos — including the core query service — requiring a full
codebase ramp-up and authentication layer refactor. Delivered on time.
• Introduced integration and smoke test coverage using in-house Karate
tooling across deployment pipelines, uncovering long-standing bugs on under-
tested endpoints that had gone undetected in production.
• Refactored a core Spring Boot REST service to implement Parquet
streaming, saving 10–15 minutes on a regulatory data feed as part of a
broader initiative to meet SLA deadlines.
• Identified and removed a poorly implemented Kafka monitoring service
that was bottlenecking message throughput, merging its functionality into an
existing application — scaling throughput from 20M → 800M messages/day
and eliminating a redundant microservice saving $10,000/year.
• Modernised core services from Java 8 to Java 21 using in-house remediation
tooling, incrementally updating repos as part of regular development work to
reduce accumulated technical debt.
• Led a year-long AWS proof of concept to evaluate replacing a legacy
batch processing system; delivered
