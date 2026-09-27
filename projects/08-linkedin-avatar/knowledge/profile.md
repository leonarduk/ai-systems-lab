<!--
Length limit: 4000 words / 25000 bytes (see [tool.profile-length] in pyproject.toml).
Check locally with: python scripts/check_profile_length.py
-->
## Contact
[redacted]
[redacted]
(LinkedIn)
github.com/leonarduk (Portfolio)
medium.com/@steveleonard11
(Blog)
## Skills
Languages: Python · Java · SQL · C++ · Bash/Unix
Cloud & Infra: AWS (Lambda, Glue, ECS, API Gateway, DynamoDB, CDK) · Docker · Nomad · Cloud Foundry · Terraform · NoSQL
AI & LLM: MCP tool server design · LLM integration (OpenAI, Anthropic APIs) · RAG retrieval pipelines · SQL generation · prompt engineering · agentic workflow design
Frameworks & Data: FastAPI · Spring Boot · Kafka · Apache Spark · microservices · event-driven architecture · REST APIs · pandas · numpy · polars · SQLAlchemy · pydantic · asyncio
Tooling & Practice: Claude Code · TDD · CI/CD (Jenkins) · Git/GitHub · Jira · SonarQube · Snyk/SCA scanning · AGENTS.md patterns · Agile/Scrum · Sybase IQ
## Spoken Languages
French (Professional Working) — DELF B2
Spanish (Limited Working)
English (Native or Bilingual)
German (Professional Working) — Goethe-Zertifikat B2
## Certifications
AWS Certified Generative AI Developer – Professional
AWS Certified Solutions Architect – Professional
AWS Certified Security – Specialty
AWS Certified Data Engineer – Associate
AWS Certified Developer – Associate
FRM, Financial Risk Manager
## Training & CPD
AI Engineer Agentic Track: The Complete Agent & MCP Course — Udemy, Ed Donner (17h, May 2026).
Covered LangChain, LangGraph, AutoGen, MCP server design, and agentic workflow patterns.
Stephen Leonard
Senior Software Engineer | AI/LLM & Agentic Engineering | Python · Java | AWS | Financial Services
London, England, United Kingdom
## Summary
Senior software engineer with 20+ years in financial services, with hands-on
experience designing and deploying GenAI and agentic solutions in production
— including MCP tool server architecture and RAG-based context retrieval for
natural-language database access — alongside 5+ years of enterprise Python:
data pipelines, reusable libraries, and cloud-native AWS services. Most
recently at JPMorgan, I designed and built a natural-language query system
over financial databases end to end: an LLM-powered internal chatbot with
enterprise authentication, backed by a three-tool MCP server I built for
discovery, context retrieval and auditable SQL generation against a Sybase IQ
warehouse — the kind of work where solid backend engineering and modern AI
capabilities reinforce each other, rather than AI being bolted on. My
background spans Swiss Re, Credit Suisse, Morgan Stanley, JPMorgan Asset
Management, and JPMorgan Chase across risk platforms, distributed services,
and cloud infrastructure — Python and Java, chosen for the job rather than a
single-language allegiance. I hold AWS Certified Generative AI Developer –
Professional, Solutions Architect – Professional, Security – Specialty,
Developer Associate, and Data Engineer Associate certifications alongside an
FRM — a combination that gives me both engineering depth and financial-domain
intuition. I'm looking for a senior or staff engineering role where Python
depth, applied AI/LLM engineering, and financial-services domain expertise
combine to deliver measurable impact — Java-led roles are fine too, and I'd
lean toward a polyglot environment over a single-language shop where
possible. My background means I think carefully about auditability,
observability and guardrails — which matters more in production AI systems
than most job specs acknowledge. Dual UK/Irish citizen. Based in London.
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
prompt libraries, AGENTS.md coding standards with extended CI quality gates
(Checkstyle, SonarQube) and Copilot guidance — while personally building MCP
tools and context-handling approaches shared across wider teams.
• Team lead of 2 direct reports; established and ran the department's AWS
Guild and AI Guild as SME in both areas, sharing generative AI and LLM best
practices and upskilling engineers across the wider department.
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
• Ran several concurrent Kafka consumers enriching PnL and Risk messages
before persistence, using lock-based synchronization to protect shared state
and prevent race conditions under high-throughput load.
• Used Snyk and custom SCA scripts to identify and remediate dependency
vulnerabilities across services.
• Modernised core services from Java 8 to Java 21 using in-house remediation
tooling, incrementally updating repos as part of regular development work to
reduce accumulated technical debt.
• Led a year-long AWS proof of concept to evaluate replacing a legacy
batch processing system, using AWS Glue and DynamoDB; delivered a
production-ready architecture with full architectural approval; wrote Python
tooling to reverse-engineer and visualise stored procedures and reconcile
UAT vs Production data, resolving the majority of long-standing technical
debt blockers.
Lead Python Developer VP
November 2021 - August 2022 (10 months)
London
• Inherited a buggy, undocumented Python data access library — a former
side project — with an aggressive AWS deployment deadline, no design spec,
and two junior developers focused on BAU. Redesigned the architecture so
a single pip-installable wheel ran across AWS, private cloud and Windows
client environments, adapting behaviour by deployment context while sharing

a common codebase. The resulting service was nominated for the 2022
American Financial Technology Awards "Best Analytics" Initiative.
• Introduced smoke tests and integration tests that surfaced significant
technical debt; refactored the codebase to reduce duplication, eliminate
bugs, and reduce production outages; built thin client and Excel add-in
enabling non-technical analysts to query financial data directly.
• Used numpy, polars, SQLAlchemy, pydantic, and asyncio extensively across
data access, reconciliation, and pipeline work throughout this and
subsequent roles.
Credit Suisse
Application Architect & Developer (VP)
December 2019 - October 2021 (1 year 11 months)
London Area, United Kingdom
• Diagnosed and resolved a critical processing bottleneck in a Java-based
risk system, reducing portfolio processing time from over an hour to several
minutes and unblocking a delayed migration project
• Seconded informally to support a Python data engineering team after they
lost their specialist; optimised a risk data extraction and reporting pipeline
using async processing, halving server usage and cutting runtime from 6 hours
to 30 minutes
• Built Docker-based microservices deployed to Nomad on AWS, tested locally
and in CI/CD.
Morgan Stanley
Developer
September 2018 - November 2019 (1 year 3 months)
London, United Kingdom
• Introduced remote debugging tooling for Linux-based C++ systems,
significantly accelerating development velocity; added gcov code coverage
and built unit tests increasing C++ coverage from 20% to 32%
• Created extensive developer documentation including system architecture
diagrams, adopted as reference material across the team
Standard Chartered Bank
Consulting Java Developer
April 2016 - September 2018 (2 years 6 months)
London, United Kingdom
• Developed Java 8 RESTful web services for enterprise risk and margin
calculation systems, with automated Python regression testing
• Nominated Most Active Contributor in the bank-wide Business Efficiency
ideas forum
Credit Suisse
Consulting Java Developer and Scrum Master
April 2014 - February 2016 (1 year 11 months)
London Area, United Kingdom
• Developed trade migration from legacy to strategic global trade store and
Fidessa in a geographically dispersed Agile team, acting as Scrum Master

•Reduced development time for field mapping changes from days  to minutes
by simplifying code logic; replaced manual deployment confirmation process
with a self-service web tool
Swiss Re
Consulting Analyst Developer within Risk Technology
September 2007 - December 2013 (6 years 4 months)
London Area, United Kingdom
• Senior software engineer on credit and market risk reporting systems for a
global team across 4 time zones,  eliciting requirements from non-technical risk
managers in Zurich and New York
• Replaced legacy Java VaR feeder application with a simpler Perl-based
solution, reducing development time from weeks to days
• Automated Engineer-to-Engineer UAT for a key Murex migration, cutting the
test cycle from 2 days to 5 hours
HBOS Treasury Services
Consulting Credit Risk Developer
June 2006 - June 2007 (1 year 1 month)
Nomura International
System Developer
July 2004 - June 2006 (2 years)
Lehman Brothers
Risk Developer
June 2001 - June 2004 (3 years 1 month)
Intuwave
Software Engineer
December 2000 - June 2001 (7 months)
Standard Chartered Bank
Analyst Programmer
November 1999 - December 2000 (1 year 2 months)
## Education
City University London
MSc (Part-time), Business Systems Analysis & Design · (2005 - 2008)

The University of Edinburgh
BSc, Physics · (1993 - 1997)
