"""Tool schemas for the DeepSeek tool-use loop.

The tool definitions are a stable, protocol-level contract, kept in their own
module so other parts of the codebase (dispatch, docs, client-side tool
registration) can import them without pulling in the notification logic.
See docs/design.md §5.
"""

RECORD_CONTACT = "record_contact"
RECORD_UNKNOWN_QUESTION = "record_unknown_question"
LOOKUP_PROJECT = "lookup_project"

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": RECORD_CONTACT,
            "description": (
                "Record that a visitor wants to be contacted. Sends a push "
                "notification with their details; call this whenever a visitor "
                "gives an email address or asks to be put in touch."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The visitor's email address.",
                    },
                    "name": {
                        "type": ["string", "null"],
                        "description": "The visitor's name, if given.",
                    },
                    "notes": {
                        "type": ["string", "null"],
                        "description": "Anything relevant about why they want to talk.",
                    },
                },
                "required": ["email", "name", "notes"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RECORD_UNKNOWN_QUESTION,
            "description": (
                "Record a question that could not be answered from the available "
                "knowledge. Call this instead of guessing whenever you don't know "
                "the answer — never invent an answer about Steve's experience."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The visitor's question, verbatim.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": LOOKUP_PROJECT,
            "description": (
                "Fetch the full record for one GitHub repo — description, "
                "languages, README excerpt and any curated note. Use this when "
                "the conversation goes into detail on a specific project named "
                "in the GitHub index."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The repo name, as it appears in the GitHub index.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
]
