"""
fmtgen basic usage examples.

These examples show how to use fmtgen with different providers and constraint types.
Set the appropriate API keys before running:
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from fmtgen import Fmt

# --- Models ---


class Sentiment(BaseModel):
    label: str
    confidence: float
    reasoning: str


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Ticket(BaseModel):
    title: str = Field(description="Short summary of the issue")
    priority: Priority
    assignee: str | None = None
    estimated_hours: float = Field(ge=0, le=1000)
    tags: list[str] = Field(default_factory=list)


class User(BaseModel):
    name: str
    age: int
    email: str


# --- Examples ---


def example_json_schema() -> None:
    """Extract structured data using a Pydantic schema."""
    fmt = Fmt()

    # Works with any provider
    result = fmt.generate(
        model="ollama/llama3",
        prompt="Extract: John Doe, 32 years old, john@example.com",
        schema=User,
    )
    print(f"User: {result}")


def example_choice() -> None:
    """Classify text into one of several categories."""
    fmt = Fmt()

    result = fmt.generate(
        model="ollama/llama3",
        prompt="Classify: 'This product completely exceeded my expectations!'",
        choice=["positive", "negative", "neutral", "mixed"],
    )
    print(f"Sentiment: {result}")


def example_regex() -> None:
    """Generate text matching a regex pattern (requires vLLM/SGLang)."""
    fmt = Fmt()

    result = fmt.generate(
        model="vllm/meta-llama/Llama-3.1-8B-Instruct",
        prompt="Generate a date for the Apollo 11 moon landing",
        regex=r"\d{4}-\d{2}-\d{2}",
    )
    print(f"Date: {result}")


def example_grammar() -> None:
    """Generate text conforming to a grammar (requires vLLM/SGLang)."""
    fmt = Fmt()

    result = fmt.generate(
        model="vllm/codellama/CodeLlama-13b-Instruct-hf",
        prompt="Write a query to find active users from 2024",
        grammar="""
        start: select_stmt
        select_stmt: "SELECT" columns "FROM" table where_clause?
        columns: "*" | column ("," column)*
        column: /[a-zA-Z_][a-zA-Z0-9_]*/
        table: /[a-zA-Z_][a-zA-Z0-9_]*/
        where_clause: "WHERE" condition ("AND" condition)*
        condition: column "=" value
        value: NUMBER | STRING
        NUMBER: /[0-9]+/
        STRING: "'" /[^']*/ "'"
        """,
    )
    print(f"SQL: {result}")


def example_raw_result() -> None:
    """Get full metadata from a generation."""
    fmt = Fmt()

    result = fmt.generate_raw(
        model="ollama/llama3",
        prompt="Extract: John Doe, 32, john@example.com",
        schema=User,
    )
    print(f"Raw: {result.raw}")
    print(f"Parsed: {result.parsed}")
    print(f"Provider: {result.provider_used}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Tokens: {result.tokens_generated}")


def example_status() -> None:
    """Check available backends and providers."""
    fmt = Fmt()
    fmt.status()


async def example_async() -> None:
    """Use async methods for non-blocking generation."""
    import asyncio

    fmt = Fmt()

    # agenerate returns parsed output, just like generate
    result = await fmt.agenerate(
        model="ollama/llama3",
        prompt="Extract: Jane Doe, 28 years old, jane@example.com",
        schema=User,
    )
    print(f"Async User: {result}")

    # agenerate_raw returns full GenerateResult metadata
    raw_result = await fmt.agenerate_raw(
        model="ollama/llama3",
        prompt="Classify: 'The service was terrible'",
        choice=["positive", "negative", "neutral"],
    )
    print(f"Async Choice: {raw_result.parsed}")
    print(f"Async Latency: {raw_result.latency_ms:.1f}ms")

    # Run multiple generations concurrently
    tasks = [
        fmt.agenerate(
            model="ollama/llama3",
            prompt=f"Extract: User {i}, age {20 + i}, user{i}@example.com",
            schema=User,
        )
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    print("=== fmtgen Status ===")
    example_status()
    print()

    # Uncomment examples based on your setup:
    # print("=== JSON Schema ===")
    # example_json_schema()
    # print()
    # print("=== Choice ===")
    # example_choice()
    # print()
    # print("=== Regex (vLLM) ===")
    # example_regex()
    # print()
    # print("=== Grammar (vLLM) ===")
    # example_grammar()
    # print()
    # print("=== Async ===")
    # import asyncio
    # asyncio.run(example_async())
