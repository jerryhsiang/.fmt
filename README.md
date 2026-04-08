<h1 align="center">.fmt</h1>

<p align="center">
  <strong>One API for structured LLM outputs. Every backend. Every provider.</strong>
</p>

<p align="center">
  <a href="https://github.com/jerryhsiang/.fmt/actions"><img src="https://img.shields.io/github/actions/workflow/status/jerryhsiang/.fmt/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+">
</p>

<p align="center">
  <a href="#the-problem">The Problem</a> · <a href="#quickstart">Quickstart</a> · <a href="#supported-backends--providers">Backends & Providers</a> · <a href="#api-reference">API</a> · <a href="#benchmarking">Benchmarks</a>
</p>

---

## The Problem

If you've tried to get structured outputs from LLMs, you've hit this wall:

**Every provider does it differently.** OpenAI uses `response_format`. Anthropic uses `tool_use`. Ollama uses a `format` parameter. vLLM uses `guided_json` in `extra_body`. Each one requires different code, different schema formats, different parsing logic.

**Every constrained decoding engine has its own API.** Outlines, XGrammar, llguidance, lm-format-enforcer — four engines, four completely different interfaces. Want to switch? Rewrite your integration.

**The result:** you end up writing and maintaining fragile glue code for every provider/backend combination you need. When a new engine comes out that's 10x faster, switching means rewriting everything.

### What fmtgen does

fmtgen is a **thin routing layer** — not a new engine. It wraps every major structured generation backend and every major LLM provider behind a single `.generate()` call:

```python
from fmtgen import Fmt
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    line_items: list[str]

fmt = Fmt()

# Same code. Same types. Any provider.
invoice = fmt.generate(
    model="ollama/llama3",  # or "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514", "vllm/..."
    prompt="Extract: Acme Corp invoice, $1,250.00 USD for consulting and travel",
    schema=Invoice,
)

print(invoice)
# Invoice(vendor='Acme Corp', amount=1250.0, currency='USD', line_items=['consulting', 'travel'])
```

**No parsing. No retries. No provider-specific code. Just types.**

You pass in a Pydantic model, regex pattern, grammar, or list of choices — fmtgen handles routing, schema compilation, backend selection, output parsing, and validation. Switch providers or backends by changing one string.

---

## Why this exists

| **Without fmtgen** | **With fmtgen** |
|---|---|
| Different code for OpenAI, Anthropic, Ollama, vLLM | **One `.generate()` call** — same code everywhere |
| Different code for Outlines vs XGrammar vs llguidance | **Auto-selects** the fastest available backend |
| Schema compilation quirks per backend (`$ref`, `$defs`) | **Handles internally** — pass a Pydantic model, get a validated instance |
| Cryptic errors deep inside constrained decoding engines | **Human-readable errors** with concrete fix suggestions |
| No way to compare backend performance | **Built-in benchmarking** across all backends |
| Switching backends means rewriting integration code | **Change one string** — nothing else changes |

### The 30-second version

```python
from fmtgen import Fmt
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str  # "positive", "negative", "neutral"
    confidence: float
    reasoning: str

fmt = Fmt()

# Works with any provider — same code, same types
result = fmt.generate(model="openai/gpt-4o-mini", prompt="Review: Amazing product!", schema=Sentiment)
result = fmt.generate(model="ollama/llama3",       prompt="Review: Amazing product!", schema=Sentiment)
result = fmt.generate(model="vllm/meta-llama/Llama-3.1-8B-Instruct", prompt="Review: Amazing product!", schema=Sentiment)
result = fmt.generate(model="anthropic/claude-sonnet-4-20250514", prompt="Review: Amazing product!", schema=Sentiment)

# They all return: Sentiment(label='positive', confidence=0.95, reasoning='...')
```

---

## Installation

```bash
# Core SDK (works with API providers out of the box)
pip install fmtgen

# Add constrained decoding backends for local models
pip install fmtgen[outlines]       # Most mature, broadest support
pip install fmtgen[xgrammar]      # Fastest (up to 100x on grammar workloads)
pip install fmtgen[llguidance]    # Low latency, ~50μs/token, minimal startup cost
pip install fmtgen[lmfe]          # Best for debugging prompt-vs-schema conflicts

# Add API provider SDKs
pip install fmtgen[openai]
pip install fmtgen[anthropic]

# Everything
pip install fmtgen[all]
```

---

## Quickstart

### JSON Schema via Pydantic

The most common use case. Define a Pydantic model, get a validated instance back.

```python
from fmtgen import Fmt
from pydantic import BaseModel, Field
from enum import Enum

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class ExtractedTicket(BaseModel):
    title: str = Field(description="Short summary of the issue")
    priority: Priority
    assignee: str | None = None
    estimated_hours: float = Field(ge=0, le=1000)
    tags: list[str] = Field(default_factory=list)

fmt = Fmt()

ticket = fmt.generate(
    model="openai/gpt-4o-mini",
    prompt="""
    Extract a support ticket from this message:

    "Hey team, the checkout flow is completely broken on mobile.
    Customers are getting a white screen after entering payment info.
    This is costing us revenue — need Sarah to look at it ASAP.
    Probably a 4-hour fix based on the last time this happened."
    """,
    schema=ExtractedTicket,
)

print(ticket.title)           # "Mobile checkout white screen after payment entry"
print(ticket.priority)        # Priority.critical
print(ticket.assignee)        # "Sarah"
print(ticket.estimated_hours) # 4.0
print(ticket.tags)            # ["checkout", "mobile", "payment", "bug"]
```

### Regex Constraints

Guarantee output matches a pattern. Useful for emails, phone numbers, IDs, dates.

```python
# Generate a valid email address
email = fmt.generate(
    model="vllm/meta-llama/Llama-3.1-8B-Instruct",
    prompt="Generate a professional email for Jane Smith at Acme Corp",
    regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)
# "jane.smith@acmecorp.com"

# Generate a date in ISO format
date = fmt.generate(
    model="vllm/meta-llama/Llama-3.1-8B-Instruct",
    prompt="When was the Apollo 11 moon landing?",
    regex=r"\d{4}-\d{2}-\d{2}",
)
# "1969-07-20"
```

### Grammar Constraints (CFG / EBNF)

Constrain output to a domain-specific language. Perfect for SQL, config files, DSLs.

```python
sql = fmt.generate(
    model="vllm/codellama/CodeLlama-13b-Instruct-hf",
    prompt="Write a query to find all active users who signed up in 2024",
    grammar="""
    start: select_stmt
    select_stmt: "SELECT" columns "FROM" table where_clause? order_clause? limit_clause?
    columns: "*" | column ("," column)*
    column: /[a-zA-Z_][a-zA-Z0-9_]*/
    table: /[a-zA-Z_][a-zA-Z0-9_]*/
    where_clause: "WHERE" condition ("AND" condition)*
    condition: column operator value
    operator: "=" | "!=" | ">" | "<" | ">=" | "<=" | "LIKE" | "IN"
    value: NUMBER | STRING | "(" value ("," value)* ")"
    order_clause: "ORDER BY" column ("ASC" | "DESC")?
    limit_clause: "LIMIT" NUMBER
    NUMBER: /[0-9]+/
    STRING: "'" /[^']*/ "'"
    """,
)
# "SELECT * FROM users WHERE status = 'active' AND signup_year = '2024' ORDER BY created_at DESC"
```

### Choice / Classification

Force the model to pick exactly one option. Zero parsing needed.

```python
sentiment = fmt.generate(
    model="openai/gpt-4o-mini",
    prompt="Classify: 'This product completely exceeded my expectations!'",
    choice=["positive", "negative", "neutral", "mixed"],
)
# "positive"

language = fmt.generate(
    model="ollama/llama3",
    prompt="What language is this: 'Bonjour, comment allez-vous?'",
    choice=["english", "french", "spanish", "german", "italian", "portuguese"],
)
# "french"
```

### Full Result Metadata

Use `generate_raw()` to get timing, backend info, and the raw response.

```python
result = fmt.generate_raw(
    model="ollama/llama3",
    prompt="Extract: John Doe, 32, john@example.com",
    schema=User,
)

print(result.raw)             # '{"name": "John Doe", "age": 32, "email": "john@example.com"}'
print(result.parsed)          # User(name='John Doe', age=32, email='john@example.com')
print(result.backend_used)    # "xgrammar"
print(result.provider_used)   # "ollama"
print(result.latency_ms)      # 142.7
print(result.constraint_type) # "json_schema"

# As a dict (useful for logging/telemetry)
print(result.to_dict())
```

### Async Support

Every method has an async counterpart. Use `agenerate()` and `agenerate_raw()` for async workflows.

```python
import asyncio
from fmtgen import Fmt
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

async def main():
    fmt = Fmt()

    # Single async call
    user = await fmt.agenerate(
        model="openai/gpt-4o-mini",
        prompt="Extract: Jane Smith, 28",
        schema=User,
    )

    # Concurrent calls
    users = await asyncio.gather(
        fmt.agenerate(model="openai/gpt-4o-mini", prompt="Extract: Alice, 30", schema=User),
        fmt.agenerate(model="openai/gpt-4o-mini", prompt="Extract: Bob, 25", schema=User),
    )

asyncio.run(main())
```

---

## Supported Backends & Providers

### Constrained Decoding Backends

These enforce structure at the **token level** during generation. The model physically cannot produce invalid output.

| Backend | JSON Schema | Regex | Grammar (CFG) | Relative Speed | Best For |
|---|---|---|---|---|---|
| **[XGrammar](https://github.com/mlc-ai/xgrammar)** | Yes | Yes | Yes | Fastest | Production throughput |
| **[llguidance](https://github.com/guidance-ai/llguidance)** | Yes | Yes | Yes | Fastest | Low-latency, minimal startup |
| **[Outlines](https://github.com/dottxt-ai/outlines)** | Yes | Yes | Yes | Fast | Broadest compatibility |
| **[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)** | Yes | Yes | No | Moderate | Debugging & diagnostics |

### API Providers

These use **provider-native** structured output mechanisms (OpenAI's `response_format`, Anthropic's `tool_use`, etc.).

| Provider | JSON Schema | Regex | Grammar | Choice | Auth |
|---|---|---|---|---|---|
| **OpenAI** | Yes (native) | No | No | Yes | `OPENAI_API_KEY` |
| **Anthropic** | Yes (tool_use) | No | No | Yes | `ANTHROPIC_API_KEY` |
| **Ollama** | Yes (native) | No | No | Yes | Local (no key) |
| **vLLM** | Yes (guided) | Yes | Yes | Yes | Local (no key) |
| **SGLang** | Yes (guided) | Yes | Yes | Yes | Local (no key) |

### Auto-Detection

When you set `backend="auto"` (the default), fmtgen selects the best available backend:

```
Priority: XGrammar > llguidance > Outlines > lm-format-enforcer
```

For API providers, fmtgen automatically uses the provider's native structured output mechanism — no backend needed.

---

## Architecture

```
                        from fmtgen import Fmt

    fmt.generate(model="...", prompt="...", schema=MyModel)
                            |
                      Model String Router
                            |
            +---------------+----------------+
            |                                |
      API Providers                    Local Backends
      (native structured outputs)      (constrained decoding)
            |                                |
    +-------+-------+            +----------+-----------+
    |   |   |   |   |            |    |     |     |     |
   OAI Ant Oll vLLM SGLang    XGram llguid Outl  LMFE
```

**Key design decisions:**

1. **Model string is the router.** `"openai/gpt-4o"` uses OpenAI. `"ollama/llama3"` uses Ollama. No config objects.
2. **Pydantic-native.** Pass a model class, get a validated instance. Schema compilation and `$ref` resolution handled internally.
3. **Backends are optional.** API providers work out of the box. Backends only needed for local inference or features the provider doesn't support natively.
4. **Errors are actionable.** Every exception includes a `suggestion` field with a concrete fix.

---

## Benchmarking

Compare backends head-to-head on the same prompt and schema.

```python
from fmtgen import benchmark
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

results = benchmark(
    prompt="Extract user info: John Doe, 32, john@example.com",
    schema=User,
    model="meta-llama/Llama-3.1-8B-Instruct",
    backends=["outlines", "xgrammar", "llguidance"],
    iterations=100,
)

results.print_table()
```

```
Backend                    Avg (ms)   p50 (ms)   p99 (ms)    Success
---------------------------------------------------------------------
xgrammar                       12.3       11.8       18.7      100.0%
llguidance                     14.8       13.2       21.2      100.0%
outlines                       34.1       32.6       52.4      100.0%
lm-format-enforcer             89.4       84.2      142.1      100.0%
```

---

## Configuration

### Provider authentication

```python
# Option 1: Environment variables (recommended)
# export OPENAI_API_KEY=sk-...
# export ANTHROPIC_API_KEY=sk-ant-...
fmt = Fmt()

# Option 2: Explicit configuration
fmt = Fmt(provider_kwargs={
    "openai": {"api_key": "sk-...", "base_url": "https://custom-endpoint.com/v1"},
    "anthropic": {"api_key": "sk-ant-..."},
    "ollama": {"base_url": "http://gpu-server:11434"},
    "vllm": {"base_url": "http://vllm-server:8000/v1", "guided_backend": "xgrammar"},
})
```

### Backend selection

```python
# Auto-select best available backend (default)
fmt = Fmt(backend="auto")

# Force a specific backend
fmt = Fmt(backend="outlines")
fmt = Fmt(backend="xgrammar")
```

### Check what's available

```python
fmt = Fmt()
fmt.status()
```

```
fmtgen Backend Status
==================================================
  xgrammar                  [json_schema, regex, grammar, choice]
  llguidance                [json_schema, regex, grammar, choice]
  outlines                  [json_schema, regex, grammar, choice]
  lm-format-enforcer        [json_schema, regex, choice]

Available Providers
==================================================
  openai
  anthropic
  ollama
  vllm
  sglang
```

---

## Error Handling

fmtgen gives you errors you can act on — not stack traces from deep inside a backend.

```python
from fmtgen.exceptions import (
    BackendNotAvailableError,    # Backend not installed
    NoBackendAvailableError,     # No backends at all
    UnsupportedConstraintError,  # Backend can't do this constraint type
    ProviderError,               # API call failed
    SchemaValidationError,       # Output didn't match schema
    GenerationError,             # Catch-all
)

try:
    fmt.generate(model="ollama/llama3", prompt="...", schema=MyModel)
except SchemaValidationError as e:
    print(e)
    # Output failed schema validation for 'MyModel'.
    # Raw output: {"name": "John", "age": "thirty-two"...
    # Validation error: age - value is not a valid integer
    #
    # Suggestion: This usually means the model ignored the schema constraint. Try:
    #   1. Using a more capable model
    #   2. Adding schema hints to your prompt
    #   3. Switching to a local backend with true constrained decoding
```

---

## API Reference

### `Fmt`

```python
class Fmt:
    def __init__(
        self,
        backend: str = "auto",
        provider_kwargs: dict[str, dict] | None = None,
    ): ...

    def generate(
        self,
        model: str,               # "provider/model_name"
        prompt: str,
        schema: type[BaseModel] | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        choice: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> BaseModel | str: ...

    def generate_raw(...) -> GenerateResult:  # Same args, full metadata
    def status() -> None:                     # Print backend availability
    def backends -> list[str]:                # List available backends
```

### `GenerateResult`

```python
@dataclass
class GenerateResult:
    raw: str                    # Raw string output
    parsed: Any                 # Validated Pydantic instance (or string)
    backend_used: str
    provider_used: str
    model: str
    latency_ms: float
    tokens_generated: int
    constraint_type: str        # "json_schema", "regex", "grammar", "choice"

    def to_dict() -> dict:      # Serializable representation
```

---

## How It Compares

fmtgen does not compete with Outlines, XGrammar, or llguidance — it wraps them. The value is the unified interface, not the engine.

| | **fmtgen** | **Outlines** | **Instructor** | **OpenAI SDK** |
|---|---|---|---|---|
| Multi-backend | Yes (wraps all 4) | No (Outlines only) | No | No |
| Multi-provider | Yes (5) | No (local only) | Yes | No (OpenAI only) |
| JSON Schema | Yes | Yes | Yes | Yes |
| Regex | Yes (via vLLM) | Yes | No | No |
| Grammar (CFG) | Yes (via vLLM) | Yes | No | No |
| Async | Yes | Yes | Yes | Yes |
| Pydantic v2 | Yes | Yes | Yes | Partial |

The closest comparison is [Instructor](https://github.com/jxnl/instructor) by Jason Liu — a mature, battle-tested library for structured LLM outputs. Instructor focuses on API providers with retries and validation. fmtgen takes a different approach: wrapping both API providers *and* local constrained decoding backends behind one interface, so you can switch between cloud and local inference without changing code.

---

## Contributing

```bash
git clone https://github.com/jerryhsiang/.fmt.git
cd .fmt
pip install -e ".[dev]"
pytest                # run tests
ruff check .          # lint
mypy fmtgen/          # type check
```

### Areas we'd love help with

- **New provider adapters** (Google Gemini, Together AI, Fireworks, Groq)
- **Streaming** for backends that support it
- **More examples** and documentation
- **Benchmarks** on different hardware and model sizes

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgments

fmtgen stands on the shoulders of giants:

- **[Outlines](https://github.com/dottxt-ai/outlines)** by .txt — the pioneering structured generation library
- **[XGrammar](https://github.com/mlc-ai/xgrammar)** by MLC AI — blazing fast grammar engine
- **[llguidance](https://github.com/guidance-ai/llguidance)** by Guidance AI — efficient constrained decoding
- **[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)** by Noam Gat — flexible format enforcement with diagnostics
- **[vLLM](https://github.com/vllm-project/vllm)** — the inference engine that integrates all of the above
