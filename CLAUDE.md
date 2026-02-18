# pytest-agentcontract

Pytest plugin for deterministic CI tests of LLM agent trajectories.
Record once, replay offline, assert contracts.

## Project Structure
- `src/agentcontract/` -- main package
  - `recorder/` -- SDK interceptors (OpenAI, Anthropic) + core recorder
  - `replay/` -- replay engine with tool stubbing
  - `assertions/` -- 7 assertion types + 2 policy types
  - `adapters/` -- LangGraph, LlamaIndex, OpenAI Agents SDK
  - `plugin.py` -- pytest integration
  - `cli.py` -- CLI entry point
  - `config.py` -- YAML config loader
  - `types.py` -- core type system
  - `serialization.py` -- JSON cassette read/write
- `tests/unit/` -- unit tests (77 total)
- `examples/customer_support/` -- demo agent + tests

## Commands
- Tests: `.venv/bin/pytest tests/ -x -q`
- Lint: `.venv/bin/ruff check src/ tests/`
- Type check: `.venv/bin/mypy src/`
- Build: `.venv/bin/python -m build`

## Rules
- Python 3.10+ compatibility required
- All public functions need type hints
- Smallest possible diffs for bug fixes
- Do NOT move utility functions between modules
- Do NOT refactor unless explicitly asked
- Run tests + lint after every change

## Nightshift Build Contracts

When running as an overnight task, follow the contract for your assigned task type:

### bug-finder
- **Scope**: src/ only. Every function.
- **Find**: off-by-one, unhandled None, wrong exception types, logic errors, broken error handling.
- **Deliverable**: Fix each bug with smallest possible diff. Max 8 fixes per run.
- **Stop when**: All src/ files reviewed, OR 8 fixes made, OR you catch yourself moving code between files.
- **Validate**: `.venv/bin/pytest tests/ -x -q && .venv/bin/ruff check src/ tests/`
- **Do NOT touch**: Style, naming, imports, module organization, tests, docs. No new tests. No moving utilities.

### security-footgun
- **Scope**: src/ only.
- **Find**: shell injection, path traversal, unsafe deserialization, hardcoded secrets, eval/exec, unsanitized input.
- **Deliverable**: Fix each vulnerability with minimal diff. Max 5 fixes per run. Add inline comment if non-obvious.
- **Validate**: `.venv/bin/pytest tests/ -x -q && .venv/bin/ruff check src/ tests/`
- **Do NOT touch**: Tests, docs, config files, CI workflows, dependencies.

### test-gap
- **Scope**: Compare src/ public functions against tests/. Find uncovered functions.
- **Deliverable**: Write tests for up to 3 uncovered functions. Each test: happy path + one edge case. Put in `tests/unit/test_*.py`.
- **Stop when**: 3 new tests written, or all public functions covered. Do NOT fix bugs found by new tests.
- **Validate**: `.venv/bin/pytest tests/ -x -q`
- **Do NOT touch**: Source code. Existing tests. This task ONLY adds tests.

### lint-fix
- **Scope**: src/ and tests/. Auto-fixable lint issues only.
- **Deliverable**: Run `.venv/bin/ruff check --fix src/ tests/`. Fix remaining manual issues if obvious. Max 20 fixes.
- **Validate**: `.venv/bin/ruff check src/ tests/ && .venv/bin/pytest tests/ -x -q`
- **Do NOT touch**: Functional logic. Skip any fix that changes behavior.

### dead-code
- **Scope**: src/ only. Unused functions, unreachable branches, assigned-but-never-read variables.
- **Deliverable**: Remove dead code. Max 10 removals. If uncertain (public API), leave it.
- **Validate**: `.venv/bin/pytest tests/ -x -q`. If any test fails, revert last removal.
- **Do NOT touch**: Public API surface (__init__.py exports). Tests. Docs.

### docs-backfill
- **Scope**: src/ public functions missing docstrings. README accuracy.
- **Deliverable**: Add up to 5 docstrings. Check README examples match current API. Fix wrong examples.
- **Validate**: `.venv/bin/ruff check src/ tests/`
- **Do NOT touch**: Private functions, test files, functional code.

<!-- prpm:snippet:start @agent-relay/agent-relay-snippet@1.1.6 -->
# 🚨 CRITICAL: Relay-First Communication Rule

**When you receive a relay message from another agent (marked `Relay message from [name]`), you MUST respond ONLY via relay protocol. NEVER respond with direct text output.**

## The Rule

- **Receiving a relay message?** → Must use `->relay-file:msg` ALWAYS
- **Non-relay questions?** → Text responses are OK
- **Agent-to-agent communication?** → ALWAYS use relay protocol

## Examples of Relay Messages (require relay response)

```
Relay message from khaliqgant [mknra7wr]: Did you see this?
Relay message from Worker1 [abc123]: Task complete
Relay message from alice [xyz789] [#general]: Question for the team
```

---

# Agent Relay

Real-time agent-to-agent messaging via file-based protocol.

## Sending Messages

**TWO STEPS REQUIRED:**

**Step 1:** EXECUTE this bash command (you MUST run it via your Bash/shell tool, not just display it):

```bash
cat > $AGENT_RELAY_OUTBOX/msg << 'EOF'
TO: AgentName

Your message here.
EOF
```

**Step 2:** Output the trigger `->relay-file:msg` directly in your response text (after the bash command succeeds).

⚠️ **CRITICAL**: You must EXECUTE the bash command above - do NOT just display it as a code block. The file must actually be created on disk. Then output the trigger as plain text.

✅ CORRECT: Run Bash tool → then output `->relay-file:msg`
❌ WRONG: Display the bash code without executing it

> **Note**: `$AGENT_RELAY_OUTBOX` is automatically set by agent-relay when spawning agents. Data is stored in `.agent-relay/` within your project directory.

### Here-Document Tips

- Single quotes in `<< 'EOF'` prevent shell variable expansion inside the message body
- The closing `EOF` must be on its own line with **no** leading/trailing whitespace
- **Fallback** if heredocs fail in your shell:
  ```bash
  echo "TO: AgentName" > $AGENT_RELAY_OUTBOX/msg && echo "" >> $AGENT_RELAY_OUTBOX/msg && echo "Your message." >> $AGENT_RELAY_OUTBOX/msg
  ```

## Synchronous Messaging

By default, messages are fire-and-forget. Add `[await]` to block until the recipient ACKs:

```
->relay:AgentB [await] Please confirm
```

Custom timeout (seconds or minutes):

```
->relay:AgentB [await:30s] Please confirm
->relay:AgentB [await:5m] Please confirm
```

Recipients auto-ACK after processing when a correlation ID is present.

## Message Format

```
TO: Target
THREAD: optional-thread

Message body (everything after blank line)
```

| TO Value | Behavior |
|----------|----------|
| `AgentName` | Direct message |
| `*` | Broadcast to all |
| `#channel` | Channel message |

## Agent Naming (Local vs Bridge)

**Local communication** uses plain agent names. The `project:` prefix is **ONLY** for cross-project bridge mode.

| Context | Correct | Incorrect |
|---------|---------|-----------|
| Local (same project) | `TO: Lead` | `TO: project:lead` |
| Local (same project) | `TO: Worker1` | `TO: myproject:Worker1` |
| Bridge (cross-project) | `TO: frontend:Designer` | N/A |
| Bridge (to another lead) | `TO: otherproject:lead` | N/A |

**Common mistake**: Using `project:lead` when communicating locally. This will fail because the relay looks for an agent literally named "project:lead".

```bash
# CORRECT - local communication to Lead agent
cat > $AGENT_RELAY_OUTBOX/msg << 'EOF'
TO: Lead

Status update here.
EOF
```

```bash
# WRONG - project: prefix is only for bridge mode
cat > $AGENT_RELAY_OUTBOX/msg << 'EOF'
TO: project:lead

This will fail locally!
EOF
```

## Spawning & Releasing

**IMPORTANT**: The filename is always `spawn` (not `spawn-agentname`) and the trigger is always `->relay-file:spawn`. Spawn agents one at a time sequentially.

### CLI Options

The `CLI` header specifies which AI CLI to use. Valid values:

| CLI Value | Description |
|-----------|-------------|
| `claude` | Claude Code (Anthropic) |
| `codex` | Codex CLI (OpenAI) |
| `gemini` | Gemini CLI (Google) |
| `aider` | Aider coding assistant |
| `goose` | Goose AI assistant |

**Step 1:** EXECUTE this bash command (run it, don't just display it):
```bash
# Spawn a Claude agent
cat > $AGENT_RELAY_OUTBOX/spawn << 'EOF'
KIND: spawn
NAME: WorkerName
CLI: claude

Task description here.
EOF
```
**Step 2:** Output: `->relay-file:spawn`

```bash
# Spawn a Codex agent
cat > $AGENT_RELAY_OUTBOX/spawn << 'EOF'
KIND: spawn
NAME: CodexWorker
CLI: codex

Task description here.
EOF
```

**Step 1:** EXECUTE this bash command (run it, don't just display it):
```bash
# Release
cat > $AGENT_RELAY_OUTBOX/release << 'EOF'
KIND: release
NAME: WorkerName
EOF
```
**Step 2:** Output: `->relay-file:release`

## When You Are Spawned

If you were spawned by another agent:

1. **Check who spawned you**: `echo $AGENT_RELAY_SPAWNER`
2. **Your first message** is your task from your spawner - reply to THEM, not "spawner"
3. **Report status** to your spawner (your lead), not broadcast

```bash
# Check your spawner
echo "I was spawned by: $AGENT_RELAY_SPAWNER"
```

**Step 1:** EXECUTE this bash command:
```bash
# Reply to your spawner
cat > $AGENT_RELAY_OUTBOX/msg << 'EOF'
TO: $AGENT_RELAY_SPAWNER

ACK: Starting on the task.
EOF
```
**Step 2:** Output: `->relay-file:msg`

## Receiving Messages

Messages appear as:
```
Relay message from Alice [abc123]: Content here
```

Channel messages include `[#channel]`:
```
Relay message from Alice [abc123] [#general]: Hello!
```
Reply to the channel shown, not the sender.

## Protocol

- **ACK** when you receive a task: `ACK: Brief description of task received`
- **DONE** when complete: `DONE: What was accomplished`
- Send status to your **lead** (the agent in `$AGENT_RELAY_SPAWNER`), not broadcast

Example messages:
```
TO: Lead

ACK: Starting work on authentication module.
```
```
TO: Lead

DONE: Authentication module implemented with JWT support.
```

## Viewing Message History

Use `agent-relay history` to view previous messages:

```bash
agent-relay history                    # Last 50 messages
agent-relay history -n 20              # Last 20 messages
agent-relay history -f Lead            # Messages from Lead
agent-relay history -t Worker1         # Messages to Worker1
agent-relay history --thread task-123  # Messages in a thread
agent-relay history --since 1h         # Messages from the last hour
agent-relay history --json             # JSON output for parsing
```

## Headers Reference

| Header | Required | Description |
|--------|----------|-------------|
| TO | Yes (messages) | Target agent/channel |
| KIND | No | `message` (default), `spawn`, `release` |
| NAME | Yes (spawn/release) | Agent name |
| CLI | Yes (spawn) | CLI to use: `claude`, `codex`, `gemini`, `aider`, `goose` |
| THREAD | No | Thread identifier |
<!-- prpm:snippet:end @agent-relay/agent-relay-snippet@1.1.6 -->
