# Agent mode

Canonical design for `deepfold chat --agent`. English is the source of
truth. Implementation follows this page.

This is a **local coding agent** that uses the same NF4 TokenLoop as
ordinary chat. The UX bar is Cursor / Claude Code: search the tree, read
with line numbers, patch files, run tests, iterate until the request is
done. The runtime bar is Deepfold: one Ampere GPU, packed `.chr`,
sandboxed workspace, no cloud round-trip.

Wave A (tools, stop, hide XML) and Wave C (trust, `run_argv`, `AGENTS.md`)
are in [`gpu/cli/agent.py`](../../gpu/cli/agent.py) and
[`gpu/cli/chat.py`](../../gpu/cli/chat.py). Wave B (session KV) is **not**:
`TokenLoop.generate` still resets and prefills the whole chat each turn.
Shipping the remaining waves does not require an IDE plugin. The TTY is
the first surface. An editor extension is a later client of the same
session, not a rewrite.

---

## 0. Claim

A Qwen2.5-14B (resident NF4) or Qwen2.5-32B (overflow) session can
inspect, edit, and test a software workspace on the user’s GPU with the
same *kinds* of moves a Cursor agent uses: glob, grep, targeted patches,
allowlisted commands, git read, a visible plan, and many tool rounds
without re-encoding the whole chat every time.

3B is for smoke and chat, not for the agent plate. Ada generate stays
experimental. Linux tok/s stay unpublished until a plate exists.

We do **not** wrap Cursor, Claude Code, VS Code, or llama.cpp. We do
**not** default to an unrestricted shell. We **do** treat “IDE-class
agent on this kernel” as the product, not a side flag.

---

## 1. What git does today

Files: [`gpu/cli/agent.py`](../../gpu/cli/agent.py),
[`gpu/cli/chat.py`](../../gpu/cli/chat.py).

The four-tool v1 loop (`list_dir` / `read_file` / `write_file` /
`run_tests`) is the compatibility core. Search, patch, git read,
allowlisted argv, todos, stop-on-tool, stream hiding, and
`--agent-trust` are on that core now. Session KV is not.

| Piece | Behavior |
|---|---|
| Tools | `list_dir`, `read_file`, `glob`, `grep`, `git_status`, `git_diff`, `str_replace`, `write_file`, `delete_file`, `run_tests`, `run_argv`, `todo` |
| Protocol | Qwen XML `<tool_call>…</tool_call>` plus optional `apply_chat_template(..., tools=)` |
| History | Flattened to `user`/`assistant` strings; tool results become `<tool_response>` user turns |
| GPU | `TokenLoop.generate` still calls `reset()` then prefills **all** prompt ids (Wave B) |
| Safety | Paths must resolve under `--workspace`; `.git` blocked; confirm follows `--agent-trust` (`ask` default). `run_argv` is an allowlist, not a shell |
| Caps | `--max-tool-rounds` default 24; read 400 lines / 1 MiB; write 256 KiB; pytest 120 s; grep 50 files / 80 hits |
| Stream | Hide `<tool_call>` XML; stop decode when a call parses; truncated JSON retried once |
| Failure | Degenerate `!!!!` tool spam retried once |

The remaining product gap is **session KV**, not “we only have four
tools.” 14B is the agent plate; 3B warns.

`deepfold run --prompt` stays non-agent. `--agent` cannot combine with
`--raw`. Those two rules stay.

---

## 2. Target loop

One user turn:

```
render chat (+ tools schema) → token ids
  if KV empty or prefix mismatch: reset, prefill all ids
  else: prefill only the new suffix at kv.seq_len
decode until EOS or a complete tool call (stop sequence)
  stream visible text; hide raw XML
if tool_calls:
  run them (reads parallel, writes serial)
  append tool results as tokens (suffix prefill)
  decode again
until no tool_calls or round cap or interrupt
```

The GPU holds **one** TokenLoop for the process. Tools that do not need
the GPU (grep, glob, git, pytest) run on the host. Several *read* tools
from one assistant message may run in parallel. Writes and commands stay
serial.

There is no second model on the same card. A 3070 8 GB cannot host a
14B agent and a 3B draft together. Same-GPU draft stays forbidden
([`gpu/loop/generate.py`](../../gpu/loop/generate.py) `draft="cpu"` is
lab-only and not a CLI flag).

---

## 3. Session KV (load-bearing)

Today every chat turn and every tool round is a cold prefill. Agent
traces grow fast (system + files + patches + pytest). On 14B that is
the difference between a coding session and a slideshow.

`KVCache` is already a fixed `[n_layers, max_seq, n_kv_heads, head_dim]`
allocation. `reset()` only sets `seq_len = 0`. Prefill already writes
slots `start_pos .. start_pos+n`. The missing API is “do not start at
zero.”

### 3.1 TokenLoop

Keep `generate()` as the one-shot path (`run --prompt`, tests). It
still `reset()`s.

Add a session path used only by `chat`:

```
loop.reset()                         # /clear, /new, prefix miss
logits = loop.prefill_from(ids, 0)   # first prompt in the process

# later, ids_all is the newly rendered template
delta = suffix_after(prefix_ids, ids_all)
if delta is None:
    loop.reset()
    logits = loop.prefill_from(ids_all, 0)
else:
    logits = loop.prefill_from(delta, loop.kv.seq_len)

gen = loop.decode_from_logits(logits, max_new=..., stop=..., on_token=...)
prefix_ids = ids_all + gen.tokens     # exact ids sitting in KV
```

`prefill_from(ids, start_pos)` is `forward` chunked the same way as
`prefill`, but the first chunk uses `start_pos` instead of `0`.
`decode_from_logits` is the existing greedy `step()` loop without the
leading `reset`/`prefill`.

Do not zero the KV tensor on reset. Do not `torch.cat` the cache
([`gpu/loop/kv_cache.py`](../../gpu/loop/kv_cache.py)).

### 3.2 Prefix stability

Chat templates can rewrite earlier bytes when a new message is added.
Never append a suffix unless the new rendering is a **token-for-token**
prefix match:

```
def suffix_after(old: Tensor, new: Tensor) -> Tensor | None:
    n = int(old.numel())
    if int(new.numel()) < n:
        return None
    if not torch.equal(old, new[:n]):
        return None
    return new[n:]
```

Store `old` on CPU. On mismatch: full prefill, stderr one line
`agent: KV prefix miss, full prefill`. That is a fallback, not a
failure of the turn.

Qwen2.5 `apply_chat_template` is expected to be prefix-stable when
messages are only appended. If a tokenizer is not, agent still works,
just slower.

### 3.3 `/clear` and `/new`

Clear JSON history **and** `loop.reset()`. Resume from `/chats` is a
full prefill (disk has text, not KV).

### 3.4 `max_seq` for agent

Chat default stays 2048 for plain talk. `--agent` should pick a budget
from VRAM, not a slogan:

| Card (after weights + 1800 MiB overhead) | Agent `--max-seq` default |
|---|---|
| 8 GB class (RTX 3070) + 14B NF4 | 2048 |
| 12 GB class (RTX 3080) + 14B NF4 | 4096 |
| 12 GB + 32B overflow | 2048 (KV fights CopyRing) |
| User flag | always wins |

Qwen2.5-14B GQA KV is on the order of
`2 * n_layers * n_kv_heads * head_dim * 2` bytes per token (~0.19 MiB/tok
at 48 / 8 / 128). 8192 context on an 8 GB 14B load is an OOM, not a
gift. Doctor / chat banner prints the chosen `max_seq` and current
`kv.seq_len`.

When `kv.seq_len` hits the cap: stop the turn, tell the user to `/clear`
or raise `--max-seq`. Do not silently drop the system prompt.

---

## 4. Tool protocol

Keep Qwen’s on-disk format: one JSON object inside
`<tool_call>…</tool_call>`. Several blocks in one assistant message are
several calls.

### 4.1 Stop on a complete call

v1 decodes until EOS or `max_new_tokens`, then parses. The model often
emits a valid call and then junk (`!!!!`, extra prose).

v2 decode **stops** when the output ends with `</tool_call>` (UTF-8
match on the decoded window, or an equivalent token-id sequence). Then
parse. Visible stream does not show the XML; the TUI prints
`tool grep path=gpu/cli query=TokenLoop`.

`TokenLoop.generate` already takes `stop: Sequence[int]` for single
tokens (EOS). Agent needs a **multi-token / decoded-suffix** stop.
Implement it in the chat callback (`on_token` already aborts on `!!!!`)
or as a loop-level stop string. Do not wait for EOS after a closed
tool call.

### 4.2 Native roles

When `tokenizer.apply_chat_template(..., tools=SCHEMAS)` works, pass
HuggingFace-style messages:

```
{"role": "assistant", "content": "", "tool_calls": [
    {"type": "function", "function": {"name": "grep", "arguments": "{...}"}}
]}
{"role": "tool", "name": "grep", "content": "<result>"}
```

Keep today’s flatten-to-XML fallback when `tools=` raises `TypeError`
(old tokenizers). Tests already cover flatten.

### 4.3 Repair

If `<tool_call>` opens and JSON is truncated at `max_new_tokens`, one
retry with a short user ping: `continue the tool call JSON`. Same as
v1’s degenerate retry, but for truncated JSON rather than punctuation
spam. After two failures, show the raw text and stop the round.

---

## 5. Tool surface

All paths still go through `resolve_under(workspace, rel)`. Home paths,
`..` escapes, and `.git` stay errors. `_SKIP_DIR` stays
(`.git`, `__pycache__`, `.venv`, `node_modules`, caches).

### 5.1 Read — no confirm

| Tool | Contract |
|---|---|
| `list_dir` | v1. Max 200 entries. |
| `read_file` | v1. Numbered lines, `offset`/`limit`, 1 MiB / 400 lines. |
| `glob` | `pattern` relative to workspace (e.g. `gpu/cli/*.py`). Max 200 paths. No recursive `**` outside workspace. |
| `grep` | `query` regex, optional `path`, optional `glob`. Cap: 50 files, 80 matches, 200 chars/line. Prefer `rg` on PATH; else a Python walk. Skip binaries (`\0` in first 4 KiB). |
| `git_status` | `git status --porcelain` in the workspace if `.git` exists; else a clear error. No flags from the model. |
| `git_diff` | `git diff --` plus an optional in-workspace path. No `--no-index` tricks, no pager. Truncate like other tools (12 KiB). |

These are how a Cursor-class agent finds code instead of `list_dir` ping-pong.

### 5.2 Edit — confirm by policy (§6)

| Tool | Contract |
|---|---|
| `str_replace` | `path`, `old_string`, `new_string`. `old_string` must match **exactly once**. If 0 or ≥2 matches: error, include a short hint (count). This is the default edit. |
| `write_file` | v1. Create or replace a whole file. Prefer `str_replace` in the system prompt. Cap 256 KiB. |
| `delete_file` | In-workspace file only, not directories. |

Show a unified-diff preview on stderr before confirm. After a successful
edit, the tool result is `ok` plus the diff hunk, not the whole file.

### 5.3 Run — confirm by policy, argv only

| Tool | Contract |
|---|---|
| `run_tests` | v1. `python -m pytest -q --tb=short --color=no -- <path>`. Timeout 120 s. |
| `run_argv` | JSON array of arguments, cwd = workspace. **No shell.** No `cmd.exe`, no `powershell`, no `bash -c`. |

`run_argv` allowlist (name of `argv[0]`, after resolving `.venv/Scripts`
and PATH inside the workspace):

```
python, pytest, ruff, go, git, deepfold
```

`git` here is extra surface for `add`/`commit` and still goes through
confirm. `git push` is refused by name (network). `python` may only
receive `-m …` or a workspace `.py` path, not `-c`. Environment is a
copy of the parent minus `HTTP(S)_PROXY` overrides the user did not
ask for; do not pass secrets files.

This is a terminal, not a login shell. Cursor’s agent terminal is
powerful because the human is watching; ours is the same idea with an
allowlist so a 14B cannot `Format-Volume`.

Unknown `argv[0]` → error listing the allowlist. Timeout 120 s.
Stdout+stderr clipped to 12 KiB.

### 5.4 Plan

| Tool | Contract |
|---|---|
| `todo` | `items`: list of `{id, content, status}` with `pending` / `in_progress` / `done`. Stored on the `AgentSession`, shown in the toolbar. Not a file unless the model writes one. |

The system prompt tells the model to keep a plan for work that is more
than one edit. This is Cursor’s todo list, not a second agent.

### 5.5 Parallelism

One assistant message may contain several tool calls.

- All-read (`list_dir` / `read_file` / `glob` / `grep` / `git_*`): run
  concurrently, preserve listed order in the history.
- Any edit or `run_*`: run the whole message serially, in order.
- Mixed read+write in one message: serial.
- `web_search` is not a workspace read: serial, and only if opted in (§5.7).

### 5.6 Out of v2

Not because they are impossible. Because they are a different client or
a second GPU:

- Unrestricted shell
- Browser / GUI automation
- MCP host (v3: stdio client to user-specified servers, same sandbox
  rules for file roots they expose)
- Multi-agent swarm (one TokenLoop)
- LSP / language server
- Cloud tools other than opt-in `web_search` (§5.7)

v3 MCP is one process speaking JSON-RPC to a server the **user**
launched. Deepfold does not download random MCP servers.

### 5.7 Web search (opt-in, off by default)

`web_search` is not a browser and not HTML scraping.

**Default backend for new users:** [Brave Search API](https://api-dashboard.search.brave.com/documentation/quickstart)
(`GET /res/v1/web/search`). How to get a key: [`web-search.md`](../web-search.md)
(English), [`web-search.ru.md`](../web-search.ru.md) (step-by-step).

**Fallback:** Google Custom Search JSON API when *both* CSE env vars are set
and Brave is not. That JSON API is closed to new Cloud projects.

| | |
|---|---|
| Tool | `web_search` `{query, num?}` |
| Default | **Off.** Not in the advertised schema until `--agent-web` or `/agent web on`. |
| Keys | `DEEPFOLD_BRAVE_KEY` (or `BRAVE_API_KEY`), else Google `DEEPFOLD_GOOGLE_CSE_KEY` + `DEEPFOLD_GOOGLE_CSE_CX`. Env or `$DEEPFOLD_HOME/cse.env`. Never in git, `.chr`, or chat JSON. |
| Confirm | Same as `run_argv`: ask at `ask`/`write`, auto at `workspace`. |
| Caps | Query 200 chars; 1–8 hits (default 5); 15 s; result clipped to 12 KiB. |
| Result | Numbered title / URL / snippet. Cite those URLs; do not invent links. |
| Prefer | If Brave is set, Brave wins even when CSE vars exist. |

Without the flag, `execute("web_search")` returns an error and does
**not** open a socket. Missing token: error, no HTTP. This is not
DuckDuckGo Instant Answer, Bing, Google HTML scrape, or the
`<script src=cse.js>` widget.

---

## 6. Permissions

Cursor’s trust dialog is the model: reads are cheap, writes are
visible, the human can raise trust for a session.

| Level | Reads | `str_replace` / `write_file` / `delete_file` | `run_tests` / `run_argv` / `web_search` |
|---|---|---|---|
| `ask` (default) | auto | ask | ask |
| `write` | auto | auto | ask |
| `workspace` | auto | auto | auto **inside the allowlist** |

CLI: `--agent-trust {ask,write,workspace}` (default `ask`).
In-session: `/agent trust ask|write|workspace`.

Always ask (even at `workspace`):

- `run_argv` whose `argv[0]` is `git` and subcommand is not a read
- anything that would leave the workspace (must not happen if
  `resolve_under` holds; if it would, refuse, do not ask)

`/agent off` drops tools and the agent system prompt; KV prefix will
miss → one full prefill.

---

## 7. System prompt and workspace rules

Keep a short built-in prompt (workspace root, tool JSON, “read before
rewrite”, “prefer `str_replace`”, “do not invent file contents”).

If the workspace contains **`AGENTS.md`** or **`.deepfold/instructions.md`**
(first existing file, UTF-8, cap 8 KiB), append it. That is Cursor’s
project rules, as a file the user owns. Deepfold does not fetch rules
from the network.

14B is the documented agent model. If `config.json` looks like 3B
(`hidden_size` 2048 and ~3e9 params) and `--agent` is on, print a
warn once: tool JSON is unreliable; continue anyway.

---

## 8. TTY

Still `prompt_toolkit`. Additions:

- Banner: `agent on  trust=ask  workspace=…  kv 0/4096`
- After each tool: dim `tool …` and a one-line preview (v1 already)
- Diff preview before an edit confirm
- `/stats` includes `kv.seq_len`, tool rounds this turn, prefix hits vs
  misses
- `/agent` without args prints status
- Stream markdown for assistant prose; never stream raw `<tool_call>`
  (buffer until the tag closes or the turn ends)

History JSON under `$DEEPFOLD_HOME/chats` grows a `tool_calls` field as
today. Forward-compatible: extra keys ignored.

---

## 9. Waves

One flag (`--agent`) when a wave lands. Do not keep `--agent-v1`
forever. Land in this order so each wave is usable alone.

### Wave A — tools and stop (no TokenLoop change) — landed in CLI

`glob`, `grep`, `str_replace`, `delete_file`, `git_status`, `git_diff`,
`todo`. Stop decode on `</tool_call>`. Hide XML in the stream. Diff
preview. Default `--max-tool-rounds` **24**. Warn on 3B. Tests for
sandbox, unique `str_replace`, grep caps.

This already changes the product: the model can search and patch.

### Wave B — session KV — **not landed**

`prefill_from` / suffix check / `decode_from_logits`. `/clear` resets
KV. Agent `max_seq` policy in §3.4. Prefix-miss fallback. CPU tests
for `suffix_after`; live test: two-turn chat tok/s and prefill_ms drop
on turn 2 when the prefix hits.

`run --prompt` and `generate()` stay cold-prefill.

### Wave C — terminal and trust — landed in CLI

`run_argv` allowlist. `--agent-trust` / `/agent trust`. Parallel
read-only tools. `AGENTS.md` injection.

### Wave D — later clients (not a blocker)

MCP stdio (user-launched). HTTP JSON session for an editor plugin
(same tools, same sandbox). Neither is required to call v2 done.

---

## 10. Acceptance

v2 is done when all of the following are true on a 14B NF4 chat:

1. `grep` + `str_replace` + `run_tests` can fix a failing test in the
   workspace without `write_file` of the whole module (human confirms
   at `ask`).
2. A second user turn in the same process does **not** redo prefill of
   the first prompt when the template prefix matches. `prefill_ms` and
   `prompt_len` in `/stats` show the suffix only.
3. A closed `<tool_call>` stops decode; punctuation spam after a call
   does not appear in the saved assistant text.
4. `..`, `~`, and `.git` still refuse. `run_argv` `["powershell", …]`
   refuses. `write_file` outside workspace refuses.
5. 3B `--agent` still starts, with the warn. 14B is the plate in docs.

Do not quote 3080 tok/s as an agent metric. Agent metrics are prefix
hit rate, rounds-to-green-test, and tool-JSON parse rate.

---

## 11. Mapping to Cursor / Claude Code

Honest overlap, not a disclaimer.

| Move | Cursor / Claude Code | Deepfold v2 |
|---|---|---|
| Search | grep / glob / semantic | `grep`, `glob` (no embedding index in v2) |
| Read | numbered, ranged | `read_file` |
| Edit | apply_patch | `str_replace` (+ `write_file` for new files) |
| Terminal | user-visible shell | `run_argv` allowlist + `run_tests` |
| Git | status/diff/commit | status/diff; commit via allowlisted `git` + confirm |
| Plan | todo | `todo` |
| Rules | `AGENTS.md` | same filename in the workspace |
| Multi-step | many tool rounds, KV lives | Wave B session KV |
| IDE | VS Code / JetBrains | TTY first; Wave D HTTP client |
| Browser | yes (Cursor) | no |
| Cloud model | yes | no; local `.chr` only |
| Subagents | yes | no (one TokenLoop) |

The gap that is **ours to close** is the left column of search / patch /
session / terminal. The gap that is **a different product** is hosting
inside VS Code and driving a browser. Closing the first gap is the
work. The second gap is a client, not a reason to keep four tools.

---

## 12. Code layout (when implementing)

| Module | Owns |
|---|---|
| `gpu/cli/agent.py` | schemas, parse, sandbox, execute, `suffix_after` for ids stays next to chat or a tiny `gpu/cli/session_ids.py` |
| `gpu/cli/chat.py` | TTY, trust, stream hiding, round loop, KV prefix |
| `gpu/loop/generate.py` | `prefill_from`, decode without reset; `generate()` unchanged |
| `gpu/cli/messages.py` | help text; 3B warn |
| `gpu/cli/test_cli.py` | CPU acceptance for parse/sandbox/trust/argv |
| live | 14B `--agent` on a fixture repo with a broken test |

Do not put tool subprocesses in `gpu/loop`. Do not import `transformers`
generate for the agent path. Do not add a second CUDA context.

---

## 13. Flags (v2)

- `--agent`
- `--workspace`
- `--max-tool-rounds` (default 24)
- `--agent-trust {ask,write,workspace}`
- `--agent-web` (`web_search`; Brave or Google CSE; off by default)

No `--shell`. No `--mcp` until Wave D.
