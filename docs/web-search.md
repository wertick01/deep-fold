# web_search: free Tavily (no card)

`deepfold chat --agent` can call `web_search`. Testers do **not** need an
account or a credit card: `--agent` turns search on with the tools. The
default backend is
[Tavily keyless](https://docs.tavily.com/documentation/keyless) (shared free
cap). Results are title / URL / snippet, not google.com ranking. Deepfold
does not scrape Google or DuckDuckGo HTML.

**In progress:** agent TTY chrome. Search wiring below still applies.

Never put tokens in git, chat, or a `.chr`.

Russian walkthrough: [`web-search.ru.md`](web-search.ru.md).
Contract: [`spec/agent.md`](spec/agent.md) §5.7.

---

## Testers

Until agent tools are on, the model does not see `web_search` and no
socket opens. `--agent` enables both. `--no-agent-web` keeps tools local.

```powershell
python -m gpu.cli chat --model D:\weights\Qwen2.5-14B-Instruct --agent --workspace C:\dev\deep-fold
```

Remember for later chats (writes `$DEEPFOLD_HOME/prefs.env`):

```text
/agent default on
```

Same as `DEEPFOLD_AGENT=1`. Undo with `/agent default off` or `--no-agent`.
14B is the agent plate. Trust `ask` confirms each search.

## Optional free Tavily key (HTTP 429)

Keyless is a shared pool. A free API key is typically 1 000 searches/month
and does **not** require a card: [app.tavily.com](https://app.tavily.com).

```powershell
[System.Environment]::SetEnvironmentVariable(
  "DEEPFOLD_TAVILY_KEY",
  "tvly-paste-here",
  "User")
```

Or UTF-8 `%LOCALAPPDATA%\deepfold\cse.env`:

```text
DEEPFOLD_TAVILY_KEY=tvly-paste-here
```

`TAVILY_API_KEY` is also read. Restart the terminal afterwards.

## Optional Brave / Google

| Order | When |
|---|---|
| 1. Brave | `DEEPFOLD_BRAVE_KEY` or `BRAVE_API_KEY` is set |
| 2. Google CSE | both `DEEPFOLD_GOOGLE_CSE_KEY` and `DEEPFOLD_GOOGLE_CSE_CX` |
| 3. Tavily | otherwise: Tavily key, else keyless |

Brave’s dashboard usually asks for a card before a plan activates — skip it
for product tests. Google’s Custom Search JSON API is closed to new Cloud
projects. The `cse.js` widget is not this tool.

## Errors

| Message | Fix |
|---|---|
| `web_search is off` | `--agent` (search follows), `--agent-web`, or `/agent web on` |
| Tavily HTTP 429 | free Tavily key (no card) |
| Tavily HTTP 401 / 403 | check `DEEPFOLD_TAVILY_KEY` |
| Brave 401 / 403 / 422 / 429 | Brave plan; unset the Brave key to fall back to Tavily |
| Google CSE HTTP 403 | new Cloud projects are refused; unset CSE keys |
