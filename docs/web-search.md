# web_search: Brave Search

`deepfold chat --agent` can call `web_search`. Google’s Custom Search JSON API
is [closed to new customers](https://developers.google.com/custom-search/v1/overview).
The path that still signs up is the
[Brave Search API](https://api-dashboard.search.brave.com/documentation/quickstart).
Results are title / URL / snippet, not google.com ranking.

Brave’s own quickstart requires a card to **activate a plan**. The Search plan
then typically includes monthly credits (about 1 000 queries). No plan → no key.

Never put the token in git, chat, or a `.chr`.

Russian walkthrough (click-by-click, this machine): [`web-search.ru.md`](web-search.ru.md).

---

## 1. Account

1. [Register](https://api-dashboard.search.brave.com/register).
2. Confirm the email.
3. Open the [dashboard](https://api-dashboard.search.brave.com/).

## 2. Search plan

1. [Plans](https://api-dashboard.search.brave.com/app/plans).
2. Activate **Search** (includes Web Search).
3. Enter a card when asked.
4. Wait until the plan is active.

Pricing: [docs](https://api-dashboard.search.brave.com/documentation/pricing).

## 3. API key

1. **API Keys** → **Add API Key** (name e.g. `deepfold`).
2. Copy the token once. If lost, revoke and create another.

## 4. Store it (Windows)

User env (restart Cursor/terminals afterwards):

```powershell
[System.Environment]::SetEnvironmentVariable(
  "DEEPFOLD_BRAVE_KEY",
  "paste-token-here",
  "User")
```

Or UTF-8 `%LOCALAPPDATA%\deepfold\cse.env`:

```text
DEEPFOLD_BRAVE_KEY=paste-token-here
```

`BRAVE_API_KEY` is also read. Env wins over the file.

## 5. Opt in

```powershell
python -m gpu.cli chat --model D:\weights\Qwen2.5-14B-Instruct --agent --agent-web --workspace C:\dev\deep-fold
```

Or `/agent web on` in an existing session. 14B is the agent plate. Trust `ask`
confirms each search.

## 6. Errors

| Message | Fix |
|---|---|
| `web_search is off` | `--agent-web` / `/agent web on` |
| `needs DEEPFOLD_BRAVE_KEY` | step 4 + new terminal |
| HTTP 401 / 403 | new key; plan active |
| HTTP 422 | plan missing Web Search |
| HTTP 429 | credits / rate limit |

If **both** Google CSE env vars are set and Brave is not, the old CSE JSON API
is used. If Brave is set, Brave wins.

Contract: [`spec/agent.md`](spec/agent.md) §5.7.
