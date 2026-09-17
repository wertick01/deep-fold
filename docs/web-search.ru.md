# web_search: бесплатный поиск (Tavily)

Агент в `deepfold chat --agent` может искать в интернете инструментом
`web_search`. Чтобы **просто протестировать продукт**, карта и аккаунт
не нужны: достаточно `--agent` (поиск включается вместе с агентом).
По умолчанию идёт
[Tavily keyless](https://docs.tavily.com/documentation/keyless) — общий
бесплатный лимит, без регистрации.

Это не выдача google.com один-в-один. Результат — заголовок, URL, сниппет.
HTML google.com / DuckDuckGo Deepfold **не** скрейпит.

**Дорабатываются:** TTY / агентный layout. Поиск ниже по-прежнему включается флагом `--agent`.

Ключи (если заведёшь позже) **не** клади в git, в чат и в `.chr`.

English: [`web-search.md`](web-search.md). Контракт:
[`spec/agent.md`](spec/agent.md) §5.7.

---

## 1. Тестер: только флаг

Пока агент выключен, модель **не видит** `web_search` и сокет не
открывается. `--agent` включает инструменты и поиск. Чтобы поиск не
включался: `--no-agent-web` или `/agent web off`.

```powershell
python -m gpu.cli chat --model D:\weights\Qwen2.5-14B-Instruct --agent --workspace C:\dev\deep-fold
```

Подставь свой каталог модели. 14B для инструментов надёжнее 3B.

Чтобы каждый следующий `chat` стартовал так же, без флагов:

```text
/agent default on
```

Это пишет `%LOCALAPPDATA%\deepfold\prefs.env` (`DEEPFOLD_AGENT=1`). Env
`DEEPFOLD_AGENT=1` делает то же самое. Снять: `/agent default off` или
`--no-agent`.

Уже внутри сессии:

```text
/agent on
/agent web on
```

На trust `ask` каждый поиск спросит `y/N`. Для сессии без вопросов:
`--agent-trust workspace` или `/agent trust workspace`.

Попроси в чате факт, которого нет в репозитории. В логе инструментов должно
мелькнуть `web_search …`, в результате — нумерованные URL.

---

## 2. Если keyless упёрся в лимит (HTTP 429)

Общий keyless-пул общий на всех. Карту по-прежнему не просят.

1. Заведи бесплатный ключ на [app.tavily.com](https://app.tavily.com)
   (обычно 1000 поисков/месяц, без карты).
2. Положи его в env или в `%LOCALAPPDATA%\deepfold\cse.env` (файл уже
   gitignored):

```powershell
[System.Environment]::SetEnvironmentVariable(
  "DEEPFOLD_TAVILY_KEY",
  "tvly-вставь-сюда",
  "User")
```

Или строка в файле:

```text
DEEPFOLD_TAVILY_KEY=tvly-вставь-сюда
```

Закрой и заново открой терминал и Cursor. Алиас `TAVILY_API_KEY` тоже
читается; предпочтительнее `DEEPFOLD_TAVILY_KEY`.

---

## 3. Опционально: Brave / Google

Если **уже есть** ключ, агент его возьмёт:

| Приоритет | Когда |
|---|---|
| 1. Brave | задан `DEEPFOLD_BRAVE_KEY` (или `BRAVE_API_KEY`) |
| 2. Google CSE | заданы **оба** `DEEPFOLD_GOOGLE_CSE_KEY` и `DEEPFOLD_GOOGLE_CSE_CX` |
| 3. Tavily | иначе: ключ Tavily или keyless |

Brave Search API при регистрации обычно **сразу просит карту** — для теста
продукта это не путь. Google Custom Search JSON API
[закрыт для новых Cloud-проектов](https://developers.google.com/custom-search/v1/overview).
Виджет `cse.js` к агенту не подключается.

---

## 4. Ошибки

| Сообщение | Что сделать |
|---|---|
| `web_search is off` | `--agent` (поиск идёт с агентом), `--agent-web`, или `/agent web on` |
| Tavily HTTP 429 | шаг 2 (бесплатный ключ, без карты) |
| Tavily HTTP 401 / 403 | проверь `DEEPFOLD_TAVILY_KEY` |
| Brave HTTP 401 / 403 / 422 / 429 | план Brave; для теста продукта убери ключ Brave, чтобы снова пошёл Tavily |
| Google CSE HTTP 403 | новый Cloud-проект JSON API не выдаёт; убери CSE-ключи |

Живой запрос эта страница сама не делает.
