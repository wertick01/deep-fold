# web_search: Brave Search (пошагово)

Агент в `deepfold chat --agent` может искать в интернете инструментом
`web_search`. Для **новых** пользователей Google Custom Search JSON API
[закрыт](https://developers.google.com/custom-search/v1/overview). Рабочий
путь — [Brave Search API](https://api-dashboard.search.brave.com/documentation/quickstart).
Это не выдача google.com один-в-один, но это настоящий веб-поиск: заголовок,
URL, сниппет.

Официальный quickstart Brave: карточка нужна, чтобы **включить план**. После
активации Search обычно дают месячные кредиты (порядка 1000 запросов). Без
плана ключ не выдают.

Ключ **не** клади в git, в чат и в `.chr`.

---

## 1. Аккаунт

1. Открой [регистрацию Brave Search API](https://api-dashboard.search.brave.com/register).
2. Укажи email и пароль.
3. Подтверди почту по ссылке из письма.
4. Войди в [дашборд](https://api-dashboard.search.brave.com/).

---

## 2. План Search

1. В дашборде открой [Plans](https://api-dashboard.search.brave.com/app/plans)
   (меню слева → Plans).
2. Выбери план **Search** (в нём есть Web Search; LLM Context для Deepfold не
   обязателен).
3. Введи карту, как просит форма. Без этого шаг «API Keys» часто пустой.
4. Дождись, пока план станет active.

Цены и кредиты: [pricing](https://api-dashboard.search.brave.com/documentation/pricing).
Следи за расходом в дашборде, не в Deepfold.

---

## 3. Ключ

1. Меню → **API Keys**
   ([прямая ссылка](https://api-dashboard.search.brave.com/app/api-keys), путь
   может чуть отличаться).
2. **Add API Key**.
3. Имя, например `deepfold-3080`. Это ярлык, не сам секрет.
4. Скопируй токен **один раз**. Его показывают при создании. Если потерял —
   revoke и сделай новый, старый не восстановить.
5. Не вставляй ключ в Cursor-чат и не коммить.

---

## 4. Куда положить ключ на этом ПК (Windows)

Нужна переменная `DEEPFOLD_BRAVE_KEY`. Достаточно **одного** из двух способов.
После записи **закрой и заново открой** терминал и Cursor: уже запущенные окна
старый env не подхватят.

### Способ A — пользовательская env (удобно)

PowerShell **от твоего пользователя**, не от администратора:

```powershell
[System.Environment]::SetEnvironmentVariable(
  "DEEPFOLD_BRAVE_KEY",
  "вставь-токен-сюда",
  "User")
```

Проверка, что строка есть (значение не печатаем):

```powershell
[bool][System.Environment]::GetEnvironmentVariable("DEEPFOLD_BRAVE_KEY", "User")
```

Должно быть `True`.

### Способ B — файл кэша Deepfold

Файл `%LOCALAPPDATA%\deepfold\cse.env` уже gitignored. UTF-8, одна строка на
ключ:

```text
DEEPFOLD_BRAVE_KEY=вставь-токен-сюда
```

Если в файле уже есть `DEEPFOLD_GOOGLE_CSE_CX` — не стирай, просто добавь
строку Brave. Env выигрывает у файла, если заданы оба.

Алиас `BRAVE_API_KEY` тоже читается (как в доке Brave). Предпочтительнее
`DEEPFOLD_BRAVE_KEY`.

---

## 5. Включить инструмент в чате

Пока нет `--agent-web` (или `/agent web on`), модель **не видит** `web_search` и
сокет не открывается.

```powershell
python -m gpu.cli chat --model D:\weights\Qwen2.5-14B-Instruct --agent --agent-web --workspace C:\dev\deep-fold
```

Подставь свой каталог модели. 14B для инструментов надёжнее 3B.

Уже внутри сессии:

```text
/agent web on
```

На trust `ask` каждый поиск спросит `y/N`. Для сессии без вопросов:
`--agent-trust workspace` или `/agent trust workspace`.

---

## 6. Как понять, что заработало

Попроси в чате что-то, чего нет в репозитории, например свежую дату релиза
библиотеки. В логе инструментов должно мелькнуть `web_search …`, в результате —
нумерованные URL.

Типичные отказы (ключ в текст ошибки не попадает):

| Сообщение | Что сделать |
|---|---|
| `web_search is off` | `--agent-web` или `/agent web on` |
| `needs DEEPFOLD_BRAVE_KEY` | шаг 4, потом новый терминал |
| HTTP 401 / 403 | новый ключ, план active |
| HTTP 422 | план Search не включает Web Search |
| HTTP 429 | кредиты/лимит; подожди или пополни в дашборде |

Живой запрос из этой инструкции Deepfold сам не делает: ключ должен появиться
у тебя, потом уже чат.

---

## 7. Google CSE

Ветка Google остаётся: если заданы **оба** `DEEPFOLD_GOOGLE_CSE_KEY` и
`DEEPFOLD_GOOGLE_CSE_CX` **и нет** ключа Brave, используется Custom Search JSON
API. Для нового Cloud-проекта это обычно 403. Если задан Brave — берётся Brave.

Виджет `cse.js` / `<div class="gcse-search">` к агенту не подключается.

English: [`web-search.md`](web-search.md). Контракт инструмента:
[`spec/agent.md`](spec/agent.md) §5.7.
