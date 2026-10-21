"""Консольный текстовый AI-помощник (ProxyAPI).

Возможности:
  * обменивается сообщениями через ProxyAPI и корректно реагирует на команду выхода;
  * хранит и подгружает историю диалога между запросами;
  * переключается между 2 режимами:
      1) обычная модель  — OpenAI-совместимый Chat Completions (/v1/chat/completions);
      2) думающая модель — Claude 4.5 Sonnet через Anthropic Messages (/v1/messages)
         с отображением показателей reasoning (блоки размышлений + расход токенов);
  * конфигурация через .env; обработка ошибок запросов и таймаутов; понятные логи.

Документация ProxyAPI: https://proxyapi.ru/docs — единый адрес api.proxyapi.ru/v1,
идентификаторы моделей с префиксом вендора (openai/..., anthropic/...).
"""

import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Красивый вывод эмодзи в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Базовые адреса ProxyAPI (см. https://proxyapi.ru/docs/api-overview)
PROXYAPI_BASE_URL_OPENAI = "https://api.proxyapi.ru/v1"     # OpenAI-совместимый API
PROXYAPI_BASE_URL_ANTHROPIC = "https://api.proxyapi.ru"     # SDK сам допишет /v1/messages

SYSTEM_PROMPT = (
    "Ты полезный ассистент. Отвечай на русском языке, "
    "кратко и по существу."
)

# Таймауты запросов, секунды
REQUEST_TIMEOUT = 120.0

# Защита от некорректного бюджета thinking: max_tokens обязан быть больше бюджета
MAX_TOKENS = 4096


@dataclass
class Settings:
    """Настройки, загружаемые из .env с понятными дефолтами."""

    key: str = ""
    regular_model: str = "openai/gpt-5-mini"
    thinking_model: str = "anthropic/claude-sonnet-4-5"
    thinking_budget: int = 1500

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("PROXYAPI_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "Не задан ключ API. Скопируйте .env.example в .env "
                "и впишите PROXYAPI_KEY=<ваш ключ> из https://console.proxyapi.ru/keys"
            )
        return cls(
            key=key,
            regular_model=os.getenv("REGULAR_MODEL", "openai/gpt-5-mini").strip(),
            thinking_model=os.getenv("THINKING_MODEL", "anthropic/claude-sonnet-4-5").strip(),
            thinking_budget=int(os.getenv("THINKING_BUDGET_TOKENS", "1500")),
        )


@dataclass
class TextAgent:
    """Умный текстовый помощник с сохранением контекста диалога."""

    settings: Settings
    mode: str = "thinking"  # "thinking" (по умолчанию) | "regular"
    history: list = field(default_factory=list)  # история диалога между запросами

    def __post_init__(self) -> None:
        # Ленивые импорты, чтобы понятные ошибки приходили даже без установленных SDK
        from openai import OpenAI
        from anthropic import Anthropic

        self._openai = OpenAI(
            api_key=self.settings.key,
            base_url=PROXYAPI_BASE_URL_OPENAI,
            timeout=REQUEST_TIMEOUT,
        )
        self._anthropic = Anthropic(
            api_key=self.settings.key,
            base_url=PROXYAPI_BASE_URL_ANTHROPIC,
            timeout=REQUEST_TIMEOUT,
        )

    # ---------- Основной интерфейс ----------

    def respond(self, user_text: str) -> str:
        """Отправить сообщение модели и получить ответ (с учётом истории)."""
        if self.mode == "thinking":
            answer = self._respond_thinking(user_text)
        else:
            answer = self._respond_regular(user_text)
        return answer

    # ---------- Режим 1: обычная модель (Chat Completions) ----------

    def _respond_regular(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        try:
            resp = self._openai.chat.completions.create(
                model=self.settings.regular_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self.history],
                # reasoning_effort работает и с моделями других вендоров;
                # для обычного режима размышления не нужны — экономим токены.
                # reasoning_effort="low",
            )
        except Exception:
            # Не портим историю неудачным запросом
            self.history.pop()
            raise
        content = resp.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": content})
        usage = getattr(resp, "usage", None)
        if usage:
            print(
                f"   [токены: вход {usage.prompt_tokens} / выход {usage.completion_tokens}]"
            )
        return content

    # ---------- Режим 2: думающая модель (Anthropic Messages) ----------

    def _respond_thinking(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        budget = self.settings.thinking_budget
        try:
            message = self._anthropic.messages.create(
                model=self.settings.thinking_model,
                max_tokens=max(MAX_TOKENS, budget + 1024),  # max_tokens > budget_tokens!
                system=SYSTEM_PROMPT,
                messages=self.history,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget,
                },
            )
        except Exception:
            self.history.pop()
            raise

        answer_parts: list[str] = []
        print()  # отступ перед размышлениями
        for block in message.content:
            if block.type == "thinking":
                # reasoning: показываем ход мыслей модели перед ответом
                print("💭 ── РАЗМЫШЛЕНИЯ ──────────────────────────────")
                print(block.thinking)
                print("💭 ───────────────────────────────────────────")
            elif block.type == "text":
                answer_parts.append(block.text)

        usage = message.usage
        reasoning_tokens = getattr(usage, "output_tokens", 0) - sum(
            len(part) for part in answer_parts
        )  # точная разбивка недоступна, показываем общий расход
        print(
            f"   [токены: вход {usage.input_tokens} / выход {usage.output_tokens} "
            f"(включая размышления)]"
        )
        answer = "\n".join(answer_parts).strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer

    # ---------- История ----------

    def show_history(self) -> None:
        print("\n── ИСТОРИЯ РАЗГОВОРА ──────────────────────────")
        for msg in self.history:
            who = "Вы" if msg["role"] == "user" else "Ассистент"
            print(f"{who}: {msg['content']}")
        print("───────────────────────────────────────────────\n")

    def clear_history(self) -> None:
        self.history.clear()
        print("История диалога очищена.")


# ---------- Консольный интерфейс ----------

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   Текстовый помощник через ProxyAPI                      ║
╚══════════════════════════════════════════════════════════╝
Команды: exit — выход · history — история · clear — очистить
"""


def choose_mode(default_thinking: str, default_regular: str) -> str:
    """Выбор режима работы в начале программы. По умолчанию — думающая модель."""
    print("Выберите режим работы:")
    print(f"  1 — Думающая модель  ({default_thinking})  ← по умолчанию")
    print(f"  2 — Обычная модель   ({default_regular})")
    answer = input("Ваш выбор [1]: ").strip()
    if answer == "2":
        print("Режим: обычная модель.\n")
        return "regular"
    print("Режим: думающая модель.\n")
    return "thinking"


def main() -> None:
    load_dotenv()
    print(BANNER)

    try:
        settings = Settings.from_env()
    except RuntimeError as err:
        print(f"❌ {err}")
        sys.exit(1)

    agent = TextAgent(settings=settings)
    agent.mode = choose_mode(settings.thinking_model, settings.regular_model)
    print("Готов к работе! Напишите сообщение (или 'exit' для выхода).\n")

    while True:
        try:
            user_text = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            agent.show_history()
            print("Завершение работы. До встречи!")
            break

        if not user_text:
            continue

        command = user_text.lower()
        if command in ("exit", "выход", "/exit"):
            agent.show_history()
            print("Завершение работы. До встречи!")
            break
        if command == "history":
            agent.show_history()
            continue
        if command == "clear":
            agent.clear_history()
            continue

        try:
            answer = agent.respond(user_text)
        except KeyboardInterrupt:
            print("\n⚠️ Запрос прерван пользователем.")
            continue
        except Exception as err:  # ошибки API, таймауты, сети
            print(f"\n⚠️ Ошибка запроса: {err}")
            print("   Попробуйте ещё раз или наберите 'exit'.\n")
            continue

        print(f"Ассистент: {answer}\n")


if __name__ == "__main__":
    main()
