"""
Gradio-интерфейс для Enterprise RAG чат-бота.

Запуск:
    python -m src.app
Откроется на http://127.0.0.1:7860
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gradio as gr
from src.rag_chain import RagBot
from src.config import OLLAMA_LLM_MODEL, TOP_K

bot = RagBot(model_name=OLLAMA_LLM_MODEL, top_k=TOP_K)


def respond(message, history, top_k):
    """Возвращает ответ бота с метаданными."""
    if int(top_k) != bot.top_k:
        bot.set_top_k(int(top_k))
    result = bot.ask(message)
    answer = result["answer"]
    meta = (
        f"\n\n---\n📎 Источники: {', '.join(result['sources']) or 'нет'}"
        f" | ⏱ {result['latency_sec']} сек | 🔎 чанков: {result['num_chunks']}"
    )
    return answer + meta


# CSS для мелкого текста и компактного отображения
custom_css = """
.small-text label {
    font-size: 0.8em !important;
}
.small-text input, .small-text .wrap {
    font-size: 0.8em !important;
}
"""

with gr.Blocks(title="Enterprise Private GPT (RAG)", css=custom_css) as demo:
    # Заголовок (можно сделать компактным, но оставим для ясности)
    gr.Markdown("# 🔒 Enterprise Private GPT — локальный RAG-чат-бот")
    gr.Markdown(
        f"Модель: **{OLLAMA_LLM_MODEL}** (Ollama, локально, без доступа в интернет). "
        "Отвечает строго на основе загруженных документов."
    )

    # Основной чат (занимает всё доступное пространство)
    chatbot = gr.Chatbot(label="Диалог", height=500)

    # Строка ввода: текстовое поле + кнопка отправки
    with gr.Row():
        msg = gr.Textbox(
            label="Сообщение",
            placeholder="Введите ваш вопрос...",
            lines=2,
            scale=4,
        )
        send_btn = gr.Button("Отправить", variant="primary", scale=1)

    # Настройки под полем ввода — мелким шрифтом
    with gr.Row():
        top_k_slider = gr.Slider(
            1,
            10,
            value=TOP_K,
            step=1,
            label="Top‑K (число фрагментов из базы)",
            elem_classes="small-text",
        )
        # Можно добавить дополнительную информацию, если нужно

    # Примеры для быстрого старта
    gr.Examples(
        examples=[
            ["Какой порядок действий при инциденте безопасности?"],
            ["Расскажи рецепт пирога"],
        ],
        inputs=[msg],
        label="Примеры запросов",
    )

    # Состояние истории (для ручного управления)
    history_state = gr.State([])

    def send_message(message, history, top_k):
        """Обработчик отправки сообщения."""
        if not message or not message.strip():
            return "", history, history

        # Добавляем вопрос пользователя в историю
        history = history + [[message, None]]

        # Получаем ответ от бота (используем существующую функцию)
        answer = respond(message, history, top_k)  # history передаётся, но не используется

        # Обновляем последний элемент истории ответом
        history[-1][1] = answer

        # Очищаем поле ввода, обновляем историю и чат
        return "", history, history

    # Привязываем события: отправка по Enter и по кнопке
    msg.submit(
        send_message,
        [msg, history_state, top_k_slider],
        [msg, history_state, chatbot],
    )
    send_btn.click(
        send_message,
        [msg, history_state, top_k_slider],
        [msg, history_state, chatbot],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)