import gradio as gr
import asyncio
import polars as pl
import json
import os
from enum import Enum
from typing import List, Optional

from services.model_api import wrapped_get_completion
from vectorize.vectorize import VacancySearchEngine
from vectorize.schema import ExperienceLevel, CandidateProfile
from services.user_profile import process_user_profile_from_history

# from config import config


API_TOKEN = ""#config.API_TOKEN
MODEL_URL = "https://api.mistral.ai/v1/chat/completions"#config.MODEL_URL
MODEL_NAME = "mistral-medium"#config.MODEL_NAME
MODEL_TEMP = 0.7#config.MODEL_TEMP
MAX_HISTORY = 10#config.MAX_HISTORY




# Инициализация поискового движка
VACANCY_DF = None
SEARCH_ENGINE = None

def init_search_engine():
    """Инициализация поискового движка"""
    global VACANCY_DF, SEARCH_ENGINE
    
    try:
        print("Загрузка данных вакансий...")
        VACANCY_DF = pl.read_parquet("./data_artefacts/vacancy_final.parquet")
        print(f"Загружено {len(VACANCY_DF)} вакансий")
        
        print("Создание поискового движка...")
        SEARCH_ENGINE = VacancySearchEngine("efederici/sentence-bert-base")

        if os.path.exists("./data_artefacts/faiss_index.index"):
            SEARCH_ENGINE.load_index("./data_artefacts/faiss_index.index", VACANCY_DF)
        else:
            SEARCH_ENGINE.fit(VACANCY_DF)
            print("Поисковый движок создан")
            SEARCH_ENGINE.save_index("./data_artefacts/faiss_index.index")
            print("Индекс сохранен")
            
    except Exception as e:
        print(f"Ошибка инициализации поискового движка: {e}")
        raise


def recommend_vacancies(career_goals: str, top_k: int = 10, **kwargs) -> tuple:
    """
    Поиск вакансий с использованием VacancySearchEngine
    
    Args:
        career_goals: описание карьерных целей
        top_k: количество возвращаемых вакансий
        **kwargs: дополнительные параметры (игнорируются для совместимости)
    
    Returns:
        tuple: (рекомендации, расширенные навыки, карьерные пути)
    """
    global SEARCH_ENGINE
    
    if SEARCH_ENGINE is None:
        init_search_engine()
    
    # Создаем профиль кандидата из career_goals
    # В реальном использовании нужно извлекать навыки из истории диалога
    candidate_profile = CandidateProfile(
        requirement_responsibility=career_goals,
        skills=[],  # Здесь можно добавить извлеченные навыки
        experience=ExperienceLevel.NO_EXPERIENCE
    )
    
    try:
        # Выполняем поиск
        results = SEARCH_ENGINE.search(candidate_profile, top_n=top_k)
        
        # Преобразуем результаты в нужный формат
        recommendations = []
        
        for row in results.rows(named=True):
            rec = {
                'title': row.get('title', ''),
                'company': row.get('company', ''),
                'experience': row.get('experience', ''),
                'salary': row.get('salary', ''),
                'skills': row.get('skills', []),
                'requirements': row.get('requirements', ''),
                'similarity_score': float(row.get('similarity_score', 0.0))
            }
            recommendations.append(rec)
        
        # Извлекаем расширенные навыки из найденных вакансий
        expanded_skills = set()
        for rec in recommendations:
            if isinstance(rec.get('skills'), list):
                expanded_skills.update(rec['skills'])
        
        # Карьерные пути можно генерировать на основе найденных вакансий
        career_paths = []
        for rec in recommendations[:5]:  # Берем топ-5 вакансий
            if rec['title'] and rec['company']:
                career_paths.append(f"{rec['title']} в {rec['company']}")
        
        return recommendations, list(expanded_skills)[:15], career_paths[:5]
        
    except Exception as e:
        print(f"Ошибка поиска вакансий: {e}")
        return [], [], []


QUESTION_BLOCKS = {
    'context': [
        "Привет! Расскажи, пожалуйста, какая у тебя сейчас должность и в какой сфере ты работаешь?",
        "Сколько лет у тебя общего опыта работы?",
    ],
    'education': [
        "Какое у тебя образование? Расскажи про вуз, специальность или курсы.",
        "Какие языки программирования, инструменты или технологии ты чаще всего используешь?"
    ],
    'goals': [
        "Кем ты себя видишь через 1–3 года? Какая должность для тебя была бы следующей целью?",
        "Какой формат работы тебе ближе - офис, удалёнка или гибрид?",
        "Какой уровень дохода для тебя комфортный и мотивирующий?",
        "Что для тебя самое важное при выборе новой работы: стабильность, рост, интересные задачи, свобода, что-то ещё?"
    ]
}


# Системный промпт для валидации ответов
VALIDATION_PROMPT = """Ты - помощник карьерного коуча. Твоя задача - проверить, ответил ли пользователь на заданный вопрос.

ВОПРОС: {question}
ОТВЕТ ПОЛЬЗОВАТЕЛЯ: {answer}

Критерии хорошего ответа:
- Краткий и ясный ответ
- Ответ может быть кратким в пару слов, это нормально
- Если ты предлагаешь конкретные варианты ответа, пользователь должен выбрать только среди них
- Длина ответа больше 5 символов

Ответь ТОЛЬКО "Да" или "Нет" без дополнительных пояснений."""


async def validate_answer(question: str, answer: str) -> bool:
    """Проверяет, подходит ли ответ пользователя к заданному вопросу"""
    if not answer or len(answer.strip()) < 3:
        return False
    
    validation_prompt = VALIDATION_PROMPT.format(question=question, answer=answer)
    messages = [{"role": "system", "content": validation_prompt}]
    
    try:
        llm_response = await wrapped_get_completion(
            MODEL_URL, API_TOKEN, messages, MODEL_NAME, 0.3
        )
        
        # Проверяем, содержит ли ответ "Да"
        return "да" in llm_response.lower().strip()[:10]
    
    except Exception as e:
        print(f"Ошибка валидации: {e}")
        # В случае ошибки считаем ответ валидным, чтобы не блокировать пользователя
        return True


def get_current_question(current_block: str, question_index: int) -> str:
    """Возвращает текущий вопрос для блока"""
    questions = QUESTION_BLOCKS.get(current_block, [])
    if question_index < len(questions):
        return questions[question_index]
    return None


def get_next_block_and_question(current_block: str, question_index: int):
    """Определяет следующий блок и вопрос"""
    questions = QUESTION_BLOCKS.get(current_block, [])
    
    # Если есть еще вопросы в текущем блоке
    if question_index + 1 < len(questions):
        return current_block, question_index + 1
    
    # Переход к следующему блоку
    block_order = list(QUESTION_BLOCKS.keys())
    current_block_index = block_order.index(current_block) if current_block in block_order else -1
    
    if current_block_index + 1 < len(block_order):
        next_block = block_order[current_block_index + 1]
        return next_block, 0
    
    # Все блоки пройдены
    return "recommendation", 0


async def chatbot_step(user_input, history, current_block, question_index, waiting_for_answer):
    
    # Если ждем ответ на конкретный вопрос
    if waiting_for_answer:
        current_question = get_current_question(current_block, question_index)
        
        if current_question:
            # Валидируем ответ
            is_valid = await validate_answer(current_question, user_input)
            
            if not is_valid:
                # Ответ не подходит, просим еще раз
                response = f"Пожалуйста, ответь более подробно на вопрос: {current_question}"
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                return history, current_block, question_index, True, response
            
            # Ответ подходит, сохраняем и переходим дальше
            history.append({"role": "user", "content": user_input})
            
            # Определяем следующий блок/вопрос
            next_block, next_question_index = get_next_block_and_question(current_block, question_index)
            
            if next_block == "recommendation":
                print("=" * 60)
                print("ВСЕ ОТВЕТЫ ПОЛЬЗОВАТЕЛЯ:")
                print("=" * 60)
                user_answers = [msg for msg in history if msg["role"] == "user"]
                for i, answer in enumerate(user_answers, 1):
                    print(f"{i}. {answer['content']}")
                print("=" * 60)

                career_goals = f"Сейчас я работаю: {user_answers[0]['content']}, через 1-3 года я бы хотел быть: {user_answers[7]['content']}"

                response = await generate_final_recommendations(history, career_goals)
                history.append({"role": "assistant", "content": response})
                return history, next_block, 0, False, response
            else:
                # Задаем следующий вопрос
                next_question = get_current_question(next_block, next_question_index)
                if next_question:
                    # Добавляем переходную фразу между блоками
                    if next_block != current_block:
                        if next_block == "education":
                            transition = "Отлично! Теперь расскажи про образование и навыки. "
                        elif next_block == "goals":
                            transition = "Понятно! Давай теперь поговорим о твоих карьерных целях. "
                        else:
                            transition = ""
                        response = transition + next_question
                    else:
                        response = next_question
                    
                    history.append({"role": "assistant", "content": response})
                    return history, next_block, next_question_index, True, response
    
    # Если не ждем ответ (начальное состояние или ошибка)
    first_question = get_current_question("context", 0)
    if first_question:
        history.append({"role": "assistant", "content": first_question})
        return history, "context", 0, True, first_question
    
    return history, current_block, question_index, waiting_for_answer, "Произошла ошибка. Попробуйте начать заново."


async def generate_final_recommendations(history, career_goals):
    """
    Генерация финальных рекомендаций
    """
    
    # Собираем профиль из истории    
    user_profile_json, user_profile_text = process_user_profile_from_history(history)
    
    # Расширяем поисковый запрос контекстом из профиля
    enhanced_query = f"{career_goals}\n\nДополнительный контекст:\n{user_profile_text}"
    
    # Получаем рекомендации на основе расширенного профиля
    recommendations, expanded_skills, career_paths = recommend_vacancies(
        career_goals, 
        top_k=10
    )
    
    if not recommendations:
        return "К сожалению, не удалось найти подходящие рекомендации. Попробуйте уточнить ваши карьерные цели."
    
    final_system_prompt = (
        "Ты — опытный карьерный коуч, специализирующийся на технических ролях. "
        "Твоя задача — проанализировать КОНКРЕТНЫЕ найденные вакансии и дать подробные персональные рекомендации.\n\n"
        
        "ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:\n"
        "1. Используй ТОЛЬКО вакансии из списка 'found_positions' — не придумывай новые.\n"
        "2. Упоминай конкретные компании и позиции по названиям.\n"
        "3. Используй навыки из 'skills_to_develop' для плана развития.\n"
        "4. Рассматривай карьерные пути из 'career_paths'.\n"
        "5. Отвечай СТРОГО в формате JSON без лишнего текста.\n\n"
        
        "ФОРМАТ ОТВЕТА:\n"
        "{\n"
        "  \"response\": \"Текстовый анализ и рекомендации\",\n"
        "  \"recommendation\": {\n"
        "    \"nearest_position\": \"Должность в компании\",\n"
        "    \"nearest_position_reason\": \"Причина выбора\",\n"
        "    \"recommended_position\": \"Должность для следующего шага\",\n"
        "    \"recommended_position_reason\": \"Причина выбора\",\n"
        "    \"skills_gap\": \"Навыки для развития\",\n"
        "    \"plan_1_2_years\": \"План развития на 1-2 года\",\n"
        "    \"recommended_courses\": [\"Курс 1\", \"Курс 2\"],\n"
        "    \"current_vacancies\": [\"Вакансия 1\", \"Вакансия 2\"]\n"
        "  }\n"
        "}"
    )

    # Подготовка данных для модели
    payload = {
        "user_profile": user_profile_json,
        "user_goals": career_goals,
        "found_positions": [
            {
                "title": rec["title"],
                "company": rec["company"],
                "experience": rec.get("experience", ""),
                "salary": rec.get("salary", ""),
                "key_skills": rec.get("skills", [])[:5],
                "requirements": rec.get("requirements", ""),
                "relevance_score": rec.get("similarity_score", 0)
            }
            for rec in recommendations[:5]
        ],
        "skills_to_develop": expanded_skills[:10],
        "career_paths": career_paths[:3],
    }

    user_message = f"""
АНАЛИЗИРУЙ СЛЕДУЮЩИЕ ДАННЫЕ И ДАЙ РЕКОМЕНДАЦИИ:

=== МОЙ ПРОФИЛЬ ===
{user_profile_json}

=== МОИ КАРЬЕРНЫЕ ЦЕЛИ ===
{career_goals}

=== НАЙДЕННЫЕ ДЛЯ МЕНЯ ВАКАНСИИ ===
{json.dumps(payload["found_positions"], ensure_ascii=False, indent=2)}

=== НАВЫКИ ДЛЯ РАЗВИТИЯ ===
{json.dumps(expanded_skills[:10], ensure_ascii=False)}

=== ВОЗМОЖНЫЕ КАРЬЕРНЫЕ ПУТИ ===
{json.dumps(career_paths[:3], ensure_ascii=False)}
"""

    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        llm_response = await wrapped_get_completion(
            MODEL_URL, API_TOKEN, messages, MODEL_NAME, MODEL_TEMP
        )
        
        print(f"[LLM response]: {llm_response[:500]}...")
        
        # Парсинг JSON ответа
        try:
            result = json.loads(llm_response)
        except json.JSONDecodeError:
            # Если не получается, ищем JSON внутри текста
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    result = {"response": llm_response}
            else:
                result = {"response": llm_response}
        
        # Форматирование ответа
        return parse_llm_response(result)
        
    except Exception as e:
        print(f"[ERROR] Ошибка при генерации рекомендаций: {e}")
        return f"Произошла ошибка при генерации рекомендаций: {e}"


def parse_llm_response(data: dict) -> str:
    """Форматирует ответ LLM в читаемый текст"""
    
    # Если это уже строка
    if isinstance(data, str):
        return data
    
    # Если это словарь с ожидаемой структурой
    if isinstance(data, dict):
        response = data.get("response", "")
        rec = data.get("recommendation", {})
        
        text_parts = ["🔎 Рекомендации по карьерным шагам:\n"]
        
        if response:
            text_parts.append(f"{response}\n")
        
        if rec.get("nearest_position"):
            text_parts.append(f"📍 **Ближайшая позиция:** {rec['nearest_position']}")
            if rec.get("nearest_position_reason"):
                text_parts.append(f"Причина: {rec['nearest_position_reason']}\n")
        
        if rec.get("recommended_position"):
            text_parts.append(f"⭐ **Рекомендуемая следующая позиция:** {rec['recommended_position']}")
            if rec.get("recommended_position_reason"):
                text_parts.append(f"Причина: {rec['recommended_position_reason']}\n")
        
        if rec.get("skills_gap"):
            text_parts.append(f"🛠 **Навыки для развития:** {rec['skills_gap']}\n")
        
        if rec.get("plan_1_2_years"):
            text_parts.append(f"📅 **План развития на 1–2 года:**\n{rec['plan_1_2_years']}\n")
        
        if rec.get("recommended_courses"):
            courses = "\n".join([f"   • {c}" for c in rec['recommended_courses'][:5]])
            text_parts.append(f"📚 **Рекомендованные курсы:**\n{courses}\n")
        
        if rec.get("current_vacancies"):
            vacancies = "\n".join([f"   • {v}" for v in rec['current_vacancies'][:5]])
            text_parts.append(f"💼 **Актуальные вакансии:**\n{vacancies}")
        
        return "\n".join(text_parts)
    
    return str(data)


def sync_chatbot(user_input, history, current_block, question_index, waiting_for_answer):
    """Синхронная обертка для асинхронной функции"""
    history, current_block, question_index, waiting_for_answer, response = asyncio.run(
        chatbot_step(user_input, history, current_block, question_index, waiting_for_answer)
    )
    return history, history, current_block, question_index, waiting_for_answer, ""


def reset_chat():
    """Сброс чата к начальному состоянию"""
    first_question = get_current_question("context", 0)
    initial_history = [{"role": "assistant", "content": first_question}]
    return initial_history, initial_history, "context", 0, True, ""


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Career Coach")
    gr.Markdown("Отвечай на вопросы подробно, чтобы получить персональные карьерные рекомендации!")

    chatbot_ui = gr.Chatbot(
        value=[{"role": "assistant", "content": get_current_question("context", 0)}],
    )

    msg = gr.Textbox(label="Ваш ответ:", placeholder="Введите ваш ответ здесь...")
    reset_btn = gr.Button("🔄 Начать заново")

    # Состояния
    history_state = gr.State(value=[{"role": "assistant", "content": get_current_question("context", 0)}])
    block_state = gr.State(value="context")
    question_index_state = gr.State(value=0)
    waiting_for_answer_state = gr.State(value=True)
    
    # Отправка сообщения
    msg.submit(
        sync_chatbot, 
        [msg, history_state, block_state, question_index_state, waiting_for_answer_state], 
        [chatbot_ui, history_state, block_state, question_index_state, waiting_for_answer_state, msg]
    )

    # Кнопка сброса
    reset_btn.click(
        reset_chat, 
        [], 
        [chatbot_ui, history_state, block_state, question_index_state, waiting_for_answer_state, msg]
    )


# Инициализация поискового движка при запуске
try:
    init_search_engine()
except Exception as e:
    print(f"Не удалось инициализировать поисковый движок: {e}")
    print("Поиск вакансий будет недоступен")

demo.launch()
