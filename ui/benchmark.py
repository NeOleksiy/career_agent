import asyncio
import json
import re
from typing import List, Dict, Any, Tuple

# Импортируем функции из вашей системы
from .app_gradio import (
    init_search_engine,
    recommend_vacancies,
    generate_final_recommendations,
    wrapped_get_completion,
    MODEL_URL,
    API_TOKEN,
    MODEL_NAME
)

# --- 6 ПРОТОТИПОВ ЛЮДЕЙ ---
PROTOTYPES = [
    {
        "id": 1,
        "description": "Студент без опыта хочет стать ComputerVision инженером",
        "current_role": "Студент 3 курса технического вуза",
        "experience": "Нет коммерческого опыта",
        "skills": "python, opencv , линейная алгебра",
        "career_goal": "Стать Computer Vision инженером"
    },
    {
        "id": 2,
        "description": "Студент с опытом полгода в аналитике, хочет стать middle data analyst",
        "current_role": "Junior Data Analyst",
        "experience": "1 год в аналитике данных",
        "skills": "sql, python, excel, tableau",
        "career_goal": "Middle Data Analyst с зарплатой от 150к"
    },
    {
        "id": 3,
        "description": "Бизнес-аналитик с опытом 4 года, хочет стать NLP/LLM инженером",
        "current_role": "Senior Business Analyst",
        "experience": "4 года в бизнес-аналитике",
        "skills": "sql, анализ требований, дашборды, python",
        "career_goal": "NLP/LLM инженер в продуктовой компании"
    },
    {
        "id": 4,
        "description": "Middle ML инженер хочет стать Team Leadом в крупной компании",
        "current_role": "Middle ML Engineer",
        "experience": "6 года в ML-инженерии",
        "skills": "python, pytorch, mlops, docker, kubernetes, ml",
        "career_goal": "Team Lead или Head"
    },
    {
        "id": 5,
        "description": "Закончил вуз, работал 1 год на кафедре аналитиком, хочет стать AI Product Manager и зарабатывать 200к рублей",
        "current_role": "Аналитик на кафедре университета",
        "experience": "1 год ",
        "skills": "python, статистика, линейная алгебра, ml, продуктовая разработка, ai, исследование, r&d",
        "career_goal": "AI product с зарплатой 150 000+ рублей"
    }
]

# --- ПРОМПТЫ ДЛЯ LLM-ОЦЕНКИ  ---
SEARCH_EVAL_PROMPT = """Ты оцениваешь качество поиска вакансий в RAG-системе для карьерных рекомендаций.

ИНФОРМАЦИЯ О КАНДИДАТЕ:
Текущая роль: {current_role}
Опыт: {experience}
Навыки: {skills}
Цель карьеры: {career_goal}

ПОИСКОВЫЙ ЗАПРОС: {query}

НАЙДЕННЫЕ ВАКАНСИИ (первые {num_vacancies} из {total_vacancies}):
{vacancies_text}

ИНСТРУКЦИЯ:
1. Оцени, насколько найденные вакансии релевантны цели кандидата
2. Учти его текущий опыт, навыки и карьерную цель
3. Верни ответ ТОЛЬКО в формате JSON:
{{
    "score": число от 1 до 5,
    "reason": "краткое обоснование на русском"
}}

Критерии оценки:
1 - Вакансии вообще не релевантны цели
2 - Большинство вакансий не релевантны
3 - Вакансии лишь немного соответствуют цели кандидата
4 - Вакансии более-менее релевантны цели и уровню кандидата
5 - Все вакансии идеально соответствуют цели и уровню кандидата
"""

RESPONSE_EVAL_PROMPT = """Ты оцениваешь качество карьерных рекомендаций от RAG-системы.

ИНФОРМАЦИЯ О КАНДИДАТЕ:
Текущая роль: {current_role}
Опыт: {experience}
Навыки: {skills}

ЦЕЛЬ КАНДИДАТА: {goal}

ОТВЕТ СИСТЕМЫ:
{response}

ИНСТРУКЦИЯ:
1. Оцени, насколько ответ полезен для достижения цели кандидата
2. Учти его текущий опыт и навыки
3. Верни ответ ТОЛЬКО в формате JSON:
{{
    "score": число от 1 до 5,
    "reason": "краткое обоснование на русском"
}}

Критерии оценки:
1 - Ответ бесполезен, не учитывает цель и опыт
2 - Ответ поверхностный, мало полезной информации
3 - Ответ содержит общие рекомендации, но без конкретики
4 - Понятные рекомендации согласно цели кандидата
5 - Идеальные рекомендации, полностью персонализированные
"""

async def evaluate_with_llm(prompt: str, temperature: float = 0.3) -> Tuple[int, str]:
    messages = [{"role": "user", "content": prompt}]
    
    try:
        llm_response = await wrapped_get_completion(
            MODEL_URL, API_TOKEN, messages, MODEL_NAME, temperature
        )

        llm_response = llm_response.strip()
        
        import re
        
        # Ищем JSON-объект
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                score = result.get("score", 3)
                reason = result.get("reason", "Нет обоснования")
                
                # Обеспечиваем корректный диапазон
                score = int(score) if isinstance(score, (int, float)) else 3
                score = max(1, min(5, score))
                
                return score, reason
            except json.JSONDecodeError:
                pass  # Пробуем другие методы
        
        score = 3
        
        # Способ 1: Ищем "score": X
        score_match = re.search(r'"score"\s*:\s*(\d+)', llm_response)
        if not score_match:
            # Способ 2: Ищем "score": X.X
            score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', llm_response)
        
        if score_match:
            try:
                score = float(score_match.group(1))
                score = int(score) if score.is_integer() else round(score)
            except:
                pass
        
        if score == 3:
            numbers = re.findall(r'\b[1-5]\b', llm_response)
            if numbers:
                try:
                    score = int(numbers[0])
                except:
                    pass
        
        reason = "Оценка выполнена автоматически"
        
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', llm_response)
        if not reason_match:
            reason_match = re.search(r"'reason'\s*:\s*'([^']*)'", llm_response)
        
        if reason_match:
            reason = reason_match.group(1)
        else:
            # Пробуем найти обоснование после "reason:"
            reason_match = re.search(r'reason\s*:\s*(.+?)(?:\n|$)', llm_response, re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # Берем всё после первого перевода строки
                lines = llm_response.split('\n')
                if len(lines) > 1:
                    reason = lines[1].strip()
        
        # Обеспечиваем корректный диапазон
        score = max(1, min(5, score))
        
        return score, reason[:200]  # Ограничиваем длину обоснования
        
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return 3, f"Ошибка при оценке: {str(e)[:100]}"

async def evaluate_search_quality(profile: Dict, query: str, vacancies: List[Dict]) -> Tuple[int, str]:
    
    if not vacancies:
        return 1, "Нет найденных вакансий для оценки"
    
    max_vacancies_to_show = min(3, len(vacancies))
    vacancies_text = ""
    
    for i, vac in enumerate(vacancies[:max_vacancies_to_show], 1):
        title = vac.get('title', 'Нет названия')
        company = vac.get('company', 'Неизвестная компания')
        experience = vac.get('experience', 'Не указано')
        skills = vac.get('skills', [])
        
        vacancies_text += f"\n{i}. {title} в {company}\n"
        vacancies_text += f"   Опыт: {experience}\n"
        if skills:
            vacancies_text += f"   Навыки: {', '.join(skills[:3])}\n"
    
    prompt = SEARCH_EVAL_PROMPT.format(
        current_role=profile['current_role'],
        experience=profile['experience'],
        skills=profile['skills'],
        career_goal=profile['career_goal'],
        query=query,
        num_vacancies=max_vacancies_to_show,
        total_vacancies=len(vacancies),
        vacancies_text=vacancies_text
    )
    
    return await evaluate_with_llm(prompt)

async def evaluate_response_quality(profile: Dict, goal: str, response: str) -> Tuple[int, str]:
    
    if not response:
        return 1, "Пустой ответ"
    
    response_preview = response[:800] + "..." if len(response) > 800 else response
    
    prompt = RESPONSE_EVAL_PROMPT.format(
        current_role=profile['current_role'],
        experience=profile['experience'],
        skills=profile['skills'],
        goal=goal,
        response=response_preview
    )
    
    return await evaluate_with_llm(prompt)

# --- СОЗДАНИЕ ИСТОРИИ ДИАЛОГА ---
def create_dialog_for_prototype(prototype: Dict) -> List[Dict]:
    """Создает историю диалога для прототипа"""
    return [
        {"role": "user", "content": f"Я {prototype['current_role']}"},
        {"role": "assistant", "content": "Сколько у вас опыта работы?"},
        {"role": "user", "content": prototype['experience']},
        {"role": "assistant", "content": "Какие у вас основные навыки?"},
        {"role": "user", "content": prototype['skills']},
        {"role": "assistant", "content": "Кем вы хотите стать через 1-2 года?"},
        {"role": "user", "content": prototype['career_goal']},
        {"role": "assistant", "content": "Какой формат работы предпочитаете?"},
        {"role": "user", "content": "Готов к разным форматам, но предпочитаю гибрид"},
        {"role": "assistant", "content": "Какие у вас зарплатные ожидания?"},
        {"role": "user", "content": "Хочу достойную оплату по рынку и устойчивый карьерный рост"}
    ]

def generate_search_query(prototype: Dict) -> str:
    """Генерирует поисковый запрос на основе профиля"""
    
    queries = {
        1: "junior computer vision engineer OpenCV Python обучение стажировка",
        2: "middle data analyst SQL Python аналитика данных",
        3: "NLP engineer LLM разработчик языковые модели BERT",
        4: "team lead ml engineer руководитель команды machine learning",
        5: "ML инженер junior machine learning 200000 рублей",
    }
    
    return queries.get(prototype['id'], prototype['career_goal'])

async def run_benchmark():
    
    print("🚀 Запуск LLM-бенчмарка\n")
    
    try:
        init_search_engine()
        print("✅ Поисковый движок готов\n")
    except Exception as e:
        print("Продолжаем без инициализации движка...\n")
    
    results = []
    
    for proto in PROTOTYPES:
        print(f"🧪 ПРОТОТИП {proto['id']}: {proto['description']}")
        
        print("\n1. 🔍 Поиск вакансий...")
        search_query = generate_search_query(proto)
        
        search_score, search_reason = 1, "Не удалось выполнить поиск"
        vacancies = []
        
        try:
            vacancies, _, _ = recommend_vacancies(search_query, top_k=4)
            print(f"   Найдено вакансий: {len(vacancies)}")
            
            if vacancies:
                print("   🤔 LLM оценивает релевантность...")
                search_score, search_reason = await evaluate_search_quality(
                    proto, search_query, vacancies
                )
                print(f"   ✅ Оценка LLM: {search_score}/5")
                print(f"   📝 Обоснование: {search_reason}")
                
                if vacancies:
                    print(f"   📋 Примеры вакансий:")
                    for i, vac in enumerate(vacancies[:2], 1):
                        title = vac.get('title', 'Без названия')
                        company = vac.get('company', 'Неизвестно')
                        print(f"     {i}. {title} в {company}")
            else:
                print("   ❌ Не найдено подходящих вакансий")
                search_reason = "Нет результатов поиска"
                
        except Exception as e:
            print(f"   ❌ Ошибка при поиске: {e}")
            search_reason = f"Ошибка поиска: {str(e)[:100]}"
        
        print("\n2. 📝 Генерация рекомендаций...")
        
        response_score, response_reason = 1, "Не удалось сгенерировать ответ"
        response_text = ""
        
        try:
            dialog = create_dialog_for_prototype(proto)
            career_goals = proto['career_goal']
            
            print("   🤖 Генерация ответа системой...")
            response_text = await generate_final_recommendations(dialog, career_goals)
            
            if response_text and len(response_text) > 10:
                print(f"   ✅ Ответ сгенерирован ({len(response_text)} символов)")
                
                print("   🤔 LLM оценивает качество ответа...")
                response_score, response_reason = await evaluate_response_quality(
                    proto, career_goals, response_text
                )
                print(f"   ✅ Оценка LLM: {response_score}/5")
                print(f"   📝 Обоснование: {response_reason}")
                
                # Показываем превью ответа
                preview = response_text[:150] + "..." if len(response_text) > 150 else response_text
                print(f"   📄 Превью ответа: {preview}")
            else:
                print("   ❌ Ответ слишком короткий или пустой")
                response_reason = "Не удалось получить содержательный ответ"
                
        except Exception as e:
            print(f"   ❌ Ошибка при генерации: {e}")
            response_reason = f"Ошибка генерации: {str(e)[:100]}"
        
        # Сохраняем результат
        results.append({
            "prototype_id": proto["id"],
            "description": proto["description"],
            "search_query": search_query,
            "vacancies_found": vacancies,
            "search_score": search_score,
            "search_reason": search_reason,
            "response_score": response_score,
            "response_reason": response_reason,
            "response_preview": response_text[:300] + "..." if response_text else ""
        })
        
        print(f"\n   📊 ИТОГ ПО ПРОТОТИПУ:")
        print(f"   • Поиск вакансий: {search_score}/5")
        print(f"   • Качество ответа: {response_score}/5")
        print(f"   • Средняя оценка: {(search_score + response_score) / 2:.1f}/5")
    
    print("ИТОГОВАЯ ОЦЕНКА СИСТЕМЫ")
    
    if results:
        # Рассчитываем средние оценки
        search_scores = [r["search_score"] for r in results]
        response_scores = [r["response_score"] for r in results]
        
        avg_search = sum(search_scores) / len(search_scores)
        avg_response = sum(response_scores) / len(response_scores)
        
        # Итоговая оценка (60% поиск, 40% ответ)
        final_score = (avg_search * 0.6) + (avg_response * 0.4)
        
        print(f"\n📊 СРЕДНИЕ ОЦЕНКИ ПО 6 ПРОТОТИПАМ:")
        print(f"   • Релевантность поиска: {avg_search:.2f}/5")
        print(f"   • Качество рекомендаций: {avg_response:.2f}/5")
        print(f"   • ИТОГОВАЯ ОЦЕНКА СИСТЕМЫ: {final_score:.2f}/5")
        
        
        
        # Сохраняем результаты в файл
        output = {
            "benchmark_date": asyncio.get_event_loop().time(),
            "final_score": round(final_score, 2),
            "avg_search_score": round(avg_search, 2),
            "avg_response_score": round(avg_response, 2),
            "total_prototypes": len(results),
            "results": results
        }
        
        with open("rag_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в 'rag_benchmark_results.json'")



if __name__ == "__main__":
    asyncio.run(run_benchmark())