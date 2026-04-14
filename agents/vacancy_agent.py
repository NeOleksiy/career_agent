import polars as pl
import difflib
import os
from logger import logger
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from definitions import DB_PATH, INDEX_URL
from .state import AgentState, RecommendationPlan
from .prompts import RECOMMENDATION_SYSTEM_PROMPT
from tools.vectorize.vectorize import Recommender
from langgraph.graph import END

load_dotenv()

llm = ChatOpenAI(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url='https://api.deepseek.com',
    max_tokens=8092,
    temperature=0.3
)

def read_vacancies_by_title(target_title: str, parquet_path: str = "./data/vacancies.parquet") -> pl.DataFrame:
    logger.info(f"[Vacancy Agent] Вливание знаний о рынке для позиции: {target_title}...")
    try:
        if not os.path.exists(parquet_path):
            logger.error(f"Файл не найден: {parquet_path}")
            return None
            
        df = pl.read_parquet(parquet_path)
        required_cols = ['vacancy_id', 'title', 'loc', 'requirement', 'responsibility', 'skills', 'company', 'salary_from', 'salary_to']
        cols = [c for c in required_cols if c in df.columns]
        df = df.select(cols)
        
        unique_titles = df["title"].unique().to_list()
        close_matches = difflib.get_close_matches(target_title, unique_titles, n=15, cutoff=0.4)
        
        if close_matches:
            return df.filter(pl.col("title").is_in(close_matches))
        else:
            return df.filter(pl.col("title").str.to_lowercase().str.contains(target_title.lower().split()[0]))
    except Exception as e:
        logger.error(f"[Vacancy Agent] Ошибка при чтении вакансий: {e}")
        return None


async def generate_final_recommendations(profile, current_recs, market_response: str) -> str:
    logger.info(f"[Vacancy Agent] Начинаем генерацию карьерного трека")
    try:
        parser = PydanticOutputParser(pydantic_object=RecommendationPlan)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n{format_instructions}"),
            ("human", "Профиль: {profile_data}\nТекущие вакансии: {vacancies}\nСостояние рынка: {market_info}")
        ])
        
        current_text = "\n".join([f"- {r['title']} в {r['company']}" for r in current_recs]) if current_recs else "Подходящих вакансий в базе не найдено."
        profile_data = profile if profile else 'Данные отсутствуют'
        
        chain = prompt_template | llm | parser
        
        plan = await chain.ainvoke({
            "system_prompt": RECOMMENDATION_SYSTEM_PROMPT,
            "format_instructions": parser.get_format_instructions(),
            "profile_data": profile_data,
            "vacancies": current_text,
            "market_info": market_response
        })

        # Формирование итогового текста (plan уже является объектом RecommendationPlan)
        output = f"## 🎯 Карьерный план развития\n\n"
        output += f"### 📊 Анализ текущего уровня\n{plan.current_level}\n\n"
        output += f"### 🚀 Технологический разрыв (Gap Analysis)\n{plan.future_gap}\n\n"
        output += "### 📚 Пошаговый план\n"
        for i, step in enumerate(plan.learning_path, 1):
            output += f"{i}. {step}\n"
        output += "### Рекомендованные вакансии\n"
        output += current_text
        
        return output

    except Exception as e:
        logger.error(f"[Vacancy Agent] Ошибка генерации трека: {e}")
        return "Извините, не удалось сформировать детальный план. Попробуйте уточнить ваш запрос."


async def vacancy_node(state: AgentState) -> dict:
    scenario = state.get("scenario")
    final_messages = []
    user_prof = state.get("user_profile")
    cand_prof = state.get("candidate_profile")
    
    if scenario in ["recommend_vacancies", "career_track"] and not cand_prof:
        logger.debug(f"[Vacancy Agent] Отсутствует обработка резюме пользователя агентом user profile")
        return {
            "messages": [AIMessage(content="Для подбора вакансий мне нужно сначала обработать ваше резюме. Пожалуйста, убедитесь, что вы предоставили ссылку или текст.")],
            "next_agent": "main_agent"
        }

    target_title = user_prof.title if user_prof else "Data Scientist"
    logger.info(f"[Vacancy Agent] Читаем вакансии по title специализации пользователя")
    market_df = read_vacancies_by_title(
        target_title, 
        parquet_path=DB_PATH
    )
    
    recommendations = []
    if scenario in ["recommend_vacancies", "career_track"]:
        logger.info(f"[Vacancy Agent] Делаем рекомендации для сценариев recommend_vacancies и career_track")
        try:
            recommender = Recommender(database_url=DB_PATH, index_url=INDEX_URL)
            recommender.init_search_engine()
            recommendations, _, _ = await recommender.recommend_vacancies(cand_prof.to_bert_string())
        except Exception as e:
            logger.error(f"Ошибка движка рекомендаций: {e}")

    market_response = None
    if scenario == "market_analysis" or "career_track":
        logger.info(f"[Vacancy Agent] Делаем анализ рынка")
        if market_df is not None and len(market_df) > 0:
            market_data_str = str(market_df.head(10).to_dicts())
            res = await llm.ainvoke([
                SystemMessage(content="Ты аналитик рынка труда. Ответь на вопрос пользователя, основываясь на данных вакансий."),
                HumanMessage(content=f"Данные: {market_data_str}\n\nВопрос: {state['messages'][-1].content}")
            ])
            market_response = res.content

        else:
            logger.debug(f"[Vacancy Agent] Не найдено вакансий по специальности, анализ невозможен")
            market_response = "К сожалению, в моей базе данных сейчас нет детальной информации по этой узкой специальности, но я могу проконсультировать вас на основе общих трендов в Data Science."
        

    career_track = ""
    if scenario == "career_track":
        career_track = await generate_final_recommendations(user_prof, recommendations, market_response)
        final_messages.append(AIMessage(content=career_track))
    
    if scenario == "market_analysis" and market_response:
        final_messages.append(AIMessage(content=market_response))
    
    if scenario == "recommend_vacancies":
        if recommendations:
            res_text = "## 🔍 Рекомендованные вакансии:\n\n"
            for i, r in enumerate(recommendations[:3], 1):
                res_text += f"{i}. **{r['title']}** в {r['company']}\n"
            final_messages.append(AIMessage(content=res_text))
        else:
            logger.debug(f"[Vacancy Agent] подходящих вакансий не найдено")
            final_messages.append(AIMessage(content="К сожалению, подходящих вакансий не найдено."))



    return {
        "recommendations": recommendations,
        "career_track": career_track,
        "messages": final_messages,
        "next_agent": "main_agent"
    }