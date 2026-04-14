import json
import os
from logger import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState, CandidateProfile, UserProfile
from .prompts import CANDIDATE_PROFILE_SYSTEM_PROMPT, RESUME_ANALYSIS_PROMPT
from tools.user_profile.profile_parser import parse_hh_resume
from dotenv import load_dotenv
from langgraph.graph import END

load_dotenv()

llm = ChatOpenAI(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url='https://api.deepseek.com',
    max_tokens=8092,
    temperature=0.3
)

async def format_to_candidate_profile(user_profile: UserProfile) -> CandidateProfile:
    try:
        parser = PydanticOutputParser(pydantic_object=CandidateProfile)
        format_instructions = parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n{format_instructions}"),
            ("human", "Преобразуй это резюме в профиль кандидата:\n{user_data}")
        ])
        
        chain = prompt | llm | parser
        user_data_str = user_profile.model_dump_json(indent=2)

        return await chain.ainvoke({
            "system_prompt": CANDIDATE_PROFILE_SYSTEM_PROMPT,
            "format_instructions": format_instructions,
            "user_data": user_data_str
        })
    except Exception as e:
        logger.error(f"Ошибка трансформации профиля: {e}")
        return CandidateProfile(
            title=user_profile.title,
            requirement_responsibility="Данные не распарсились корректно",
            skills=user_profile.skills
        )

async def user_profile_node(state: AgentState) -> dict:
    logger.info("[User Profile Agent] Работа с данными пользователя...")
    
    last_msg = state["messages"][-1].content
    user_profile = state.get("user_profile")
    candidate_profile = state.get("candidate_profile")
    analysis_report = None

    if "hh.ru/resume" in last_msg:
        try:
            url = [word for word in last_msg.split() if "hh.ru" in word][0]
            user_profile = await parse_hh_resume(url)
            
            if not user_profile or not user_profile.title:
                return {
                    "messages": [AIMessage(content="К сожалению, я не смог получить данные по этой ссылке. Пожалуйста, проверьте, что профиль открыт, или скопируйте текст резюме сюда.")],
                    "next_agent": END 
                }
                
            candidate_profile = await format_to_candidate_profile(user_profile)
            
        except Exception as e:
            logger.error(f"Критическая ошибка парсера: {e}")
            return {
                "messages": [AIMessage(content="Произошла техническая ошибка при чтении ссылки. Попробуйте прислать текст вашего опыта вручную.")],
                "next_agent": END
            }

    if user_profile and not user_profile.skills:
        logger.debug("[User Profile Agent] Не обнаружил в резюме skills")
        return {
            "user_profile": user_profile,
            "messages": [AIMessage(content="Я загрузил ваше резюме, но не нашел в нем списка ключевых навыков. Какие технологии (например, Python, PyTorch, SQL) вы используете?")],
            "next_agent": END
        }

    if state["scenario"] == "parse_resume" and user_profile:
        logger.info("[User Profile Agent] Генерирую аудит резюме...")
        try:
            analysis_response = await llm.ainvoke([
                SystemMessage(content=RESUME_ANALYSIS_PROMPT),
                HumanMessage(content=f"Проанализируй этот профиль: {user_profile.model_dump_json()}")
            ])
            analysis_report = analysis_response.content
        except Exception as e:
            analysis_report = "Не удалось провести детальный анализ, но ваш профиль сохранен."

    next_step = "main_agent"
    if state["scenario"] in ["recommend_vacancies", "career_track"] and candidate_profile:
        next_step = "vacancy_agent"

    return {
        "user_profile": user_profile,
        "candidate_profile": candidate_profile,
        "messages": [AIMessage(content=analysis_report)] if analysis_report else [],
        "next_agent": next_step
    }