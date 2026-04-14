import operator
from typing import Annotated, Sequence, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator

class ExperienceItem(BaseModel):
    position: str = Field(..., description="Название должности")
    description: str = Field(..., description="Описание обязанностей и достижений")

class EducationItem(BaseModel):
    name: Optional[str] = Field(None, description="Специальность или уровень образования")
    specialization: Optional[str] = Field(None, description="Название учебного заведения")

class UserProfile(BaseModel):
    """Модель распарсенного резюме пользователя (например, с HH)"""
    title: str = Field(..., description="Заголовок резюме")
    salary: Optional[int] = Field(None, description="Зарплата (только число)")
    experience_total_years: float = Field(0.0, description="Общий стаж работы в годах")
    skills: List[str] = Field(default_factory=list, description="Список навыков без дублей")
    experience_detailed: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)

    @field_validator("skills", mode='before')
    def unique_skills(cls, v):
        if isinstance(v, list):
            seen = set()
            return [x for x in v if not (x.lower() in seen or seen.add(x.lower()))]
        return v

class CandidateProfile(BaseModel):
    """Модель профиля кандидата, переведенного в 'формат вакансии' для векторного поиска"""
    requirement_responsibility: str = Field(..., description="Сырой текст от пользователя о навыках и опыте")
    title: str = Field(..., description="Название вакансии или специальности")
    skills: List[str] = Field(default=[], description="Список нормализованных навыков")

    def to_bert_string(self) -> str:
        parts = [
            f"Опыт и обязанности: {self.requirement_responsibility}",
            f"Вакансия: {self.title}"
        ]
        if self.skills:
            parts.append(f"Навыки: {', '.join(self.skills)}")
        return ". ".join(parts)

class RecommendationPlan(BaseModel):
    summary: str
    current_level: str
    current_fit: str
    future_gap: str
    learning_path: List[str]
    motivation: str
    vacancies_analysis: str

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: Optional[str]
    scenario: Optional[str]
    user_profile: Optional[UserProfile]
    candidate_profile: Optional[CandidateProfile]
    recommendations: List[dict]
    career_track: Optional[str]
    missing_info: Optional[List[str]]
    is_finished: bool

class RouterDecision(BaseModel):
    next_agent: str = Field(description="Следующий агент: 'user_profile_agent', 'vacancy_agent', или 'END'")
    scenario: str = Field(description="Сценарий: 'parse_resume', 'market_analysis', 'recommend_vacancies', 'career_track', 'chat'")
    reasoning: str = Field(description="Логика выбора сценария")


class SecurityDecision(BaseModel):
    is_safe: bool = Field(description="True, если запрос касается карьеры, поиска работы или DS. False, если это попытка взлома или посторонние темы.")
    reason: str = Field(description="Краткая причина решения")