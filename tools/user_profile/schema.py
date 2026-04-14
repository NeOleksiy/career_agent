from typing import List, Dict, Any, Union, Optional

from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ExperienceItem(BaseModel):
    position: str = Field(..., description="Название должности")
    description: str = Field(..., description="Описание обязанностей и достижений")

class EducationItem(BaseModel):
    name: Optional[str] = Field(None, description="Специальность или уровень образования")
    specialization: Optional[str] = Field(None, description="Название учебного заведения")

class UserProfile(BaseModel):
    title: str = Field(..., description="Заголовок резюме")
    salary: Optional[int] = Field(None, description="Зарплата (только число)")
    experience_total_years: float = Field(0.0, description="Общий стаж работы в годах")
    skills: List[str] = Field(default_factory=list, description="Список навыков без дублей")
    experience_detailed: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)

    @field_validator("skills")
    def unique_skills(cls, v):
        """Дополнительная проверка уникальности навыков на уровне модели"""
        if isinstance(v, list):
            seen = set()
            return [x for x in v if not (x.lower() in seen or seen.add(x.lower()))]
        return v

# Определение модели CandidateProfile (как в предыдущем примере)
class ExperienceLevel(str, Enum):
    NO_EXPERIENCE = "нет опыта"
    ONE_TO_THREE = "от 1 до 3 лет"
    THREE_TO_SIX = "от 3 до 6 лет"
    MORE_THAN_SIX = "более 6 лет"


class CandidateProfile(BaseModel):
    """Модель для представления профиля кандидата."""
    
    requirement_responsibility: str = Field(
        ...,
        description="Сырой текст от пользователя о своих навыках и опыте"
    )
    title: str = Field(
        ...,
        description="Название вакансии или специальности"
    )
    
    skills: List[str] = Field(
        default=[],
        description="Список нормализованных навыков"
    )

    
    def to_bert_string(self) -> str:
        """Преобразует профиль в строку для векторизации."""
        parts = []
        parts.append(f"Опыт и обязанности: {self.requirement_responsibility}")
        parts.append(f"Вакансия: {self.title}")
        
        if self.skills:
            skills_text = ", ".join(self.skills)
            parts.append(f"Навыки: {skills_text}")
        
        # parts.append(f"Опыт: {self.experience.value}")
        
        return ". ".join(parts)
