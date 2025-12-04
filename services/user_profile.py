import re
from typing import List, Dict, Any
from vectorize.schema import CandidateProfile, ExperienceLevel


def extract_user_data_from_history(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Упрощённое извлечение навыков, опыта и текстового описания из истории диалога."""

    user_messages = [m["content"] for m in history if m["role"] == "user"]
    full_text = " ".join(user_messages)

    # --- Skills ---
    skill_patterns = [
        r"(?:знаю|владею|использую|работаю с)\s+([^.!?\n]+)",
        r"(?:skills?|навыки)\s*[:\-]\s*([^.!?\n]+)"
    ]

    skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, full_text, flags=re.I)
        for block in matches:
            parts = [s.strip() for s in re.split(r"[;,]", block) if 1 < len(s.strip()) < 40]
            skills.extend(parts)

    skills = list(set(skills))  # Уникальные

    # --- Experience level ---
    exp_enum = ExperienceLevel.NO_EXPERIENCE
    match = re.search(r"(\d+)\s*(год|года|лет)", full_text)
    if match:
        years = int(match.group(1))
        if years <= 0:
            exp_enum = ExperienceLevel.NO_EXPERIENCE
        elif years <= 3:
            exp_enum = ExperienceLevel.ONE_TO_THREE
        elif years <= 6:
            exp_enum = ExperienceLevel.THREE_TO_SIX
        else:
            exp_enum = ExperienceLevel.MORE_THAN_SIX

    # --- Responsibility text ---
    requirement_text = full_text.strip()

    return {
        "skills": skills,
        "experience_enum": exp_enum,
        "requirement_text": requirement_text
    }


def process_user_profile_from_history(history: List[Dict[str, str]]) -> CandidateProfile:
    """Возвращает Pydantic-модель CandidateProfile."""
    
    data = extract_user_data_from_history(history)

    return CandidateProfile(
        requirement_responsibility=data["requirement_text"],
        skills=data["skills"],
        experience=data["experience_enum"]
    )
