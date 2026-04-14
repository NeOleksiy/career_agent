import polars as pl
import numpy as np
import logging
import faiss
import os
from logger import logger
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from tools.user_profile.schema import CandidateProfile, ExperienceLevel
import time


class VacancySearchEngine:
    """Класс для поиска вакансий с использованием Sentence-BERT и FAISS."""
    
    def __init__(self, model_name: str = "efederici/sentence-bert-base"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.df = None
        self.dimension = None
        self.vacancy_profiles = []
        
    # Заполняет модель pydantic данными из датасета
    def _create_vacancy_profile(self, row: dict) -> CandidateProfile:
        """Создает CandidateProfile из строки вакансии."""

        req_resp = ""
        if "requirement" in row and row["requirement"]:
            req_resp += row["requirement"].lower()
        if "responsibility" in row and row["responsibility"]:
            if req_resp:
                req_resp += " "
            req_resp += row["responsibility"].lower()

        title = ""
        if "title" in row and row["title"]:
            title += row["title"].lower()

        
        skills = []
        if "skills" in row and row["skills"]:
            if isinstance(row["skills"], str):
                skills = [skill.strip() for skill in row["skills"].split(",")]
            elif isinstance(row["skills"], list):
                skills = row["skills"]
        
        experience = ExperienceLevel.NO_EXPERIENCE
        if "experience" in row and row["experience"]:
            exp_str = row["experience"].lower()
            if "1" in exp_str and "3" in exp_str:
                experience = ExperienceLevel.ONE_TO_THREE
            elif "3" in exp_str and "6" in exp_str:
                experience = ExperienceLevel.THREE_TO_SIX
            elif "6" in exp_str or "более" in exp_str:
                experience = ExperienceLevel.MORE_THAN_SIX
        
        return CandidateProfile(
            title=title,
            requirement_responsibility=req_resp,
            skills=skills,
            # experience=experience
        )
    
    def fit(self, df: pl.DataFrame) -> None:
        """
        Векторизация вакансий и создание FAISS индекса.
        
        Args:
            df: DataFrame Polars с вакансиями
        """
        self.df = df
        
        self.vacancy_profiles = []
        texts = []
        
        for row in df.iter_rows(named=True):
            profile = self._create_vacancy_profile(row)
            self.vacancy_profiles.append(profile)
            texts.append(profile.to_bert_string())
        
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self.dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        logging.info(f"Индекс создан для {len(embeddings)} вакансий, размерность: {self.dimension}")
    
    def search(
        self,
        query: Union[str, CandidateProfile],
        top_n: int = 5,
        filters: Dict[str, Any] = None
    ) -> pl.DataFrame:

        if self.index is None or self.df is None:
            raise ValueError("Сначала необходимо обучить модель методом fit()")

        if isinstance(query, CandidateProfile):
            query_text = query.to_bert_string()
        else:
            query_text = query

        start_embed = time.time()
        query_vector = self.model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype(np.float32)

        start_search = time.time()
        distances, indices = self.index.search(
            query_vector,
            min(top_n * 3, len(self.df))
        )

        results = []

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.df):
                continue

            row = self.df.row(idx, named=True)

            if filters:
                skip = False
                for field, value in filters.items():
                    if field in row and row[field] != value:
                        skip = True
                        break
                if skip:
                    continue

            similarity = float(1 / (1 + distances[0][i]))

            result = dict(row)
            result["similarity_score"] = similarity
            results.append(result)

            if len(results) >= top_n:
                break


        if not results:
            return pl.DataFrame()

        similarities = [r["similarity_score"] for r in results]
        companies = {r["company"] for r in results if "company" in r}
        
        return pl.DataFrame(results)
    
    def search_by_profile(self, profile: CandidateProfile, top_n: int = 5, filters: Dict[str, Any] = None) -> pl.DataFrame:
        """
        Поиск вакансий по профилю кандидата.
        
        Args:
            profile: Профиль кандидата
            top_n: Количество возвращаемых результатов
            filters: Словарь фильтров (поле: значение)
            
        Returns:
            DataFrame с результатами поиска
        """
        return self.search(profile, top_n, filters)
    
    def save_index(self, index_path: str) -> None:
        """Сохраняет FAISS индекс на диск."""
        if self.index is None:
            raise ValueError("Индекс не создан")
        faiss.write_index(self.index, index_path)
    
    def load_index(self, index_path: str, df: pl.DataFrame) -> None:
        """Загружает FAISS индекс с диска."""
        self.df = df
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        self.vacancy_profiles = []
        for row in df.iter_rows(named=True):
            self.vacancy_profiles.append(self._create_vacancy_profile(row))



class Recommender:
    def __init__(self, database_url, index_url):
        self.database_url = database_url
        self.index_url = index_url
        self.VACANCY_DF = None
        self.SEARCH_ENGINE = None

    def init_search_engine(self):
        """Инициализация поискового движка"""
        global VACANCY_DF, SEARCH_ENGINE
        
        try:
            logger.info("Загрузка данных вакансий...")
            VACANCY_DF = pl.read_parquet(self.database_url)
            VACANCY_DF = VACANCY_DF.unique("vacancy_id")
            logger.info(f"Загружено {len(VACANCY_DF)} вакансий")
            
            logger.info("Создание поискового движка...")
            SEARCH_ENGINE = VacancySearchEngine("efederici/sentence-bert-base")

            if os.path.exists(self.index_url):
                SEARCH_ENGINE.load_index(self.index_url, VACANCY_DF)
            else:
                SEARCH_ENGINE.fit(VACANCY_DF)
                logger.info("Поисковый движок создан")
                SEARCH_ENGINE.save_index(self.index_url)
                logger.info("Индекс сохранен")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации поискового движка: {e}")
            raise


    async def recommend_vacancies(self, profile: CandidateProfile, top_k: int = 10) -> tuple:
        """
        Поиск вакансий с использованием VacancySearchEngine
        
        Args:
            profile: профиль кандидата
            top_k: количество возвращаемых вакансий
        
        Returns:
            tuple: (рекомендации, расширенные навыки, карьерные пути)
        """
        global SEARCH_ENGINE
        
        if SEARCH_ENGINE is None:
            self.init_search_engine()
        
        try:
            # Выполняем поиск
            results = SEARCH_ENGINE.search(profile, top_n=top_k)
            
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
            
            # Карьерные пути можно генерировать н.а основе найденных вакансий
            career_paths = []
            for rec in recommendations[:5]:  # Берем топ-5 вакансий
                if rec['title'] and rec['company']:
                    career_paths.append(f"{rec['title']} в {rec['company']}")
            
            return recommendations, list(expanded_skills)[:15], career_paths[:5]
            
        except Exception as e:
            logger.error(f"Ошибка поиска вакансий: {e}")
            return [], [], []
