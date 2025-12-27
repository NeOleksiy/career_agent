import polars as pl
import numpy as np
import logging
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from vectorize.schema import CandidateProfile, ExperienceLevel
from metrics.logger import log_metric
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

        # --- METRIC 1: проверка готовности поиска ---
        if self.index is None or self.df is None:
            log_metric("search_not_initialized", 1)
            raise ValueError("Сначала необходимо обучить модель методом fit()")

        # --- METRIC 2: тип запроса ---
        log_metric(
            "search_query_type",
            "profile" if isinstance(query, CandidateProfile) else "text"
        )

        if isinstance(query, CandidateProfile):
            query_text = query.to_bert_string()
        else:
            query_text = query

        # --- METRIC 3: длина запроса ---
        log_metric("query_text_length", len(query_text))

        # --- METRIC 4: latency эмбеддингов ---
        start_embed = time.time()
        query_vector = self.model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype(np.float32)
        log_metric("embedding_latency", time.time() - start_embed)

        # --- METRIC 5: latency FAISS ---
        start_search = time.time()
        distances, indices = self.index.search(
            query_vector,
            min(top_n * 3, len(self.df))
        )
        log_metric("faiss_latency", time.time() - start_search)

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

        # --- METRIC 6: результаты поиска ---
        log_metric("search_results_count", len(results))

        if not results:
            log_metric("search_empty", 1)
            return pl.DataFrame()

        # --- METRIC 7: качество выдачи ---
        similarities = [r["similarity_score"] for r in results]
        log_metric("search_mean_similarity", sum(similarities) / len(similarities))
        log_metric("search_min_similarity", min(similarities))
        log_metric("search_max_similarity", max(similarities))

        # --- METRIC 8: разнообразие ---
        companies = {r["company"] for r in results if "company" in r}
        log_metric("search_unique_companies", len(companies))

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
