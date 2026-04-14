# Инструменты парсинга и рекомендаций

## parse_hh_resume()

Парсер резюме hh.ru по прямой ссылке.

| Свойство | Описание |
|----------|----------|
| Назначение | Парсинг резюме с hh.ru по URL |
| Вход | Ссылка на резюме (например, `https://hh.ru/resume/...`) |
| Выход | Структурированные данные: опыт, навыки, образование, желаемая должность, зарплата |

---

## class Vacancy_parser

Тул для создания базы данных вакансий с hh.ru.

**Используемые запросы:**

```python
vacancies = [
    'Computer vision',
    'Data Analyst', 'Data Engineer', 'Data Science', 'Data Scientist', 'ML Engineer',
    'MLOps инженер', 'AI',
    'Product Manager', 'Python Developer', 'Web Analyst', 'Аналитик данных',
    'Бизнес-аналитик', 'Системный аналитик', 'Финансовый аналитик', 'ML',
    'Deep Learning', 'NLP', 'LLM', 'Project Manager', 'Product Owner', 'Time series',
]
```

## Recommender Тул для рекомендации вакансий на основе профиля кандидата.

Входной формат – CandidateProfile:

```python
class CandidateProfile(BaseModel):
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
```

Выходные данные:

Поле	Тип	Описание
recommendations	List[Dict]	Список вакансий с title, company, salary, skills, similarity_score
expanded_skills	List[str]	Навыки из найденных вакансий (для обогащения профиля)
career_paths	List[str]	Примеры карьерных траекторий (должность + компания)
Основные методы:

Метод	Описание
init_search_engine()	Загружает FAISS-индекс и данные вакансий
recommend_vacancies(profile, top_k=10)	Асинхронный поиск вакансий по профилю


## read_vacancies_by_title
Тул для поиска вакансий по названию специализации.

Свойство	Описание
Назначение	Находит все вакансии по title специализации пользователя
Вход	title – строка (например, "Data Scientist")
Выход	DataFrame с вакансиями, где поле title соответствует запросу
Логика поиска	Точное совпадение → частичное совпадение → по ключевым навыкам
Кэширование	Да
Пример:

python
vacancies_df = read_vacancies_by_title("ML Engineer")
# Возвращает все вакансии, связанные с ML Engineer
text
