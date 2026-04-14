# Документация Retriever для поиска вакансий

## 1. Источники данных (Sources)

| Источник | Формат | Описание |
|----------|--------|----------|
| **База вакансий** | Parquet | Содержит поля: `title`, `company`, `experience`, `salary`, `skills`, `requirement`, `responsibility`, `vacancy_id` |
| **FAISS-индекс** | бинарный файл | Предварительно сохранённый индекс эмбеддингов для быстрого запуска |

Загрузка: `pl.read_parquet(database_url)`, дедупликация по `vacancy_id`.

## 2. Индекс (Index)

### Создание
1. Из каждой вакансии формируется текст: `title` + `requirement` + `responsibility` + `skills` (через `to_bert_string()`)
2. Векторизация моделью `SentenceTransformer("efederici/sentence-bert-base")` → эмбеддинги `float32`
3. Построение плоского FAISS-индекса с L2-метрикой: `faiss.IndexFlatL2`
4. Сохранение на диск: `faiss.write_index()`

### Загрузка
- Если индекс существует: `faiss.read_index()`, восстанавливаются метаданные
- Иначе создаётся заново через `fit()`

## 3. Поиск (Search)

### Методы
- `search(query, top_n=5, filters=None)` – поиск по строке или `CandidateProfile`
- `search_by_profile(profile, ...)` – обёртка

### Алгоритм
1. Векторизация запроса (той же моделью)
2. `index.search(query_vector, k = top_n * 3)` – получение кандидатов
3. Пост-фильтрация по полям (если заданы)
4. Поиск по faiss индексу
5. Возврат `top_n` результатов в `pl.DataFrame`

### Пример
```python
engine = VacancySearchEngine()
engine.fit(vacancies_df)
results = engine.search("python developer", top_n=10, filters={"city": "Moscow"})