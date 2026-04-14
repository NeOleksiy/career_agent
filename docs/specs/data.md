## Данные
Данные представляют собой вакансии из hh.ru, полученные по официальному api, связанные с Data Science. Основные поля, с которыми будет работать llm:

Список ключевых слов для сбора вакансий:

```
   'Computer vision',
   'Data Analyst', 'Data Engineer', 'Data Science', 'Data Scientist', 'ML Engineer',
   'MLOps инженер', 'AI',
   'Product Manager', 'Python Developer', 'Web Analyst', 'Аналитик данных',
   'Бизнес-аналитик', 'Системный аналитик', 'Финансовый аналитик', 'ML',
   'Deep Learning', 'NLP', 'LLM', "Project Manager", 'Product Owner','Time series',
```

- **requirements** - описание требования от соискателя в вакансии
- **responsbility** - описание задач, которые будет выполнять потенциальный работник
- **skills** - теги навыков вакансии (например SQL, Python и тд.)
- **work_format_names** - формат работы: гибрид, офис или удалёнка
- **company** - компания факансии
- **salary** - зарплата в рублях( где-то 30-40% не null)
- **experience** - класс опыта работы ( 0, 1-3, 3-6, 6+ лет)
  
Список ключевых слов для сбора вакансий:

Вакансий около 10 тысяч, в данный момент хранятся в parquet локально
Вакансии спаршены в середине декабря 2025 года
([data_artefacts/vacancy_final.parquet](data_artefacts/vacancy_final.parquet))