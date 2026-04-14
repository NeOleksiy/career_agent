import polars as pl
from ..user_profile.schema import CandidateProfile
from .vectorize import VacancySearchEngine

df = pl.read_parquet("/Users/yakto/projects/career_agent/parser/data/*")
print("read")

search_engine = VacancySearchEngine("efederici/sentence-bert-base")
search_engine.load_index("./data_artefacts/faiss_index.index", df)
# search_engine.fit(df)

user_input = CandidateProfile(
    title="CV engineer",
    requirement_responsibility="Заниматься генеративными моделями, быть ближе к Computer Vision, заниматься детекцией",
    skills=["Python", "SQL", "Computer Vision"]
)
print("fit")

search_engine.save_index("./data_artefacts/faiss_index.index")

results = search_engine.search(
    user_input,
    top_n=10,
)

print(results.select(["vacancy_id", "title", "company", "similarity_score"]))