import os
import logging
from typing import List, Any, Dict
from vectorize.schema import CandidateProfile

# Настройка красивого вывода в консоль
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Validation")

class ConsoleValidator:
    
    @staticmethod
    def validate_env():
        """Проверка переменных окружения"""
        required_keys = ["API_TOKEN", "MODEL_URL", "MODEL_NAME"]
        missing = [key for key in required_keys if not os.getenv(key)]
        
        if missing:
            logger.error(f"❌ ОШИБКА КОНФИГУРАЦИИ: Отсутствуют ключи: {', '.join(missing)}")
        else:
            logger.info("✅ Переменные окружения загружены корректно.")

    @staticmethod
    def validate_profile(profile: CandidateProfile):
        """Проверка качества извлеченного профиля"""
        print("\n" + "="*50)
        print("🔍 ВАЛИДАЦИЯ ПРОФИЛЯ ПОЛЬЗОВАТЕЛЯ (Internal)")
        
        # Проверка навыков
        if not profile.skills:
            print("⚠️  Warning: Навыки не обнаружены. Поиск может быть неточным.")
        else:
            print(f"✅ Навыки ({len(profile.skills)}): {', '.join(profile.skills)}")

        # Проверка текста требований
        text_len = len(profile.requirement_responsibility)
        if text_len < 20:
            print(f"⚠️  Warning: Слишком короткое описание ({text_len} симв.). Нужно больше контекста.")
        else:
            print(f"✅ Длина описания: {text_len} симв.")

        # Проверка опыта
        # print(f"✅ Уровень опыта: {profile.experience}")
        # print("="*50 + "\n")

    @staticmethod
    def validate_search_results(results: List[Any]):
        """Проверка результатов поиска в FAISS"""
        if not results:
            logger.error("❌ ПОИСК: FAISS вернул 0 результатов. Проверьте индекс или эмбеддинги.")
        else:
            logger.info(f"✅ ПОИСК: Найдено {len(results)} подходящих вакансий.")

    @staticmethod
    def validate_llm_response(response: str):
        """Проверка того, что пришло от Mistral"""
        if not response or len(response.strip()) == 0:
            logger.error("❌ LLM: Получен пустой ответ от модели.")
        elif "error" in response.lower():
            logger.warning(f"⚠️  LLM: В ответе содержится упоминание ошибки: {response[:100]}...")
        else:
            logger.info(f"✅ LLM: Ответ успешно получен (длина: {len(response)}).")