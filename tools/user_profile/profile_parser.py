import requests
from bs4 import BeautifulSoup
import json
import re

from .schema import UserProfile


async def parse_hh_resume(url: str)-> UserProfile:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. Заголовок
        title_el = soup.find('span', {'data-qa': 'resume-block-title-position'})
        title = title_el.get_text(strip=True) if title_el else "Не указано"

        # 2. Зарплата (число)
        salary_block = soup.find(class_='resume-block__salary')
        salary = None
        if salary_block:
            salary_raw = ''.join(re.findall(r'\d+', salary_block.get_text()))
            salary = int(salary_raw) if salary_raw else None

        # 3. Общий опыт работы (число лет)
        experience_total_years = 0
        # Ищем блок, где написано "Опыт работы 5 лет 2 месяца"
        exp_total_el = soup.find('span', class_='resume-block__title-text_sub')
        if exp_total_el:
            text = exp_total_el.get_text().lower()
            # Извлекаем года и месяцы через регулярку
            years_match = re.search(r'(\d+)\s+(?:год|лет|года)', text)
            months_match = re.search(r'(\d+)\s+(?:месяц|месяца|месяцев)', text)
            
            years = int(years_match.group(1)) if years_match else 0
            months = int(months_match.group(1)) if months_match else 0
            
            # Переводим в число (например, 5 лет и 6 месяцев = 5.5 лет)
            experience_total_years = round(years + (months / 12), 1)

        # 4. Скиллы (без дублей)
        skills = []
        seen_skills = set()
        skills_container = soup.find('div', class_='bloko-tag-list')
        if skills_container:
            for tag in skills_container.find_all(recursive=True):
                txt = tag.get_text(strip=True)
                if txt and txt.lower() not in seen_skills:
                    skills.append(txt)
                    seen_skills.add(txt.lower())

        # 5. Детальный опыт (без дублей)
        experience_items = []
        exp_sections = soup.find_all('div', class_='resume-block-item-gap')
        for section in exp_sections:
            pos_el = section.find('div', {'data-qa': 'resume-block-experience-position'})
            desc_el = section.find('div', {'data-qa': 'resume-block-experience-description'})
            if pos_el or desc_el:
                entry = {
                    'position': pos_el.get_text(strip=True) if pos_el else "",
                    'description': desc_el.get_text(separator='\n', strip=True) if desc_el else ""
                }
                if entry not in experience_items:
                    experience_items.append(entry)

        # 6. Образование (без дублей)
        education = []
        edu_section = soup.find('div', {'data-qa': 'resume-block-education'}) or soup.find(class_='resume-block-education')
        if edu_section:
            edu_items = edu_section.find_all('div', class_='resume-block-item-gap')
            for item in edu_items:
                name = item.find(attrs={'data-qa': 'resume-block-education-name'})
                org = item.find(attrs={'data-qa': 'resume-block-education-organization'})
                if name or org:
                    edu_entry = {
                        'name': name.get_text(strip=True) if name else "",
                        'specialization': org.get_text(strip=True) if org else ""
                    }
                    if edu_entry not in education:
                        education.append(edu_entry)

        raw_data = {
            "title": title,
            "salary": salary,
            "experience_total_years": experience_total_years,
            "skills": skills,
            "experience_detailed": experience_items,
            "education": education
        }

        return UserProfile(**raw_data)

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    url = "https://hh.ru/resume/1b220398ff0c7a5e380039ed1f53494a695a61" 

    data = parse_hh_resume(url)
    print(data)
