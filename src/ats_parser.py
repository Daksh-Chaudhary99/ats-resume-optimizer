import re
import spacy
from sentence_transformers import SentenceTransformer, util


class ATSParser:
    """Deterministic and semantic parsing engine simulating enterprise ATS screening."""

    STANDARD_SECTIONS = [
        "experience",
        "work experience",
        "employment history",
        "education",
        "skills",
        "technical skills",
        "projects",
        "certifications",
    ]

    TECH_SYNONYMS = {
        "cad": ["computer-aided design", "cad models", "3d cad"],
        "gcp": ["google cloud platform", "google cloud"],
        "aws": ["amazon web services"],
        "js": ["javascript"],
        "ml": ["machine learning"],
        "ai": ["artificial intelligence"],
        "fea": ["finite element analysis"],
        "hvac": ["heating, ventilation, and air conditioning"]
    }

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.82, 
    ):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        
        # Reverse the synonym dictionary for bi-directional lookup once upon initialization
        self.reverse_synonyms = {}
        for key, values in self.TECH_SYNONYMS.items():
            for val in values:
                self.reverse_synonyms[val] = key

    def _detect_sections(self, text: str) -> list[str]:
        found_sections = []
        lower_text = text.lower()
        for section in self.STANDARD_SECTIONS:
            pattern = rf"(?m)^\s*{re.escape(section)}\s*[:\n\r]"
            if re.search(pattern, lower_text):
                found_sections.append(section)
        return list(set(found_sections))

    def _extract_candidate_noun_chunks(self, text: str) -> list[str]:
        doc = self.nlp(text)
        candidate_phrases = set()

        for chunk in doc.noun_chunks:
            cleaned = " ".join(
                [
                    token.text
                    for token in chunk
                    if token.pos_ not in ("PRON", "DET", "PUNCT", "SPACE")
                    and not token.is_stop
                ]
            ).strip()

            if len(cleaned) > 2 and not cleaned.isnumeric():
                candidate_phrases.add(cleaned.lower())

        return list(candidate_phrases)
        
    def _get_safe_pattern(self, term: str) -> str:
        """Helper function for safe word boundaries (handles C++, C#, etc.)"""
        escaped = re.escape(term)
        end_boundary = r"\b" if term[-1].isalnum() else r"(?!\w)"
        start_boundary = r"\b" if term[0].isalnum() else r"(?<!\w)"
        return rf"{start_boundary}{escaped}{end_boundary}"

    def calculate_score(self, resume_text: str, jd_skills: dict[str, list[str]]) -> dict:
        technical_skills = jd_skills.get("technical_skills", [])
        domain_knowledge = jd_skills.get("domain_knowledge", [])
        soft_skills = jd_skills.get("soft_skills", [])
        
        all_skills = technical_skills + domain_knowledge + soft_skills
        hard_skills_count = len(technical_skills) + len(domain_knowledge)

        if not all_skills:
            return {"score": 0, "sections_found": [], "semantic_matches": []}

        lower_resume = resume_text.lower()
        sections_found = self._detect_sections(resume_text)

        exact_matches = []
        synonym_matches = []
        unmatched_skills = []

        # 1. Exact string matching using safe boundaries
        for skill in all_skills:
            clean_skill = skill.strip().lower()
            pattern = self._get_safe_pattern(clean_skill)
            
            if re.search(pattern, lower_resume):
                exact_matches.append(clean_skill)
            else:
                unmatched_skills.append(clean_skill)

        # 2. Bi-directional Hard Synonym Checking
        skills_for_vector_engine = []
        for skill in unmatched_skills:
            matched_via_synonym = False
            
            base_key = self.reverse_synonyms.get(skill, skill if skill in self.TECH_SYNONYMS else None)
            
            if base_key:
                equivalents = [base_key] + self.TECH_SYNONYMS[base_key]
                if skill in equivalents:
                    equivalents.remove(skill)
                    
                for eq_term in equivalents:
                    pattern = self._get_safe_pattern(eq_term)
                    if re.search(pattern, lower_resume):
                        synonym_matches.append(skill)
                        matched_via_synonym = True
                        break
                        
            if not matched_via_synonym:
                skills_for_vector_engine.append(skill)

        hard_skill_matches = [
            skill for skill in (exact_matches + synonym_matches) 
            if skill in (technical_skills + domain_knowledge)
        ]
        
        score = 0
        if hard_skills_count > 0:
            score = int((len(hard_skill_matches) / hard_skills_count) * 100)
            score = min(score, 100)

        # 3. Semantic matching on remaining unmatched skills
        semantic_matches = []
        if skills_for_vector_engine:
            resume_phrases = self._extract_candidate_noun_chunks(resume_text)

            if resume_phrases:
                jd_embeddings = self.embedding_model.encode(
                    skills_for_vector_engine, convert_to_tensor=True
                )
                resume_embeddings = self.embedding_model.encode(
                    resume_phrases, convert_to_tensor=True
                )

                similarity_matrix = util.cos_sim(jd_embeddings, resume_embeddings)

                for i, skill in enumerate(skills_for_vector_engine):
                    best_match_idx = int(similarity_matrix[i].argmax())
                    highest_sim = float(similarity_matrix[i][best_match_idx])

                    if highest_sim >= self.similarity_threshold:
                        semantic_matches.append(
                            {
                                "jd_skill": skill,
                                "resume_term": resume_phrases[best_match_idx],
                                "similarity_score": round(highest_sim, 2),
                            }
                        )

        return {
            "score": score,
            "sections_found": sections_found,
            "semantic_matches": semantic_matches,
            "synonym_matches": synonym_matches,
            "exact_matches": exact_matches
        }