import os
import spacy
import streamlit as st

from src.config import Config
from src.model import AzureOpenAIProvider, HuggingFaceProvider
from src.interview_predictor import InterviewPredictor
from src.keyword_analyzer import KeywordAnalyzer
from src.pdf_processor import PDFProcessor

st.set_page_config(
    page_title="ATS Resume Optimizer",
    page_icon="📄",
    layout="wide"
)

st.title("ATS & Resume Optimization Tool")

# ==========================================
# PROVIDER INITIALIZATION (Azure vs Hugging Face)
# ==========================================
def get_llm_provider():
    """Initializes the active LLM provider based on config settings."""
    provider_type = Config.ACTIVE_LLM_PROVIDER.lower()
    
    if provider_type == "azure":
        if not Config.AZURE_ENDPOINT or not Config.AZURE_API_KEY:
            st.error("Azure credentials missing in .env file.")
            st.stop()
        return AzureOpenAIProvider(
            endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION,
            deployment_name=Config.AZURE_DEPLOYMENT_NAME
        )
        
    elif provider_type == "huggingface":
        if not Config.HF_TOKEN:
            st.error("Hugging Face token (`HF_TOKEN`) missing in .env file.")
            st.stop()
        return HuggingFaceProvider(
            token=Config.HF_TOKEN,
            model_name=Config.HF_MODEL_NAME
        )
        
    else:
        st.error(f"Unsupported provider specified: '{provider_type}'. Use 'azure' or 'huggingface'.")
        st.stop()

# Instantiate lightweight providers and application services
llm_provider = get_llm_provider()
keyword_analyzer = KeywordAnalyzer(llm_provider)
interview_predictor = InterviewPredictor(llm_provider)

# Use Streamlit's cache so the heavy ML models only load into memory ONCE
@st.cache_resource(show_spinner="Loading NLP models into memory (this may take a minute on first run)...")
def load_ats_parser():
    from src.ats_parser import ATSParser
    return ATSParser()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Settings & Uploads")
    st.info(f"**Active Engine:** {Config.ACTIVE_LLM_PROVIDER.upper()}")
    
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_text_input = st.text_area("Paste Job Description Here", height=300)
    analyze_button = st.button("Run Analysis", type="primary", use_container_width=True)

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if analyze_button:
    if not resume_file or not jd_text_input.strip():
        st.warning("Please upload a resume PDF and paste a job description before running the analysis.")
    else:
        try:
            # Load the heavy ATS parser only AFTER the user clicks the button
            ats_parser = load_ats_parser()

            # 1. Extract raw text from PDF
            with st.spinner("Extracting text from resume..."):
                resume_text = PDFProcessor.extract_text(resume_file)

            if not resume_text.strip():
                st.error("Could not extract any readable text from the uploaded PDF. Please check the document formatting.")
                st.stop()

            # 2. Setup 4 Application Tabs
            tab_raw, tab1, tab2, tab3 = st.tabs([
                "📄 Raw Extracted Text",
                "🎯 Keyword Analysis & Rewrites",
                "📊 ATS Match Score",
                "👔 Interview Likelihood"
            ])

            # --------------------------------------
            # TAB 0: Raw Extracted Text
            # --------------------------------------
            with tab_raw:
                st.subheader("Raw Resume Text Extraction")
                st.caption("Verify how standard ATS parsers read your document sequentially.")
                
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.metric("Total Characters", len(resume_text))
                with col_meta2:
                    st.metric("Total Words", len(resume_text.split()))

                st.text_area(
                    label="Extracted Content",
                    value=resume_text,
                    height=450,
                    disabled=True
                )

            # --------------------------------------
            # TAB 1: Keyword Analysis & Rewrites
            # --------------------------------------
            with st.spinner("Analyzing keyword alignment and generating rewrites..."):
                analysis_results = keyword_analyzer.analyze(resume_text, jd_text_input)

            with tab1:
                st.subheader("Keyword Gap Analysis")
                
                matched = analysis_results.get("matched", [])
                bridgeable = analysis_results.get("missing_bridgeable", [])
                unbridgeable = analysis_results.get("missing_unbridgeable", [])
                suggestions = analysis_results.get("suggestions", [])

                st.markdown("#### ✅ Matched Skills")
                if matched:
                    st.write(", ".join(matched))
                else:
                    st.info("No direct keyword matches identified.")

                st.markdown("#### 🌉 Bridgeable Missing Skills")
                st.caption("Skills related to your existing background that can be logically adapted into your experience.")
                if bridgeable:
                    st.write(", ".join(bridgeable))
                else:
                    st.write("None identified.")

                st.markdown("#### ❌ Unbridgeable Missing Skills")
                st.caption("Required competencies currently outside your documented skill set.")
                if unbridgeable:
                    st.write(", ".join(unbridgeable))
                else:
                    st.write("None identified.")

                st.markdown("#### 💡 Resume Rewrite Suggestions")
                if suggestions:
                    for suggestion in suggestions:
                        # Check if the LLM provided a structured dictionary
                        if isinstance(suggestion, dict):
                            req = suggestion.get("target_requirement", "Target Skill")
                            quotes = suggestion.get("resume_anchor_quotes", [])
                            rewrite = suggestion.get("rewrite_suggestion", "")
                            
                            # Create a nice visual card for each suggestion
                            with st.container(border=True):
                                st.markdown(f"##### 🎯 Target Requirement: `{req}`")
                                
                                if quotes:
                                    st.markdown("**Anchor Quotes from Resume:**")
                                    for q in quotes:
                                        # Use Markdown blockquotes for the original text
                                        st.markdown(f"> *{q}*")
                                
                                st.markdown("**Actionable Rewrite:**")
                                # Use a success box to highlight the final recommendation
                                st.success(rewrite)
                        
                        # Fallback just in case the LLM returns a plain string instead of JSON
                        else:
                            st.info(suggestion)
                else:
                    st.write("No specific rewrites required.")

            # --------------------------------------
            # TAB 2: Simulated ATS Match
            # --------------------------------------
            with tab2:
                st.subheader("Simulated ATS Match")
                
                total_jd_skills = matched + bridgeable + unbridgeable

                if total_jd_skills:
                    with st.spinner("Running deterministic and semantic ATS parsing..."):
                        ats_results = ats_parser.calculate_score(resume_text, total_jd_skills)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Exact ATS Match Score", value=f"{ats_results.get('score', 0)}%")
                    
                    with col2:
                        sections = ats_results.get("sections_found", [])
                        st.markdown("**Detected Standard Sections:**")
                        if sections:
                            st.write(", ".join(sections).title())
                        else:
                            st.warning("Standard ATS headers (Experience, Education, Skills, Projects) were not detected.")

                    semantic_matches = ats_results.get("semantic_matches", [])
                    if semantic_matches:
                        st.markdown("---")
                        st.markdown("#### 🔄 Semantic Skill Matches")
                        st.caption("Skills where your resume phrasing closely aligns with the requirement without an exact string match.")
                        for match in semantic_matches:
                            st.markdown(
                                f"- **JD Requirement:** `{match['jd_skill']}` ➔ **Resume Phrase:** `{match['resume_term']}` "
                                f"*(Similarity: {match['similarity_score']})*"
                            )
                else:
                    st.info("No skills available to score ATS match.")

            # --------------------------------------
            # TAB 3: Interview Likelihood
            # --------------------------------------
            with tab3:
                st.subheader("Hiring Manager Assessment")
                with st.spinner("Generating quantitative candidate evaluation..."):
                    feedback_data = interview_predictor.assess_likelihood(resume_text, jd_text_input)
                
                if feedback_data:
                    likelihood = feedback_data.get("overall_likelihood", "Unknown")
                    if "high" in likelihood.lower():
                        st.success(f"**Overall Interview Likelihood:** {likelihood}")
                    elif "medium" in likelihood.lower():
                        st.warning(f"**Overall Interview Likelihood:** {likelihood}")
                    else:
                        st.error(f"**Overall Interview Likelihood:** {likelihood}")
                        
                    st.info(feedback_data.get("final_verdict", "No verdict provided."))
                    
                    st.markdown("---")
                    st.markdown("### Domain Analysis")
                    
                    domains = feedback_data.get("domain_analysis", [])
                    if domains:
                        for domain_info in domains:
                            domain_name = domain_info.get("domain", "Skill Domain")
                            score = domain_info.get("score_out_of_10", "N/A")
                            
                            with st.expander(f"{domain_name} - Score: {score}/10", expanded=True):
                                col_strengths, col_gaps = st.columns(2)
                                with col_strengths:
                                    st.markdown("##### ✅ Strengths")
                                    for strength in domain_info.get("strengths", []):
                                        st.markdown(f"- {strength}")
                                with col_gaps:
                                    st.markdown("##### ⚠️ Gaps")
                                    for gap in domain_info.get("gaps", []):
                                        st.markdown(f"- {gap}")
                    else:
                        st.write("No specific domain breakdown generated.")

        except Exception as e:
            st.error(f"Execution error: {e}")
else:
    st.info("Upload a resume PDF, paste the job description in the sidebar, and click **Run Analysis** to begin.")