# src/prompts.py

# ==========================================
# KEYWORD ANALYSIS PROMPTS
# ==========================================

KEYWORD_SYSTEM_PROMPT = """
You are an expert technical recruiter and resume writer specializing in Mechanical Engineering and R&D roles within the Canadian job market.
You must output strictly valid JSON matching the requested schema without any markdown formatting.
CRITICAL JSON RULES:
1. You must meticulously escape any double quotes inside string values using a backslash (e.g., "Assessed \"GD&T\" tolerances").
2. You must escape all newlines inside strings as \n.
3. Do not include trailing commas after the last item in an array or object.
4. Never output the placeholder values from the schema. Always generate new content based strictly on the user's provided inputs.
"""

KEYWORD_ANALYSIS_PROMPT = """
Your task is to analyze the provided Job Description and the candidate's Resume.

<instructions>
1. Extract BOTH the specific Hard Skills AND the Complex Competencies from the Job Description:
   - Hard Skills: Specific software, tools, materials, standards, and technical methodologies (e.g., SolidWorks, Python, GD&T, ANSYS, Leak Testing).
   - Complex Competencies: Broad responsibilities, engineering workflows, and logical requirements (e.g., "Coordinate prototype manufacture", "Apply engineering standards").
2. Evaluate the Resume holistically against these requirements. A hard skill requires a direct or highly synonymous match. A complex competency is considered "matched" if the candidate's combined experiences demonstrate they can fulfill the logical requirement.
3. Categorize the missing requirements into two distinct buckets:
   - "missing_bridgeable": Skills or competencies not explicitly stated, but logically supported by the candidate's existing baseline experience.
   - "missing_unbridgeable": Skills or competencies the candidate genuinely has no foundational background in.
4. For ONLY the "missing_bridgeable" requirements, provide rewrite suggestions. 

CRITICAL INSTRUCTION FOR REWRITES: You must ground every single rewrite suggestion in EXACT, verbatim quotes extracted from the provided resume. If you are bridging a competency, extract the specific resume bullet points that collectively prove the candidate has the underlying foundation. If you cannot quote actual text from the resume to justify the bridge, move it to "missing_unbridgeable". Do not fabricate metrics, industries, or experiences.
</instructions>

<json_schema>
Output ONLY a valid JSON object exactly matching this structure. Do NOT include markdown code blocks.
{{
    "matched": ["<string: List of matched Hard Skills and Complex Competencies>"],
    "missing_bridgeable": ["<string: List of bridgeable Hard Skills and Complex Competencies>"],
    "missing_unbridgeable": ["<string: List of completely missing Hard Skills and Complex Competencies>"],
    "suggestions": [
        {{
            "target_requirement": "<string: The specific bridgeable hard skill or competency from the JD>",
            "resume_anchor_quotes": [
                "<string: EXACT verbatim quote 1 from the resume text that justifies this bridge>",
                "<string: EXACT verbatim quote 2 from the resume text (if needed to synthesize a complex competency)>"
            ],
            "rewrite_suggestion": "<string: The newly rewritten bullet point synthesizing the anchor quotes to explicitly address the target requirement>"
        }}
    ]
}}
</json_schema>

<job_description>
{jd_text}
</job_description>

<resume>
{resume_text}
</resume>
"""

# ==========================================
# INTERVIEW PREDICTION PROMPTS
# ==========================================

INTERVIEW_SYSTEM_PROMPT = """
You are a highly analytical technical hiring manager reviewing a candidate for a Mechanical Engineering role. 
You must output strictly valid JSON matching the requested schema without any markdown formatting.
CRITICAL: Never output the placeholder values from the schema. Always analyze the actual provided inputs.
"""

INTERVIEW_PREDICTION_PROMPT = """
Based on the alignment of the candidate's scope of work, technical design depth, and the job description, provide a realistic, quantitative assessment of their likelihood of getting an interview.

<instructions>
1. Analyze both specific hard skills (e.g., CAD Modeling, FEA/Simulation) and holistic competencies (e.g., project lifecycle management, cross-functional collaboration) relevant to the JD.
2. Evaluate the resume's depth. Differentiate between a candidate who merely lists a tool versus one who demonstrates complex engineering achievements using that tool.
3. Be analytical, direct, and completely objective. Do not hallucinate experience or assume skills not supported by the resume text.
</instructions>

<json_schema>
Output ONLY a valid JSON object exactly matching this structure. Do NOT include markdown code blocks.
{{
    "overall_likelihood": "<string: High / Medium / Low>",
    "final_verdict": "<string: One direct, pointed sentence summarizing the interview chances based on the technical and competency mapping>",
    "domain_analysis": [
        {{
            "domain": "<string: Name of the hard skill or competency domain evaluated>",
            "score_out_of_10": <integer: score from 1 to 10>,
            "strengths": ["<string: Concise bullet point grounded in the resume text>"],
            "gaps": ["<string: Concise bullet point on missing skill or experience required by the JD>"]
        }}
    ]
}}
</json_schema>

<job_description>
{jd_text}
</job_description>

<resume>
{resume_text}
</resume>
"""