from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-r1:latest", base_url="https://ollama.gti-ia.upv.es/v1", temperature=0)
# llm = ChatOpenAI(model="gpt-4.1", base_url="https://api.openai.com/v1", temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'."
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a lenient grader assessing relevance of retrieved documents to a user question about EPDs (Environmental Product Declarations). \n
Grade as RELEVANT ('yes') if the document:
- Contains ANY keywords, product names, or technical terms from the question
- Discusses the same product family, manufacturer, or material type
- Provides contextual information that could help answer the question
- Contains numerical data, specifications, or lifecycle assessment information

Only grade as NOT RELEVANT ('no') if the document is:
- Completely unrelated to the product, manufacturer, or topic
- About a different industry or domain entirely

When in doubt, prefer 'yes' to ensure comprehensive answers."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader