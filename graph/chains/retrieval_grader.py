from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="deepseek-r1:latest", base_url="https://ollama.gti-ia.upv.es/v1", temperature=0)
llm = ChatOpenAI(model="gpt-4.1", base_url="https://api.openai.com/v1", temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'."
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a document filter for an industrial machinery documentation system.
Your ONLY job is to discard documents that are completely unrelated to the question. You are NOT scoring quality or completeness — you are acting as a last-resort spam filter.

Always answer 'yes' (keep the document) if ANY of the following are true:
- The document mentions the same machine, model, or component as the question (even partially)
- The document is from the same industrial domain (manufacturing, maintenance, safety, etc.)
- The document contains technical data, parameters, procedures, part numbers, or specifications of any kind
- The document was retrieved alongside other documents that seem relevant (retrieval systems don't return random documents)
- You are unsure — default to 'yes'

Only answer 'no' (discard) if ALL of the following are true:
- The document is clearly about an entirely different machine or unrelated industry with zero overlap
- There is absolutely no term, concept, or data point that could contribute to answering the question
- You are 100% certain this document was retrieved by mistake

Err heavily on the side of keeping documents. A false negative (discarding a useful document) is far more harmful than a false positive (keeping a marginally relevant one)."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader