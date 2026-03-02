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

system = """You are a relevance grader for an industrial machinery documentation system.
Your job is to decide whether a retrieved document chunk is useful for answering a user question about industrial machines, their operation, maintenance, components, or safety.

Grade as RELEVANT ('yes') if the document chunk:
- Mentions the same machine, model number, or component referenced in the question
- Describes a procedure, parameter, or configuration related to the question topic
- Contains technical specifications, part numbers, or step-by-step instructions relevant to the question
- Provides safety, compliance, or regulatory information (DGUV, ISO, IEC, etc.) related to the question
- Is from the same machine family or document type (manual, maintenance plan, spare parts list, wiring diagram, etc.)
- Contains any numerical data, error codes, or settings that could help answer the question

Grade as NOT RELEVANT ('no') ONLY if the document chunk:
- Is about a completely different machine with no connection to the question
- Is from a totally unrelated industrial domain or process
- Contains no information that could plausibly help answer the question

Important: Documents retrieved by machine ID or entity match are very likely relevant even if they don't directly answer the exact question — keep them.
When in doubt, prefer 'yes'."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader