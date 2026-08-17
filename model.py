import os
from typing import TypedDict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# --- RAG Setup (ChromaDB) ---
def init_vector_store(openai_api_key: str):
    """Initializes ChromaDB vector store loaded with Spanish regulatory & UX compliance standards."""
    compliance_docs = [
        Document(
            page_content="Spanish Anti-Money Laundering (SEPBLAC) regulations require customer identity validation via valid DNI/NIE within 24 hours.",
            metadata={"source": "SEPBLAC_Compliance"}
        ),
        Document(
            page_content="Bank of Spain guidelines dictate live chat support must not present misleading fee info and respond within acceptable SLA windows (<15 mins).",
            metadata={"source": "Bank_of_Spain_Guidelines"}
        ),
        Document(
            page_content="Usability benchmark: Onboarding flow friction should not require more than 3 upload retries for OCR document processing.",
            metadata={"source": "Fintech_UX_Standard"}
        )
    ]
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(
        documents=compliance_docs,
        embedding=embeddings,
        collection_name="spain_fintech_cx_rules"
    )
    return vector_store

# --- LangGraph State Definition ---
class CXAuditState(TypedDict):
    mystery_shopper_report: str
    retrieved_compliance_rules: List[str]
    identified_frictions: List[str]
    compliance_gaps: List[str]
    final_evaluation: str

# --- LangGraph Workflow Construction ---
def build_evaluation_graph(openai_api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.2)
    vector_store = init_vector_store(openai_api_key)

    def retrieve_context(state: CXAuditState):
        query = state["mystery_shopper_report"]
        docs = vector_store.similarity_search(query, k=2)
        rules = [doc.page_content for doc in docs]
        return {"retrieved_compliance_rules": rules}

    def analyze_friction_and_compliance(state: CXAuditState):
        report = state["mystery_shopper_report"]
        rules = "\n".join(state["retrieved_compliance_rules"])
        
        prompt = f"""
        Analyze this mystery shopper report for an online financial service in Spain:
        Report: {report}
        
        Relevant Compliance Standards:
        {rules}
        
        Provide:
        1. List of UX Frictions & Bug Bottlenecks encountered.
        2. Compliance or Service Quality Gaps identified.
        """
        response = llm.invoke(prompt)
        content = str(response.content)
        
        frictions = [line for line in content.split("\n") if "Friction" in line or "Bug" in line or line.startswith("-")]
        gaps = [line for line in content.split("\n") if "Compliance" in line or "Gap" in line or "SLA" in line]
        
        return {
            "identified_frictions": frictions,
            "compliance_gaps": gaps,
            "final_evaluation": content
        }

    builder = StateGraph(CXAuditState)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("analyze_friction_and_compliance", analyze_friction_and_compliance)
    
    builder.set_entry_point("retrieve_context")
    builder.add_edge("retrieve_context", "analyze_friction_and_compliance")
    builder.add_edge("analyze_friction_and_compliance", END)
    
    return builder.compile()

def run_cx_audit(report_text: str, openai_api_key: str) -> dict:
    graph = build_evaluation_graph(openai_api_key)
    initial_state = {
        "mystery_shopper_report": report_text,
        "retrieved_compliance_rules": [],
        "identified_frictions": [],
        "compliance_gaps": [],
        "final_evaluation": ""
    }
    return graph.invoke(initial_state)