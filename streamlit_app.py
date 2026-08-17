import os
import streamlit as st

st.set_page_config(
    page_title="Spain Financial Platform CX Mystery Shopping Auditor",
    page_icon="🏦",
    layout="wide"
)

# Safely extract OpenAI key from Streamlit Secrets or Environment (Without hardcoding secrets files)
def get_api_key() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.environ.get("OPENAI_API_KEY", "")

openai_api_key = get_api_key()

st.title("🏦 Spain Fintech Customer Experience & Compliance Evaluator")
st.markdown("Automated AI evaluation platform for mystery shopping reports across Spanish online financial platforms.")

with st.sidebar:
    st.header("Authentication & Config")
    if not openai_api_key:
        openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
        st.caption("Provide key here or configure via Streamlit Secrets.")
    else:
        st.success("OpenAI API Key detected via Environment/Secrets.")
    
    st.info("Target Region: **Spain (ES)**\nCompliance Standard: **SEPBLAC / Bank of Spain**")

shopper_input = st.text_area(
    label="Mystery Shopper Interaction Log:",
    height=200,
    placeholder="e.g., Intenté abrir una cuenta desde Madrid con mi DNI español. Al subir el documento de identidad, el sistema dio un error genérico tres veces..."
)

if st.button("Analyze Mystery Shopping Log", type="primary"):
    if not openai_api_key:
        st.error("Please enter a valid OpenAI API Key to proceed.")
    elif not shopper_input.strip():
        st.warning("Please paste a mystery shopping interaction log.")
    else:
        with st.spinner("Executing RAG lookup & LangGraph compliance audit..."):
            from model import run_cx_audit
            try:
                audit_result = run_cx_audit(shopper_input, openai_api_key)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Audit Summary & Analysis")
                    st.write(audit_result["final_evaluation"])
                
                with col2:
                    st.subheader("⚖️ Retrieved Compliance Reference Rules")
                    for idx, rule in enumerate(audit_result["retrieved_compliance_rules"], 1):
                        st.info(f"**Rule {idx}:** {rule}")
                        
            except Exception as e:
                st.error(f"Error executing analysis: {e}")