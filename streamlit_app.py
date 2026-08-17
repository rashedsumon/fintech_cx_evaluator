import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Spain Financial Platform CX Mystery Shopping Auditor",
    page_icon="🏦",
    layout="wide"
)

def get_env_or_secret(key_name: str, default: str = "") -> str:
    """Safely retrieves a configuration key across local, Cloud Run, and Streamlit Cloud environments."""
    env_val = os.environ.get(key_name, "")
    if env_val:
        return env_val
    
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
        
    return default

openai_api_key = get_env_or_secret("OPENAI_API_KEY")
n8n_webhook_url = get_env_or_secret("N8N_WEBHOOK_URL")

st.title("🏦 Spain Fintech Customer Experience & Compliance Evaluator")
st.markdown("Automated AI evaluation platform for mystery shopping reports across Spanish online financial platforms.")

with st.sidebar:
    st.header("Authentication & Config")
    
    # Engine Selection
    eval_engine = st.radio(
        "Evaluation Engine:",
        ["n8n Webhook Orchestrator", "Direct Python (RAG + LangGraph)"],
        index=0 if n8n_webhook_url else 1
    )
    
    st.divider()
    
    if eval_engine == "n8n Webhook Orchestrator":
        if not n8n_webhook_url:
            n8n_webhook_url = st.text_input("Enter n8n Webhook URL:", type="default", placeholder="https://n8n.example.com/webhook/eval-spain-cx")
            st.caption("Provide webhook URL here or set `N8N_WEBHOOK_URL` in environment/secrets.")
        else:
            st.success("n8n Webhook URL detected.")
    else:
        if not openai_api_key:
            openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
            st.caption("Provide key here or configure via Streamlit Secrets / Environment Variables.")
        else:
            st.success("OpenAI API Key detected.")
    
    st.info("Target Region: **Spain (ES)**\nCompliance Standard: **SEPBLAC / Bank of Spain**")

# Form Inputs
col_platform, col_shopper = st.columns(2)
with col_platform:
    platform_name = st.text_input("Financial Platform Name:", value="BancaDigital Spain")
with col_shopper:
    shopper_id = st.text_input("Shopper ID:", value="SHOPPER_ES_042")

shopper_input = st.text_area(
    label="Mystery Shopper Interaction Log:",
    height=200,
    placeholder="e.g., Intenté abrir una cuenta desde Madrid con mi DNI español. Al subir el documento de identidad, el sistema dio un error genérico tres veces..."
)

if st.button("Analyze Mystery Shopping Log", type="primary"):
    if not shopper_input.strip():
        st.warning("Please paste a mystery shopping interaction log.")
    elif eval_engine == "n8n Webhook Orchestrator" and not n8n_webhook_url:
        st.error("Please provide a valid n8n Webhook URL.")
    elif eval_engine == "Direct Python (RAG + LangGraph)" and not openai_api_key:
        st.error("Please enter a valid OpenAI API Key.")
    else:
        if eval_engine == "n8n Webhook Orchestrator":
            with st.spinner("Dispatching evaluation task to n8n workflow pipeline..."):
                try:
                    payload = {
                        "shopper_id": shopper_id,
                        "platform_name": platform_name,
                        "report_text": shopper_input
                    }
                    response = requests.post(n8n_webhook_url, json=payload, timeout=45)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Audit completed successfully! Record ID: {res_data.get('record_id', 'N/A')}")
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader("📋 Audit Summary & Analysis (via n8n)")
                            st.markdown(res_data.get("audit_summary", "No summary returned."))
                        with col2:
                            st.subheader("ℹ️ Execution Metadata")
                            st.json({
                                "status": res_data.get("status"),
                                "platform": res_data.get("platform"),
                                "record_id": res_data.get("record_id")
                            })
                    else:
                        st.error(f"n8n Webhook returned error HTTP {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with n8n Webhook: {e}")
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
                    st.error(f"Error executing local Python analysis: {e}")