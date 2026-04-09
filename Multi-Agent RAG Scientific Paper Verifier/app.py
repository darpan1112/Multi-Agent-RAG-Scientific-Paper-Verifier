import streamlit as st
import numpy as np
import time
from vector_store import VectorStore
from agents.retrieval_agent import RetrievalAgent
from agents.verification_agent import VerificationAgent
from agents.explanation_agent import ExplanationAgent
from agents.uncertainty_agent import UncertaintyAgent

st.set_page_config(page_title="Multi-Agent RAG Scientific Paper Verifier", page_icon="🔬", layout="wide")

# Advanced Colorful Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-weight: 300;
        margin-bottom: 30px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
        color: white;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #ec4899;
        box-shadow: 0 0 0 1px #ec4899;
    }

    /* Verdict Boxes */
    .verdict-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.8s ease-out;
    }
    .verdict-supported {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        color: #ecfdf5;
        border: 1px solid #10b981;
    }
    .verdict-partial {
        background: linear-gradient(135deg, #78350f 0%, #d97706 100%);
        color: #fffbeb;
        border: 1px solid #f59e0b;
    }
    .verdict-not {
        background: linear-gradient(135deg, #7f1d1d 0%, #e11d48 100%);
        color: #fff1f2;
        border: 1px solid #f43f5e;
    }
    
    .verdict-box h2 {
        margin: 0;
        font-size: 2.2rem;
    }

    /* Metrics Container Customization */
    div[data-testid="metric-container"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 Multi-Agent RAG Scientific Paper Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter a scientific claim to verify it against a database of scientific papers using a multi-agent AI system.</div>', unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    vector_store = VectorStore("data.csv")
    return (
        RetrievalAgent(vector_store),
        VerificationAgent(),
        ExplanationAgent(),
        UncertaintyAgent()
    )

# Loading bar for initial setup
if 'system_initialized' not in st.session_state:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("🚀 **Initializing Vector Spaces...**")
    time.sleep(0.5)
    progress_bar.progress(30)
    
    status_text.markdown("🧠 **Summoning AI Agents...**")
    try:
        retrieval_agent, verification_agent, explanation_agent, uncertainty_agent = initialize_system()
        st.session_state['agents'] = (retrieval_agent, verification_agent, explanation_agent, uncertainty_agent)
        progress_bar.progress(100)
        status_text.markdown("✅ **System Ready!**")
        time.sleep(0.5)
        st.session_state['system_initialized'] = True
        st.rerun()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()
else:
    retrieval_agent, verification_agent, explanation_agent, uncertainty_agent = st.session_state['agents']

st.markdown("<br>", unsafe_allow_html=True)

with st.form(key="verification_form", border=False):
    query = st.text_input("📝 Enter a scientific claim to verify against the knowledge base:", 
                          placeholder="e.g., Deep learning significantly accelerates protein folding prediction.")
    
    submit_button = st.form_submit_button(label="🚀 Verify Knowledge", type="primary")

if submit_button:
    if query.strip() == "":
        st.warning("⚠️ Please enter a valid claim.")
    else:
        with st.spinner("🔮 Agents are investigating cross-references..."):
            # Dummy delay for effect
            time.sleep(1)
            
            docs, embeddings, query_embedding = retrieval_agent.retrieve(query)
            verdict, score, similarities = verification_agent.verify(query_embedding, embeddings)
            variance, uncertainty_level = uncertainty_agent.calculate_uncertainty(similarities)
            
            top_similarity = float(max(similarities))
            
            st.markdown("### 📊 Verification Analysis")
            
            # Display verdict with custom styling
            if verdict == "Supported":
                verdict_class = "verdict-supported"
                icon = "✅"
            elif verdict == "Partially Supported":
                verdict_class = "verdict-partial"
                icon = "〽️"
            else:
                verdict_class = "verdict-not"
                icon = "❌"
                
            st.markdown(f'<div class="verdict-box {verdict_class}"><h2>{icon} {verdict}</h2></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="🎯 Confidence Score", value=f"{score:.2f}", delta="Reliability")
            with col2:
                st.metric(label="🔥 Top Match Similarity", value=f"{top_similarity:.2f}", delta="Context Fit", delta_color="normal")
            with col3:
                st.metric(label="⚖️ Uncertainty Level", value=f"{uncertainty_level}", delta=f"Var {variance:.3f}", delta_color="inverse")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.info("""
            **💡 Understanding the Metrics:**
            *   **🎯 Confidence Score:** Indicates how certain the system is about the final verdict. A higher score means stronger supporting evidence was found.
            *   **🔥 Top Match Similarity:** Represents how closely your claim matches the single most relevant scientific paper in the database (1.0 = exact 100% text match).
            *   **⚖️ Uncertainty Level:** Measures consistency across the top evidence. 'Low Uncertainty' means the retrieved papers agree with each other. 'High Uncertainty' means there are contradictory opinions or diverse claims among the papers.
            """)

            st.markdown("### 📚 Extracted Scientific Evidence")
            for i, doc in enumerate(docs):
                sim = similarities[i]
                with st.expander(f"🔬 Source Document {i+1} — Match: {sim:.2%}"):
                    st.markdown(f"*{doc}*")