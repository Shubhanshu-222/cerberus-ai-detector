import streamlit as st
import pandas as pd
import spacy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import re
import time
import joblib
import os
import warnings

import subprocess
import sys

try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Text Robustness Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
warnings.filterwarnings('ignore') # Hides sklearn/joblib warnings

# --- Global Configs ---
MODEL_DIR = "production_models"
SIMPLE_FEATURE_COLS = ['word_count', 'char_count', 'avg_word_length', 'punctuation_density']
EMBEDDING_BATCH_SIZE = 16

# ---
# --- START: HELPER FUNCTIONS COPIED FROM train_models.py ---
# --- These MUST be in the global scope for joblib.load() to find them ---
# ---
@st.cache_resource
def load_spacy_model():
    # We keep this cached to load spacy only once
    try:
        nlp = spacy.load("en_core_web_sm")
    except IOError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run: `python -m spacy download en_core_web_sm`")
        st.stop()
    return nlp

def get_pos_tags(text_series, nlp):
    def get_tags(text):
        if not isinstance(text, str): return ""
        doc = nlp(text.lower())
        tags = [token.pos_ for token in doc if not token.is_punct and not token.is_space]
        return " ".join(tags)
    return text_series.apply(get_tags)

def get_text_as_is(text_series):
    return text_series

FUNCTION_WORDS = [
    'a', 'about', 'above', 'after', 'all', 'an', 'and', 'any', 'are', 'around', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could', 'did', 'do', 'does', 'down', 'during', 'each', 'either', 'enough', 'every', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'least', 'less', 'like', 'many', 'me', 'might', 'more', 'most', 'much', 'must', 'my', 'myself', 'near', 'neither', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'one', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'shall', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'us', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
]

def create_linguistic_pipeline(nlp):
    # This function definition is also needed by joblib
    pos_pipeline = Pipeline([
        ('get_text', FunctionTransformer(get_text_as_is, validate=False)),
        ('get_pos', FunctionTransformer(get_pos_tags, validate=False, kw_args={'nlp': nlp})),
        ('vectorize_pos', TfidfVectorizer(ngram_range=(1, 3), lowercase=False, token_pattern=r"\b\w+\b"))
    ])
    func_word_pipeline = Pipeline([
        ('get_text', FunctionTransformer(get_text_as_is, validate=False)),
        ('vectorize_func', TfidfVectorizer(vocabulary=FUNCTION_WORDS, lowercase=True, use_idf=False, norm='l1'))
    ])
    return FeatureUnion([
        ('pos_features', pos_pipeline),
        ('func_word_features', func_word_pipeline)
    ])
# ---
# --- END: HELPER FUNCTIONS COPIED FROM train_models.py ---
# ---


# --- Model Loading (Cached) ---
@st.cache_resource
def load_production_models():
    print("--- Loading PRODUCTION models from disk... ---")
    
    # 1. Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory '{MODEL_DIR}' not found! Please run `python train_models.py` first.")
        st.stop()
        
    # 2. Load Base Models (spaCy, RoBERTa)
    # Use AutoModel/AutoTokenizer for broader compatibility across transformers versions
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    embed_model = AutoModel.from_pretrained('roberta-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Streamlit: Using device: {device} ---")
    embed_model.to(device)
    embed_model.eval()
    
    # 3. Load our 5 saved model files
    try:
        model_stats = joblib.load(os.path.join(MODEL_DIR, "model_4_stats_new.joblib"))
        scaler_stats = joblib.load(os.path.join(MODEL_DIR, "scaler_4_stats_new.joblib"))
        model_ling = joblib.load(os.path.join(MODEL_DIR, "model_5_ling_new.joblib"))
        pipeline_ling = joblib.load(os.path.join(MODEL_DIR, "pipeline_5_ling_new.joblib"))
        model_embed = joblib.load(os.path.join(MODEL_DIR, "model_6_embed_new.joblib"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    except FileNotFoundError as e:
        st.error(f"Error loading model file: {e}. Please run `python train_models.py` to create the models.")
        st.stop()
        
    print("--- All production models loaded successfully. ---")
    
    return {
        "tokenizer": tokenizer, "embed_model": embed_model, "device": device,
        "scaler": scaler_stats, "model_stats": model_stats,
        "ling_pipeline": pipeline_ling, "model_ling": model_ling,
        "model_embed": model_embed, "class_names": label_encoder.classes_
    }

# --- Feature Engineering Helpers ---
def get_roberta_embeddings(text_series, _tokenizer, _embed_model, _device):
    all_embeddings = []
    text_list = text_series.astype(str).tolist()
    with torch.no_grad():
        for i in range(0, len(text_list), EMBEDDING_BATCH_SIZE):
            batch = text_list[i:i + EMBEDDING_BATCH_SIZE]
            inputs = _tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(_device)
            outputs = _embed_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings)

def calculate_stats_for_input(text):
    text = str(text)
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    if word_count == 0:
        avg_word_length = 0
        punctuation_density = 0
    else:
        avg_word_length = char_count / word_count
        punctuation_count = len(re.findall(r'[.,!?;:]', text))
        punctuation_density = punctuation_count / word_count
    return pd.DataFrame(
        [[word_count, char_count, avg_word_length, punctuation_density]],
        columns=SIMPLE_FEATURE_COLS
    )

# --- NEW: Helper function to convert DataFrame to CSV ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main App UI ---
st.title("🔬 AI Text Robustness Analyzer")
st.caption("Based on the RPT-03 research benchmark comparing Statistical, Linguistic, and Embedding-based models.")

st.info(
    "**How This Works:** This app uses models trained on our full research dataset. It runs your text "
    "through three distinct models to provide a hybrid, diagnostic verdict.",
    icon="ℹ️"
)

# Load all models (this is cached)
with st.spinner("Loading production models from disk..."):
    # This also loads spacy, which is cached
    load_spacy_model()
    models_bundle = load_production_models()

st.divider()

col1, col2 = st.columns([0.4, 0.6], gap="large")

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Paste text here... (30-100 words recommended)",
        height=280,
        key="input_text",
        placeholder="The new local coffee shop has a cozy atmosphere..."
    )
    analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    
    if "results" not in st.session_state:
        st.session_state.results = None

    if analyze_button:
        if not input_text or len(input_text.split()) < 10:
            st.warning("Please enter text with at least 10 words.", icon="⚠️")
        else:
            with st.spinner("Running diagnostics on all 3 models..."):
                # 1. Prepare input data
                input_text_series = pd.Series([input_text])
                
                # Stats
                input_stats_df = calculate_stats_for_input(input_text)
                input_stats_scaled = models_bundle["scaler"].transform(input_stats_df)
                
                # Linguistics
                # The loaded pipeline_ling contains all steps (get_text, get_pos, vectorize)
                input_ling = models_bundle["ling_pipeline"].transform(input_text_series)
                
                # Embeddings
                input_embed = get_roberta_embeddings(
                    input_text_series, 
                    models_bundle["tokenizer"], 
                    models_bundle["embed_model"], 
                    models_bundle["device"]
                )
                
                # 2. Get predictions (probabilities)
                prob_stats = models_bundle["model_stats"].predict_proba(input_stats_scaled)[0]
                prob_ling = models_bundle["model_ling"].predict_proba(input_ling)[0]
                prob_embed = models_bundle["model_embed"].predict_proba(input_embed)[0]
                
                # class_names is [AIGT, HGT], so prob[0] is for AIGT, prob[1] is for HGT
                prob_ai_stats = prob_stats[0]
                prob_ai_ling = prob_ling[0]
                prob_ai_embed = prob_embed[0]
                
                # 3. Final Ensemble Verdict
                final_ai_prob = (prob_ai_stats + prob_ai_ling + prob_ai_embed) / 3
                
                if final_ai_prob > 0.6:
                    verdict = "Highly Likely AI-Generated"
                    verdict_icon = "🤖"
                    verdict_color = "error"
                elif final_ai_prob > 0.4:
                    verdict = "Likely AI-Human Hybrid / Unclear"
                    verdict_icon = "🧑‍💻"
                    verdict_color = "warning"
                else:
                    verdict = "Highly Likely Human-Written"
                    verdict_icon = "🧑"
                    verdict_color = "success"
                
                # 4. Create Copyable Report
                report_content = f"""
--- AI Text Analysis Report ---
Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
Final Verdict: {verdict}
Ensemble AI Confidence: {final_ai_prob*100:.2f}%

--- Model Breakdown ---
1. Statistical Model (Stats):
   - Verdict: {'AI' if prob_ai_stats > 0.5 else 'Human'}
   - AI Confidence: {prob_ai_stats*100:.2f}%

2. Linguistic Model (Ling):
   - Verdict: {'AI' if prob_ai_ling > 0.5 else 'Human'}
   - AI Confidence: {prob_ai_ling*100:.2f}%

3. Embedding Model (Embed):
   - Verdict: {'AI' if prob_ai_embed > 0.5 else 'Human'}
   - AI Confidence: {prob_ai_embed*100:.2f}%

--- Input Text Features ---
Word Count: {input_stats_df['word_count'].values[0]}
Character Count: {input_stats_df['char_count'].values[0]}
Avg. Word Length: {input_stats_df['avg_word_length'].values[0]:.2f}
Punctuation Density: {input_stats_df['punctuation_density'].values[0]:.2f}

--- Raw Input Text ---
{input_text}
"""
                # --- Create Chart Data ---
                chart_data = pd.DataFrame({
                    "Model": ["Statistical", "Linguistic", "Embedding"],
                    "AI Confidence": [prob_ai_stats, prob_ai_ling, prob_ai_embed]
                })
                chart_data["AI Confidence"] = (chart_data["AI Confidence"] * 100).round(1)

                # --- Create DataFrame for Download ---
                df_data = {
                    'Final Verdict': [verdict],
                    'Ensemble AI Confidence (%)': [round(final_ai_prob * 100, 2)],
                    'Stats AI Confidence (%)': [round(prob_ai_stats * 100, 2)],
                    'Linguistic AI Confidence (%)': [round(prob_ai_ling * 100, 2)],
                    'Embedding AI Confidence (%)': [round(prob_ai_embed * 100, 2)],
                    'Word Count': [input_stats_df['word_count'].values[0]],
                    'Char Count': [input_stats_df['char_count'].values[0]],
                    'Avg Word Length': [round(input_stats_df['avg_word_length'].values[0], 2)],
                    'Punct Density': [round(input_stats_df['punctuation_density'].values[0], 2)],
                    'Analyzed Text': [input_text]
                }
                results_df = pd.DataFrame(df_data)
                csv_data = convert_df_to_csv(results_df)
                
                # 5. Save results to session state
                st.session_state.results = {
                    "stats_df": input_stats_df,
                    "prob_ai_stats": prob_ai_stats,
                    "prob_ai_ling": prob_ai_ling,
                    "prob_ai_embed": prob_ai_embed,
                    "final_ai_prob": final_ai_prob,
                    "verdict": verdict,
                    "verdict_icon": verdict_icon,
                    "verdict_color": verdict_color,
                    "report_content": report_content,
                    "chart_data": chart_data,
                    "results_df": results_df,
                    "csv_data": csv_data
                }
    
    # Display results if they exist in session state
    if st.session_state.results:
        res = st.session_state.results
        
        # --- Final Verdict ---
        # streamlit may not have `st.alert` in all versions; map to standard message functions
        verdict_text = f"{res.get('verdict_icon','')}  **{res['verdict']}**"
        vc = res.get('verdict_color')
        if vc == 'error':
            st.error(verdict_text)
        elif vc == 'warning':
            st.warning(verdict_text)
        elif vc == 'success':
            st.success(verdict_text)
        else:
            st.info(verdict_text)

        # Show ensemble progress and explicit percentage
        st.progress(res['final_ai_prob'])
        st.write(f"**Ensemble AI Confidence:** `{res['final_ai_prob']*100:.2f}%`")
        
        st.divider()

        # --- Model Breakdown ---
        st.markdown("**Model Verdicts**")
        
        # --- NEW: Polished UI with Cards ---
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            with st.container(border=True):
                st.metric(label="📊 Statistical Model", value=f"{res['prob_ai_stats']*100:.1f}% AI")
                st.progress(res['prob_ai_stats'])
                if res['prob_ai_stats'] > 0.5:
                    st.markdown("**Verdict: <span style='color: #FF4B4B;'>Detected AI</span>**", unsafe_allow_html=True)
                else:
                    st.markdown("**Verdict: <span style='color: #33D69F;'>Predicted Human</span>**", unsafe_allow_html=True)
        
        with m_col2:
            with st.container(border=True):
                st.metric(label="🔬 Linguistic Model", value=f"{res['prob_ai_ling']*100:.1f}% AI")
                st.progress(res['prob_ai_ling'])
                if res['prob_ai_ling'] > 0.5:
                    st.markdown("**Verdict: <span style='color: #FF4B4B;'>Detected AI</span>**", unsafe_allow_html=True)
                else:
                    st.markdown("**Verdict: <span style='color: #33D69F;'>Predicted Human</span>**", unsafe_allow_html=True)

        with m_col3:
            with st.container(border=True):
                st.metric(label="🧠 Embedding Model", value=f"{res['prob_ai_embed']*100:.1f}% AI")
                st.progress(res['prob_ai_embed'])
                if res['prob_ai_embed'] > 0.5:
                    st.markdown("**Verdict: <span style='color: #FF4B4B;'>Detected AI</span>**", unsafe_allow_html=True)
                else:
                    st.markdown("**Verdict: <span style='color: #33D69F;'>Predicted Human</span>**", unsafe_allow_html=True)
        
        # --- NEW: Bar Chart ---
        st.markdown("", help="This chart visualizes the 'AI Confidence' from each model for easy comparison.")
        st.bar_chart(res['chart_data'], x="Model", y="AI Confidence", height=250)
        
        st.divider()

        # --- Input Text Analysis ---
        st.markdown("**Input Text Analysis**")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        stats_df = res['stats_df']
        s_col1.metric("Word Count", stats_df['word_count'].values[0])
        s_col2.metric("Char Count", stats_df['char_count'].values[0])
        s_col3.metric("Avg. Word Len", f"{stats_df['avg_word_length'].values[0]:.2f}")
        s_col4.metric("Punct. Density", f"{stats_df['punctuation_density'].values[0]:.2f}")

    else:
        st.info("Results will appear here after analysis.", icon="📊")

# --- Bottom Section: Copyable & Downloadable Reports ---
if st.session_state.results:
    st.divider()
    st.subheader("Analysis Reports")
    
    tab1, tab2 = st.tabs(["📋 Data for Excel", "📄 Raw Text Report"])

    with tab1:
        st.markdown("**Data for Excel/Sheets**")
        st.write("Use the icon in the top-right of the table to copy this data.")
        st.dataframe(st.session_state.results['results_df'])
        st.download_button(
            label="Download Results as CSV",
            data=st.session_state.results['csv_data'],
            file_name=f"ai_text_analysis_{int(time.time())}.csv",
            mime='text/csv',
        )

    with tab2:
        st.markdown("**Copyable Text Report**")
        st.text_area("Report", value=st.session_state.results['report_content'], height=250, disabled=True, key="report_text_area")
