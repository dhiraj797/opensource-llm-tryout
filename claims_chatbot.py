"""
RBIN Claims Assist Chatbot - Streamlit PoC
============================================
A RAG-based chatbot for RBIN expense claims policy Q&A.
Uses TF-IDF retrieval + Llama 3.2-1B generation (with mock fallback).

Usage:
    streamlit run claims_chatbot.py

Set USE_LLM=true environment variable to use the actual Llama model.
Default mode uses smart template-based responses (no GPU required).
"""

import os
import re
import math
import streamlit as st
from collections import Counter

from policy_data import POLICY_CHUNKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USE_LLM = os.environ.get("USE_LLM", "false").lower() == "true"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
TOP_K_CHUNKS = 4  # Number of policy chunks to retrieve per query


# ---------------------------------------------------------------------------
# TF-IDF Retrieval Engine (no external dependencies)
# ---------------------------------------------------------------------------
class TFIDFRetriever:
    """Lightweight TF-IDF retriever using only Python builtins."""

    def __init__(self, documents):
        self.documents = documents
        self.vocab = set()
        self.doc_tokens = []
        self.idf = {}
        self._build_index()

    @staticmethod
    def _tokenize(text):
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        return re.findall(r"[a-z0-9/]+", text.lower())

    def _build_index(self):
        """Build TF-IDF index over all document chunks."""
        n_docs = len(self.documents)

        # Tokenize each document (combine title + content + keywords)
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
            tokens = self._tokenize(text)
            self.doc_tokens.append(tokens)
            self.vocab.update(tokens)

        # Compute IDF
        doc_freq = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        for term, df in doc_freq.items():
            self.idf[term] = math.log((n_docs + 1) / (df + 1)) + 1

    def _tfidf_vector(self, tokens):
        """Compute TF-IDF vector for a list of tokens."""
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = {}
        for term in set(tokens):
            vector[term] = (tf[term] / total) * self.idf.get(term, 1.0)
        return vector

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        """Compute cosine similarity between two sparse vectors (dicts)."""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query, top_k=TOP_K_CHUNKS):
        """Retrieve top-k most relevant chunks for a query."""
        query_tokens = self._tokenize(query)
        query_vec = self._tfidf_vector(query_tokens)

        scores = []
        for i, doc_tokens in enumerate(self.doc_tokens):
            doc_vec = self._tfidf_vector(doc_tokens)
            score = self._cosine_similarity(query_vec, doc_vec)

            # Boost: extra weight if query tokens appear in keywords
            keyword_tokens = set(self._tokenize(" ".join(self.documents[i]["keywords"])))
            keyword_overlap = len(set(query_tokens) & keyword_tokens)
            score += keyword_overlap * 0.1

            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                results.append({
                    "score": round(score, 4),
                    "title": self.documents[idx]["title"],
                    "section": self.documents[idx]["section"],
                    "content": self.documents[idx]["content"],
                })
        return results


# ---------------------------------------------------------------------------
# LLM Integration (Llama 3.2-1B or mock)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_llm():
    """Load the Llama model (only when USE_LLM=true)."""
    if not USE_LLM:
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.2,
        )
        return pipe
    except Exception as e:
        return None


def generate_answer_llm(pipe, query, context_chunks):
    """Generate answer using the actual Llama model."""
    context_text = "\n\n".join(
        f"[{c['title']}]\n{c['content']}" for c in context_chunks
    )

    prompt = f"""The following is a conversation between an employee and an RBIN Claims Assistant. The assistant answers questions using ONLY the policy information provided below. If the answer is not in the provided context, the assistant says "I don't have enough information in the policy to answer that." The assistant is concise and cites the relevant policy section.

Policy Context:
{context_text}

Employee: {query}
Assistant:"""

    result = pipe(prompt, return_full_text=False)
    return result[0]["generated_text"].strip()


def generate_answer_template(query, context_chunks):
    """Generate a template-based answer from retrieved chunks (no LLM needed)."""
    if not context_chunks:
        return (
            "I couldn't find relevant policy information for your question. "
            "Please try rephrasing, or ask about specific topics like:\n"
            "- NTRE claims and approval limits\n"
            "- Inland/abroad travel rules\n"
            "- Fuel card eligibility\n"
            "- Gratuity guidelines\n"
            "- Settlement timelines"
        )

    # Build a structured answer from the top chunks
    answer_parts = []
    answer_parts.append(f"Based on the **{context_chunks[0]['section']}** policy:\n")

    for i, chunk in enumerate(context_chunks):
        content = chunk["content"]
        # Clean up content for display
        if i == 0:
            answer_parts.append(f"{content}\n")
        else:
            if chunk["section"] != context_chunks[0]["section"]:
                answer_parts.append(f"\n**Also relevant from {chunk['section']}:**\n")
            answer_parts.append(f"{content}\n")

    answer_parts.append(
        f"\n---\n*Source: {', '.join(set(c['section'] for c in context_chunks))}*"
    )

    return "\n".join(answer_parts)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="RBIN Claims Assist",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .policy-chip {
        display: inline-block;
        padding: 2px 10px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.75em;
        background-color: #e8f0fe;
        color: #1a73e8;
    }
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #4caf50, #8bc34a);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Bosch-logo.svg/330px-Bosch-logo.svg.png", width=150)
        st.title("Claims Assist")
        st.caption("Powered by Llama 3.2-1B + RAG")

        st.divider()
        st.subheader("Policy Coverage")
        policies = [
            ("NTRE Reimbursement", "Non-travel expense claims"),
            ("Inland Travel (GL/R/01)", "Domestic travel rules"),
            ("Abroad Travel (GL/R/02)", "International travel policy"),
            ("Fuel Card (GL/P/39)", "Fuel card eligibility & limits"),
            ("Bill Passing (GL/P/22)", "Cheque/DD/NEFT procedures"),
            ("Gratuities", "Public officials & third parties"),
        ]
        for name, desc in policies:
            st.markdown(f"**{name}**  \n{desc}")

        st.divider()
        st.subheader("Settings")
        show_sources = st.toggle("Show retrieved sources", value=True)
        mode = "LLM (Llama 3.2-1B)" if USE_LLM else "Template (No GPU)"
        st.info(f"Mode: **{mode}**")
        if not USE_LLM:
            st.caption("Set `USE_LLM=true` to use the Llama model for natural language answers.")

        st.divider()
        st.caption("PoC Demo v1.0 | RBIN Internal Use Only")

    # --- Initialize retriever ---
    if "retriever" not in st.session_state:
        st.session_state.retriever = TFIDFRetriever(POLICY_CHUNKS)

    # --- Initialize LLM ---
    llm_pipe = None
    if USE_LLM:
        llm_pipe = load_llm()

    # --- Header ---
    st.title("RBIN Claims Assist Chatbot")
    st.markdown(
        "Ask questions about **expense claims, travel policies, fuel cards, approvals, "
        "gratuities**, and more. Your policy assistant is here to help!"
    )

    # --- Quick Action Buttons ---
    st.markdown("**Quick questions:**")
    quick_questions = [
        "What expenses are covered under NTRE?",
        "What is the approval limit for Department Head?",
        "How do I submit an inland travel claim?",
        "Who is eligible for a fuel card?",
        "What are the air travel booking rules for abroad travel?",
        "What are the gratuity rules for public officials?",
    ]

    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_question = q

    st.divider()

    # --- Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm the **RBIN Claims Assistant**. I can help you with questions about:\n\n"
                    "- **NTRE** (Non-Travel Related Expenses)\n"
                    "- **Inland Travel** policy and allowances\n"
                    "- **Abroad Travel** rules and forex\n"
                    "- **Fuel Card** eligibility and limits\n"
                    "- **Bill Passing** procedures\n"
                    "- **Gratuity** guidelines\n\n"
                    "Go ahead and ask me anything!"
                ),
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and show_sources:
                with st.expander("View Policy Sources", expanded=False):
                    for src in message["sources"]:
                        score_pct = min(src["score"] * 100, 100)
                        st.markdown(
                            f"**{src['title']}** — `{src['section']}` "
                            f"(relevance: {score_pct:.0f}%)"
                        )
                        st.progress(score_pct / 100)
                        st.caption(src["content"][:200] + "..." if len(src["content"]) > 200 else src["content"])
                        st.markdown("---")

    # --- Handle pending quick question ---
    if "pending_question" in st.session_state:
        query = st.session_state.pop("pending_question")
        _handle_query(query, llm_pipe, show_sources)
        st.rerun()

    # --- Chat Input ---
    if query := st.chat_input("Ask about claims, travel, fuel cards, approvals..."):
        _handle_query(query, llm_pipe, show_sources)
        st.rerun()


def _handle_query(query, llm_pipe, show_sources):
    """Process a user query: retrieve context, generate answer, update chat."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve relevant policy chunks
    retriever = st.session_state.retriever
    chunks = retriever.retrieve(query, top_k=TOP_K_CHUNKS)

    # Generate answer
    if llm_pipe is not None:
        answer = generate_answer_llm(llm_pipe, query, chunks)
    else:
        answer = generate_answer_template(query, chunks)

    # Build assistant message
    assistant_msg = {"role": "assistant", "content": answer}
    if show_sources and chunks:
        assistant_msg["sources"] = chunks

    st.session_state.messages.append(assistant_msg)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
