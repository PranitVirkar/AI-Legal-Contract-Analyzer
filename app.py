import streamlit as st

from extractor import extract_text_from_pdf
from analyzer import analyze_contract

st.set_page_config(
    page_title="AI Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 0%,rgba(99,102,241,.08),transparent 28%),radial-gradient(circle at 90% 5%,rgba(14,165,233,.07),transparent 25%),#f7f8fc}
.block-container{max-width:1400px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:1.8rem 2rem;border:1px solid rgba(99,102,241,.15);border-radius:24px;background:linear-gradient(135deg,rgba(255,255,255,.97),rgba(245,247,255,.97));box-shadow:0 10px 35px rgba(15,23,42,.06);margin-bottom:1.25rem}
.badge{display:inline-block;padding:.35rem .75rem;border-radius:999px;background:#eef2ff;color:#4f46e5;font-size:.78rem;font-weight:700;letter-spacing:.03em;margin-bottom:.65rem}
.hero h1{font-size:2.25rem;font-weight:800;letter-spacing:-.04em;color:#111827;margin:0}.hero p{color:#64748b;font-size:1rem;margin-top:.55rem;max-width:850px;line-height:1.6}
.card{background:rgba(255,255,255,.96);border:1px solid #e5e7eb;border-radius:18px;padding:1rem 1.1rem;box-shadow:0 5px 20px rgba(15,23,42,.045);height:100%}
.label{color:#64748b;font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em}.value{color:#111827;font-size:1.55rem;font-weight:800;margin-top:.35rem}.detail{color:#64748b;font-size:.82rem;margin-top:.25rem}
.score{background:rgba(255,255,255,.98);border:1px solid #e5e7eb;border-radius:22px;padding:1.35rem;box-shadow:0 8px 28px rgba(15,23,42,.055);height:100%}.score-num{font-size:3rem;font-weight:850;line-height:1;color:#111827}.score-label{color:#64748b;font-size:.84rem;margin-top:.3rem}
.pill{display:inline-block;padding:.42rem .8rem;border-radius:999px;font-size:.82rem;font-weight:800}.low{background:#dcfce7;color:#166534}.medium{background:#fef3c7;color:#92400e}.high{background:#fee2e2;color:#991b1b}
.clause{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:.85rem 1rem;margin:.55rem 0;box-shadow:0 3px 12px rgba(15,23,42,.03);color:#475569;line-height:1.55}
.ai{background:linear-gradient(180deg,#fff,#fafbff);border:1px solid #e0e7ff;border-radius:18px;padding:1rem 1.1rem;margin-bottom:.8rem;box-shadow:0 5px 18px rgba(79,70,229,.055)}
.ai-tag{display:inline-block;background:#eef2ff;color:#4338ca;padding:.3rem .65rem;border-radius:999px;font-size:.78rem;font-weight:800}
.footer{padding:1rem 1.15rem;border-radius:14px;background:rgba(255,255,255,.8);border:1px solid #e5e7eb;color:#64748b;font-size:.82rem;line-height:1.55}
div.stButton>button[kind="primary"]{border-radius:12px;font-weight:750;min-height:3rem}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.9);border:2px dashed #c7d2fe;border-radius:18px;padding:.5rem}
div[data-testid="stExpander"]{border-radius:14px;border-color:#e5e7eb}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="badge">AI-POWERED CONTRACT SCREENING</div>
<h1>⚖️ AI Legal Contract Analyzer</h1>
<p>Analyze contracts for important clauses, key contract information, potential risks, missing clauses, and semantic clause classifications.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📄 Upload a Contract")
st.caption("Supported formats: PDF and TXT • Upload a contract to begin analysis")
uploaded_file = st.file_uploader("Upload Contract", type=["pdf", "txt"], label_visibility="collapsed", help="Upload a PDF or TXT contract for analysis.")

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        if not text.strip():
            st.error("❌ No readable text was found in this document.")
            st.stop()
        st.success(f"✅ Document loaded: **{uploaded_file.name}**")
    except Exception as e:
        st.error(f"❌ Error reading the document: {e}")
        st.stop()

    with st.expander("📄 View Extracted Contract Text"):
        st.text_area("Contract Text", text, height=300, label_visibility="collapsed")

    if st.button("🔍 Analyze Contract", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing contract with the AI pipeline..."):
            try:
                result = analyze_contract(text)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.stop()

        st.success("✅ Contract analysis completed successfully.")

        parties = result.get("parties", [])
        dates = result.get("dates", [])
        amounts = result.get("amounts", [])
        clauses = result.get("clauses", {})
        risks = result.get("risks", [])
        missing_clauses = result.get("missing_clauses", [])
        ml_predictions = result.get("ml_clause_predictions", [])
        detected_clause_count = sum(1 for values in clauses.values() if values)

        st.markdown("### 📊 Contract Overview")
        st.caption("A quick snapshot of the analyzed document.")
        cols = st.columns(5)
        overview = [("👥", "Parties", len(parties)), ("📅", "Dates", len(dates)), ("💰", "Amounts", len(amounts)), ("📑", "Clause Types", detected_clause_count), ("⚠️", "Risk Factors", len(risks))]
        for col, (icon, label, value) in zip(cols, overview):
            with col:
                st.markdown(f'<div class="card"><div class="label">{icon} {label}</div><div class="value">{value}</div><div class="detail">Detected / identified</div></div>', unsafe_allow_html=True)

        st.markdown("### 📋 Contract Summary")
        st.info(result.get("summary", "No summary available."))

        st.markdown("### 📊 Contract Risk Assessment")
        risk_score = result.get("risk_score", {})
        score = risk_score.get("score", 0)
        category = risk_score.get("category", "Low")
        pill = "high" if category == "High" else "medium" if category == "Medium" else "low"
        icon = "🔴" if category == "High" else "🟡" if category == "Medium" else "🟢"
        c1, c2, c3 = st.columns([1.15, 1, 1])
        with c1:
            st.markdown(f'<div class="score"><div class="score-label">OVERALL RISK SCORE</div><div class="score-num">{score}<span style="font-size:1.1rem;color:#94a3b8"> / 100</span></div><div style="margin-top:.65rem"><span class="pill {pill}">{icon} {category} Risk</span></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="score"><div class="score-label">RISK FACTORS</div><div class="score-num">{len(risks)}</div><div class="score-label">Potential factors identified</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="score"><div class="score-label">DETECTED CLAUSES</div><div class="score-num">{detected_clause_count}</div><div class="score-label">Contract clause categories</div></div>', unsafe_allow_html=True)
        st.progress(min(max(score / 100, 0.0), 1.0))

        with st.expander("📋 View Risk Score Breakdown"):
            breakdown = risk_score.get("breakdown", [])
            if breakdown:
                for item in breakdown:
                    risk_type = item.get("type", "Unknown")
                    points = item.get("points", 0)
                    severity = item.get("severity", "Unknown")
                    msg = f"**{risk_type}** · +{points} points · {severity}"
                    st.error("🔴 " + msg) if severity == "High" else st.warning("🟡 " + msg) if severity == "Medium" else st.info("🟢 " + msg)
            else:
                st.success("No risk factors contributed to the score.")

        st.markdown("### 📑 Contract Information")
        info = st.columns(3)
        blocks = [("👥 Parties", parties, "No parties detected."), ("📅 Important Dates", dates, "No dates detected."), ("💰 Monetary Values", amounts, "No monetary values detected.")]
        for col, (title, values, empty) in zip(info, blocks):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    if values:
                        for value in values:
                            st.markdown(f"• {value}")
                    else:
                        st.caption(empty)

        tab_clauses, tab_risks, tab_ai, tab_document = st.tabs(["📑 Clauses", "⚠️ Risks & Missing", "🤖 AI Analysis", "📄 Document"])

        with tab_clauses:
            st.markdown("### 📑 Important Contract Clauses")
            for key, title in [("payment", "💰 Payment Terms"), ("termination", "🚪 Termination")]:
                sentences = clauses.get(key, [])
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    if sentences:
                        for sentence in sentences:
                            st.markdown(f'<div class="clause">{sentence}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("No clause detected.")

            st.markdown("### 🔎 All Detected Clauses")
            found = False
            for clause_name, sentences in clauses.items():
                if sentences:
                    found = True
                    display = clause_name.replace("_", " ").title()
                    with st.expander(f"📌 {display} · {len(sentences)}"):
                        for sentence in sentences:
                            st.markdown(f'<div class="clause">{sentence}</div>', unsafe_allow_html=True)
            if not found:
                st.info("No predefined clauses were detected.")

        with tab_risks:
            st.markdown("### ⚠️ Potential Risks")
            if risks:
                for risk in risks:
                    risk_type = risk.get("type", "Unknown Risk")
                    severity = risk.get("severity", "Unknown")
                    reason = risk.get("reason", "No explanation available.")
                    source_clauses = risk.get("clauses", [])
                    icon = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                    with st.container(border=True):
                        st.markdown(f"#### {icon} {risk_type}")
                        st.caption(f"Severity: **{severity}**")
                        st.write(reason)
                        if source_clauses:
                            with st.expander("View related contract text"):
                                for clause in source_clauses:
                                    st.markdown(f'<div class="clause">{clause}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No predefined contractual risks detected.")
            st.markdown("### 🔎 Potentially Missing Clauses")
            if missing_clauses:
                for clause in missing_clauses:
                    st.warning(f"⚠️ {clause.replace('_', ' ').title()}")
            else:
                st.success("✅ No predefined important clauses appear to be missing.")

        with tab_ai:
            st.markdown("### 🤖 Legal-BERT Clause Classification")
            st.caption("Semantic clause classifications generated using Legal-BERT embeddings and the trained classifier.")
            if ml_predictions:
                for i, prediction in enumerate(ml_predictions, 1):
                    sentence = prediction.get("sentence", "")
                    clause = prediction.get("clause", "Unknown")
                    confidence = prediction.get("confidence", 0.0)
                    with st.container(border=True):
                        st.markdown(f'<span class="ai-tag">AI CLASSIFICATION #{i}</span>', unsafe_allow_html=True)
                        st.markdown(f"### {clause.replace('_', ' ').title()}")
                        st.write(sentence)
                        st.progress(min(max(confidence, 0.0), 1.0))
                        st.caption(f"Model confidence: **{confidence:.2%}**")
            else:
                st.info("No Legal-BERT clause predictions are available.")

        with tab_document:
            st.markdown("### 📄 Extracted Contract Text")
            st.caption("Text extracted from the uploaded document and supplied to the analyzer.")
            st.text_area("Extracted text", text, height=500, label_visibility="collapsed")
            st.download_button("⬇️ Download Extracted Text", data=text, file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.txt", mime="text/plain", use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="footer">⚖️ <strong>Disclaimer:</strong> This tool provides automated contract screening and does not constitute legal advice. Detected risks, classifications, and potentially missing clauses should be reviewed by a qualified legal professional.</div>', unsafe_allow_html=True)
