"""
app.py
------
InterviewMate AI — main Streamlit application.

Run with:
    streamlit run app.py
"""

import uuid
import streamlit as st
from dotenv import load_dotenv

import rag_engine
import granite_service
import guardrails
import interview_history
from guardrails import GuardrailViolation

# Load .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InterviewMate AI",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── layout ── */
    .block-container { padding-top: 1.8rem; max-width: 800px; }

    /* ── score badge ── */
    .score-pill {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 1.3rem;
        color: #ffffff;
    }
    .score-high { background: #198038; }
    .score-mid  { background: #f1c21b; color: #1c1c1c !important; }
    .score-low  { background: #da1e28; }

    /* ── question card ── */
    .q-card {
        background: #f4f4f4;
        border-left: 4px solid #0f62fe;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        font-size: 1rem;
        line-height: 1.55;
    }

    /* ── tip card ── */
    .tip-card {
        background: #edf5ff;
        border-left: 4px solid #4589ff;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.92rem;
    }

    /* ── confidence banner ── */
    .conf-card {
        background: #defbe6;
        border-left: 4px solid #198038;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.93rem;
    }

    /* ── prep strategy card ── */
    .prep-card {
        background: #f6f2ff;
        border-left: 4px solid #7c5cd8;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "questions":        [],
        "current_q_idx":    0,
        "answers":          {},
        "feedbacks":        {},
        "rag_context":      {},
        "profile":          {},
        "prep_strategy":    {},
        "stage":            "profile",  # profile | prep | interview | report
        "session_id":       str(uuid.uuid4()),   # unique per browser session
        "guardrail_warn":   "",                  # last guardrail message for display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _score_badge(score: int) -> str:
    css = "score-high" if score >= 7 else ("score-mid" if score >= 4 else "score-low")
    return f'<span class="score-pill {css}">{score}/10</span>'

def _reset():
    # Clear history for this session before resetting
    sid = st.session_state.get("session_id")
    if sid:
        interview_history.clear_session(sid)
    for key in ["questions", "current_q_idx", "answers",
                "feedbacks", "rag_context", "profile", "prep_strategy",
                "stage", "session_id", "guardrail_warn"]:
        if key in st.session_state:
            del st.session_state[key]
    _init_state()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎯 InterviewMate AI")
st.caption(
    "Personalised AI-powered interview practice — "
    "powered by **IBM Granite** on IBM watsonx.ai"
)
st.divider()

# ===========================================================================
# STAGE 1 — PROFILE + INTERVIEW TYPE
# ===========================================================================
if st.session_state.stage == "profile":

    st.subheader("📋 Your Profile")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Your Name", placeholder="e.g. Arjun Sharma")
        exp  = st.selectbox(
            "Experience Level",
            ["Beginner (0–1 yr)", "Intermediate (2–4 yrs)", "Senior (5+ yrs)"],
        )
    with col2:
        role   = st.text_input("Target Job Role",
                               placeholder="e.g. Python Developer, Java Engineer, DevOps")
        skills = st.text_input(
            "Key Skills (comma-separated)",
            placeholder="e.g. Python, SQL, Machine Learning",
        )

    st.subheader("🗂️ Interview Type")
    itype = st.radio(
        "Choose the interview type:",
        ["Technical Interview", "HR Interview", "Behavioral Interview", "Soft Skills Assessment"],
        horizontal=True,
    )

    num_q = st.slider("Number of questions", min_value=3, max_value=8, value=5)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        start_btn = st.button("🚀 Start Interview", type="primary", use_container_width=True)
    with col_b:
        prep_btn = st.button("📚 Get Preparation Strategy First", use_container_width=True)

    if start_btn or prep_btn:
        if not name.strip() or not role.strip() or not skills.strip():
            st.warning("⚠️ Please fill in your Name, Job Role, and Skills before continuing.")
            st.stop()

        # ── Guardrail: validate profile (Pydantic) ──
        try:
            guardrails.validate_profile(
                name=name,
                job_role=role,
                experience_level=exp,
                skills=skills,
                interview_type=itype,
                num_questions=num_q,
            )
        except GuardrailViolation as gv:
            st.error(f"⚠️ {gv}")
            st.stop()

        # ── Guardrail: scope check on job role and skills ──
        try:
            guardrails.check_scope(role + " " + skills)
        except GuardrailViolation as gv:
            st.warning(
                f"🤖 {gv}\n\n"
                "InterviewMate AI only supports interview preparation for "
                "technical and HR roles."
            )
            st.stop()

        # ── RAG retrieval ──
        with st.spinner("Retrieving relevant knowledge base context…"):
            ctx = rag_engine.retrieve_context(
                interview_type=itype,
                job_role=role,
                experience_level=exp,
                skills=skills,
            )

        st.session_state.rag_context = ctx
        st.session_state.profile = {
            "name": name, "role": role, "exp": exp,
            "skills": skills, "itype": itype, "num_q": num_q,
        }

        # ── Register session in history ──
        interview_history.create_session(
            session_id=st.session_state.session_id,
            candidate_name=name,
            job_role=role,
            interview_type=itype,
            experience_level=exp,
        )

        if prep_btn:
            # ── Generate preparation strategy ──
            with st.spinner("IBM Granite is building your personalised preparation strategy…"):
                try:
                    strategy = granite_service.get_preparation_strategy(
                        job_role=role,
                        experience_level=exp,
                        skills=skills,
                        interview_type=itype,
                        rag_context=ctx,
                    )
                except EnvironmentError as e:
                    st.error(f"🔑 Credential error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ IBM Granite error: {e}")
                    st.stop()
            st.session_state.prep_strategy = strategy
            st.session_state.stage = "prep"
            st.rerun()

        else:
            # ── Retrieve history for dedup ──
            prev_asked = interview_history.get_asked_questions(
                st.session_state.session_id
            )

            # ── Generate questions ──
            with st.spinner("IBM Granite is generating your personalised questions…"):
                try:
                    questions = granite_service.generate_questions(
                        candidate_name=name,
                        job_role=role,
                        experience_level=exp,
                        skills=skills,
                        interview_type=itype,
                        rag_context=ctx,
                        num_questions=num_q,
                        previously_asked=prev_asked,
                    )
                except EnvironmentError as e:
                    st.error(f"🔑 Credential error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ IBM Granite error: {e}")
                    st.stop()

            # ── Dedup check ──
            new_qs, dupe_qs = interview_history.filter_new_questions(
                st.session_state.session_id, questions
            )
            if dupe_qs:
                st.session_state.guardrail_warn = (
                    f"⚠️ {len(dupe_qs)} question(s) were similar to previously "
                    "asked questions and were filtered. Some repetition may occur "
                    "if the knowledge base is exhausted."
                )

            # Record new questions into history
            for q in new_qs:
                interview_history.record_question(st.session_state.session_id, q)

            st.session_state.questions     = questions   # show all (dedup is advisory)
            st.session_state.current_q_idx = 0
            st.session_state.stage         = "interview"
            st.rerun()

# ===========================================================================
# STAGE 1b — PREPARATION STRATEGY  (shown before interview starts)
# ===========================================================================
elif st.session_state.stage == "prep":

    p        = st.session_state.profile
    strategy = st.session_state.prep_strategy
    ctx      = st.session_state.rag_context

    with st.sidebar:
        st.markdown(f"**Candidate:** {p['name']}")
        st.markdown(f"**Role:** {p['role']}")
        st.markdown(f"**Level:** {p['exp']}")
        st.markdown(f"**Type:** {p['itype']}")
        st.divider()
        if st.button("🔄 Start Over"):
            _reset()
            st.rerun()

    st.subheader("📚 Your Personalised Preparation Strategy")
    st.caption(
        f"Tailored for **{p['name']}** — {p['role']} · {p['exp']} · {p['itype']}"
    )
    st.divider()

    # ── KB tips (from knowledge base — always available) ──
    kb_tips = ctx.get("preparation_tips", [])
    if kb_tips:
        st.markdown("#### 📖 Knowledge Base Tips")
        for tip in kb_tips:
            st.markdown(
                f'<div class="tip-card">💡 {tip}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

    # ── Granite-generated strategy ──
    if strategy:
        st.markdown("#### 🤖 IBM Granite — Personalised Strategy")

        sections = [
            ("📅 Study Plan",        "study_plan"),
            ("🎯 Key Areas to Focus","key_areas"),
            ("🗣️ Practice Advice",   "practice_advice"),
            ("☀️ Day-of Tips",       "day_of_tips"),
            ("🔗 Recommended Resources", "resources"),
        ]
        for label, key in sections:
            content = strategy.get(key, "N/A")
            if content and content != "N/A":
                with st.expander(label, expanded=True):
                    st.write(content)

    st.divider()

    # ── Topics covered ──
    topics = ctx.get("topics", [])
    if topics:
        st.markdown("#### 📚 Topics Covered in This Interview Type")
        tcols = st.columns(3)
        for i, t in enumerate(topics):
            tcols[i % 3].markdown(f"- {t}")
        st.markdown("")

    st.divider()

    if st.button("🚀 Start the Interview Now", type="primary", use_container_width=True):
        with st.spinner("IBM Granite is generating your personalised questions…"):
            try:
                questions = granite_service.generate_questions(
                    candidate_name=p["name"],
                    job_role=p["role"],
                    experience_level=p["exp"],
                    skills=p["skills"],
                    interview_type=p["itype"],
                    rag_context=ctx,
                    num_questions=p["num_q"],
                )
            except EnvironmentError as e:
                st.error(f"🔑 Credential error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ IBM Granite error: {e}")
                st.stop()

        st.session_state.questions     = questions
        st.session_state.current_q_idx = 0
        st.session_state.stage         = "interview"
        st.rerun()

# ===========================================================================
# STAGE 2 — INTERVIEW  (answer questions one by one)
# ===========================================================================
elif st.session_state.stage == "interview":

    p          = st.session_state.profile
    questions  = st.session_state.questions
    idx        = st.session_state.current_q_idx
    total      = len(questions)

    # ── sidebar progress ──
    with st.sidebar:
        st.markdown("### 📊 Progress")
        answered = len(st.session_state.feedbacks)
        st.progress(answered / total, text=f"{answered}/{total} answered")
        st.markdown(f"**Candidate:** {p['name']}")
        st.markdown(f"**Role:** {p['role']}")
        st.markdown(f"**Level:** {p['exp']}")
        st.markdown(f"**Type:** {p['itype']}")
        st.divider()
        if st.session_state.prep_strategy:
            if st.button("📚 View Prep Strategy"):
                st.session_state.stage = "prep"
                st.rerun()
        if st.button("🔄 Start Over"):
            _reset()
            st.rerun()

    # ── question header ──
    st.subheader(f"💬 Question {idx + 1} of {total}")
    st.markdown(
        f'<div class="q-card">💡 {questions[idx]}</div>',
        unsafe_allow_html=True,
    )

    # ── show a preparation tip if available ──
    kb_tips = st.session_state.rag_context.get("preparation_tips", [])
    if kb_tips:
        tip_idx = idx % len(kb_tips)
        st.markdown(
            f'<div class="tip-card">📌 <strong>Prep tip:</strong> {kb_tips[tip_idx]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── answer area ──
    prior_answer = st.session_state.answers.get(idx, "")
    answer = st.text_area(
        "Your Answer",
        value=prior_answer,
        placeholder="Type your answer here…",
        height=185,
        key=f"answer_{idx}",
    )

    col_prev, col_submit, col_next = st.columns([1, 2, 1])

    with col_prev:
        if idx > 0:
            if st.button("⬅️ Previous"):
                st.session_state.answers[idx] = answer
                st.session_state.current_q_idx -= 1
                st.rerun()

    # ── show guardrail warning if present ──
    if st.session_state.get("guardrail_warn"):
        st.info(st.session_state.guardrail_warn, icon="ℹ️")

    with col_submit:
        submit_label = (
            "✅ Submit & Evaluate"
            if idx not in st.session_state.feedbacks
            else "🔁 Re-evaluate"
        )
        if st.button(submit_label, type="primary", use_container_width=True):
            if not answer.strip():
                st.warning("Please write an answer before submitting.")
                st.stop()

            # ── Guardrail: validate answer submission ──
            try:
                guardrails.validate_answer_submission(
                    question=questions[idx],
                    answer=answer,
                    job_role=p["role"],
                    experience_level=p["exp"],
                    interview_type=p["itype"],
                )
            except GuardrailViolation as gv:
                st.error(f"⚠️ {gv}")
                st.stop()

            st.session_state.answers[idx] = answer

            with st.spinner("IBM Granite is evaluating your answer…"):
                try:
                    feedback = granite_service.evaluate_answer(
                        question=questions[idx],
                        candidate_answer=answer,
                        job_role=p["role"],
                        experience_level=p["exp"],
                        interview_type=p["itype"],
                        rag_context=st.session_state.rag_context,
                    )
                except Exception as e:
                    st.error(f"❌ Evaluation error: {e}")
                    st.stop()

            st.session_state.feedbacks[idx] = feedback
            st.rerun()

    with col_next:
        if idx < total - 1:
            if st.button("Next ➡️"):
                st.session_state.answers[idx] = answer
                st.session_state.current_q_idx += 1
                st.rerun()
        elif len(st.session_state.feedbacks) == total:
            if st.button("📊 View Report", type="primary"):
                st.session_state.stage = "report"
                st.rerun()

    # ── feedback for this question ──
    if idx in st.session_state.feedbacks:
        fb = st.session_state.feedbacks[idx]
        st.divider()
        st.markdown("#### 📝 Feedback")

        score_html = _score_badge(fb["score"])
        st.markdown(f"**Score:** {score_html}", unsafe_allow_html=True)

        with st.expander("✅ Strengths", expanded=True):
            st.write(fb.get("strengths", "—"))

        with st.expander("⚠️ Missing Concepts"):
            st.write(fb.get("missing_concepts", "—"))

        with st.expander("🔧 Areas for Improvement"):
            st.write(fb.get("improvements", "—"))

        with st.expander("💡 Suggested Model Answer"):
            st.write(fb.get("model_answer", "—"))

        # ── confidence tip (new) ──
        conf_tip = fb.get("confidence_tip", "")
        if conf_tip:
            st.markdown(
                f'<div class="conf-card">🌟 <strong>Confidence tip:</strong> {conf_tip}</div>',
                unsafe_allow_html=True,
            )

    # ── completion prompt ──
    if len(st.session_state.feedbacks) == total and idx == total - 1:
        st.divider()
        st.success(
            "🎉 You've answered all questions! "
            "Click **View Report** above to see your overall performance."
        )

# ===========================================================================
# STAGE 3 — OVERALL PERFORMANCE REPORT
# ===========================================================================
elif st.session_state.stage == "report":

    p         = st.session_state.profile
    questions = st.session_state.questions
    feedbacks = st.session_state.feedbacks
    answers   = st.session_state.answers

    with st.sidebar:
        if st.session_state.prep_strategy:
            if st.button("📚 View Prep Strategy"):
                st.session_state.stage = "prep"
                st.rerun()
        if st.button("🔄 Start New Interview"):
            _reset()
            st.rerun()

    st.subheader("📊 Overall Performance Report")
    st.caption(
        f"Candidate: **{p['name']}** · Role: **{p['role']}** · "
        f"Level: **{p['exp']}** · Type: **{p['itype']}**"
    )
    st.divider()

    # ── aggregate metrics ──
    scores = [fb["score"] for fb in feedbacks.values()]
    avg    = round(sum(scores) / len(scores), 1) if scores else 0
    grade  = (
        "Excellent 🏆" if avg >= 8 else
        ("Good 👍"     if avg >= 6 else
        ("Average 📈"  if avg >= 4 else
         "Needs Work 📚"))
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Average Score",        f"{avg} / 10")
    m2.metric("Questions Answered",   f"{len(feedbacks)} / {len(questions)}")
    m3.metric("Overall Grade",        grade)

    st.divider()

    # ── per-question breakdown ──
    st.markdown("#### 🗒️ Question-by-Question Summary")
    for i, q in enumerate(questions):
        if i not in feedbacks:
            continue
        fb    = feedbacks[i]
        badge = _score_badge(fb["score"])
        with st.expander(
            f"Q{i+1}: {q[:80]}{'…' if len(q) > 80 else ''}",
            expanded=False,
        ):
            st.markdown(f"**Score:** {badge}", unsafe_allow_html=True)
            st.markdown("**Your Answer:**")
            st.info(answers.get(i, "_(no answer)_"))
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**✅ Strengths**")
                st.write(fb.get("strengths", "—"))
                st.markdown("**⚠️ Missing Concepts**")
                st.write(fb.get("missing_concepts", "—"))
            with cols[1]:
                st.markdown("**🔧 Improvements**")
                st.write(fb.get("improvements", "—"))
                st.markdown("**💡 Model Answer**")
                st.write(fb.get("model_answer", "—"))

    st.divider()

    # ── topics covered ──
    ctx_topics = st.session_state.rag_context.get("topics", [])
    if ctx_topics:
        st.markdown("#### 📚 Topics Covered in This Session")
        tcols = st.columns(3)
        for i, topic in enumerate(ctx_topics):
            tcols[i % 3].markdown(f"- {topic}")

    st.divider()

    # ── preparation tips recap (always shown from KB, no extra API call) ──
    kb_tips = st.session_state.rag_context.get("preparation_tips", [])
    if kb_tips:
        st.markdown("#### 🎯 What to Work on Next")
        for tip in kb_tips:
            st.markdown(
                f'<div class="tip-card">💡 {tip}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

    st.divider()

    # ── action buttons ──
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("🔄 Try Again (same profile)", use_container_width=True):
            saved = st.session_state.profile.copy()
            saved_ctx = st.session_state.rag_context.copy()
            saved_prep = st.session_state.prep_strategy.copy()
            _reset()
            st.session_state.profile = saved
            st.session_state.rag_context = saved_ctx
            st.session_state.prep_strategy = saved_prep
            st.session_state.stage = "interview"
            # Re-generate questions
            with st.spinner("Regenerating questions…"):
                try:
                    qs = granite_service.generate_questions(
                        candidate_name=saved["name"],
                        job_role=saved["role"],
                        experience_level=saved["exp"],
                        skills=saved["skills"],
                        interview_type=saved["itype"],
                        rag_context=saved_ctx,
                        num_questions=saved["num_q"],
                    )
                    st.session_state.questions = qs
                except Exception:
                    st.session_state.stage = "profile"
            st.rerun()
    with btn2:
        if st.button("📚 Get New Prep Strategy", use_container_width=True):
            st.session_state.stage = "prep"
            st.rerun()
    with btn3:
        if st.button("🆕 New Interview", type="primary", use_container_width=True):
            _reset()
            st.rerun()

    st.divider()
    st.caption(
        "Built for IBM SkillsBuild AICTE-2026 · "
        "Problem Statement #22 – Interview Trainer Agent · "
        "Powered by IBM Granite on IBM watsonx.ai"
    )
