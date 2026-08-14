"""CertCoach demo UI — requires the FastAPI server running at API_BASE."""
import json

import requests
import streamlit as st

st.set_page_config(page_title="CertCoach", page_icon="🎓", layout="centered")
st.title("🎓 CertCoach")
st.caption("RAG-powered cloud certification assistant")

with st.sidebar:
    st.header("Settings")
    import os
    api_base = st.text_input("API base URL", value=os.environ.get("API_BASE", "http://localhost:8000"))
    tenant = st.selectbox("Tenant", ["aws-saa", "gcp-ace", "terraform-assoc"])
    st.divider()
    try:
        health = requests.get(f"{api_base}/health", timeout=3).json()
        db_ok = health.get("db") == "ok"
        if db_ok:
            st.success("API ✓  DB ✓")
        else:
            st.warning("API ✓  DB ✗")
    except Exception:
        st.error("API unreachable")

tab_ask, tab_quiz = st.tabs(["Ask", "Quiz"])

# ── Ask tab ──────────────────────────────────────────────────────────────────
with tab_ask:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption("Sources: " + " · ".join(msg["citations"]))
            if msg.get("below_threshold"):
                st.warning("Low confidence — answer may be incomplete.")

    if prompt := st.chat_input("Ask a certification question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving…"):
                try:
                    resp = requests.post(
                        f"{api_base}/ask",
                        headers={"X-Tenant": tenant, "Content-Type": "application/json"},
                        json={"question": prompt},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    below = data.get("below_threshold", False)
                    confidence = data.get("confidence", 0.0)
                except Exception as e:
                    answer = f"Error: {e}"
                    citations = []
                    below = False
                    confidence = 0.0

            st.markdown(answer)
            if citations:
                st.caption("Sources: " + " · ".join(citations))
            if below:
                st.warning("Low confidence — answer may be incomplete.")
            st.caption(f"confidence: {confidence:.2f}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "below_threshold": below,
        })

# ── Quiz tab ──────────────────────────────────────────────────────────────────
with tab_quiz:
    st.subheader("Generate practice questions")

    quiz_topic = st.text_input("Topic", placeholder="e.g. IAM policies, VPC routing, S3 storage classes")
    quiz_n = st.slider("Number of questions", min_value=1, max_value=10, value=3)

    if st.button("Generate Quiz", disabled=not quiz_topic):
        with st.spinner("Generating…"):
            try:
                resp = requests.post(
                    f"{api_base}/quiz",
                    headers={"X-Tenant": tenant, "Content-Type": "application/json"},
                    json={"topic": quiz_topic, "n": quiz_n},
                    timeout=60,
                )
                resp.raise_for_status()
                raw = resp.json().get("questions", "")
                questions = json.loads(raw)
                st.session_state["quiz_questions"] = questions
                st.session_state["quiz_topic"] = quiz_topic
            except json.JSONDecodeError:
                st.error("Model returned malformed JSON. Raw output:")
                st.code(raw)
            except Exception as e:
                st.error(f"Error: {e}")

    questions = st.session_state.get("quiz_questions", [])
    if questions:
        if "quiz_grades" not in st.session_state:
            st.session_state["quiz_grades"] = {}

        st.divider()
        for i, q in enumerate(questions, start=1):
            question_text = q.get("question", "")
            correct_text = q.get("answer", "")
            options = q.get("options", [])

            with st.expander(f"Q{i}: {question_text}", expanded=True):
                chosen = st.radio(
                    "Your answer",
                    options=options,
                    key=f"quiz_q{i}",
                    index=None,
                )

                if st.button("Submit & Grade", key=f"quiz_check{i}", disabled=chosen is None):
                    with st.spinner("Grading…"):
                        try:
                            resp = requests.post(
                                f"{api_base}/grade",
                                headers={"X-Tenant": tenant, "Content-Type": "application/json"},
                                json={
                                    "question": question_text,
                                    "learner_answer": chosen,
                                    "reference": correct_text,
                                },
                                timeout=30,
                            )
                            resp.raise_for_status()
                            raw_result = resp.json().get("result", "{}")
                            grade = json.loads(raw_result)
                            grade["chosen"] = chosen
                            st.session_state["quiz_grades"][i] = grade
                        except Exception as e:
                            st.session_state["quiz_grades"][i] = {"error": str(e)}

                grade = st.session_state["quiz_grades"].get(i)
                if grade:
                    if "error" in grade:
                        st.error(f"Grading error: {grade['error']}")
                    else:
                        score = grade.get("score", 0.0)
                        correct = grade.get("correct", False)
                        feedback = grade.get("feedback", "")
                        gap = grade.get("gap", "")

                        if correct:
                            st.success(f"Correct! ({score:.0%})")
                        else:
                            st.error(f"Incorrect — {score:.0%}")
                            st.markdown(f"**Correct answer:** {correct_text}")

                        if feedback:
                            st.markdown(f"**Feedback:** {feedback}")
                        if gap:
                            st.markdown(f"**Gap:** {gap}")

# (Grade tab removed — grading is now inline under the Quiz tab)
