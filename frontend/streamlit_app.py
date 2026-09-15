import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Policy Assistant", page_icon="📄")
st.title("📄 Company Policy Assistant")
st.caption("Answers are grounded in company policy documents, with sources cited for every response.")

if "history" not in st.session_state:
    st.session_state.history = []
    
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("citations"):
            with st.expander("Sources"):
                for c in turn["citations"]:
                    st.markdown(f"**{c['source']}** — *{c['section']}*")
                    st.caption(c["snippet"])


question = st.chat_input("Ask about PTO, remote work, security, expenses...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = None, []
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": question},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                citations = data.get("citations", [])
                latency = data.get("latency_ms")
            except requests.exceptions.ConnectionError:
                answer = f"Can't reach the backend at {BACKEND_URL}. Is it running?"
            except requests.exceptions.Timeout:
                answer = "The backend took too long to respond. Try again."
            except requests.exceptions.HTTPError:
                answer = f"Backend returned an error: {resp.status_code} — {resp.text}"
            except Exception as e:
                answer = f"Unexpected error: {e}"

            st.markdown(answer)
            if latency := locals().get("latency"):
                st.caption(f"Latency: {latency} ms")
            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        st.markdown(f"**{c['source']}** — *{c['section']}*")
                        st.caption(c["snippet"])

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )