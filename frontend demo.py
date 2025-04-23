import streamlit as st
from query_service import ReportGenerator
import os
from datetime import datetime

st.set_page_config(page_title="Document Analysis Assistant", layout="wide")
st.title("Document Analysis Assistant")

if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "current_report_path" not in st.session_state:
    st.session_state.current_report_path = None
if "follow_ups" not in st.session_state:
    st.session_state.follow_ups = []

HISTORY_FILE = "report_history.txt"

def save_history(filename):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if os.path.exists(line.strip())]
    return []

def delete_from_history(filename_to_delete):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            for line in lines:
                if line.strip() != filename_to_delete:
                    f.write(line)

def generate_and_append(query_text, is_new=False):
    generator = ReportGenerator()
    if is_new:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"report_{timestamp}.md"
        follow_ups = generator.generate_report(query_text, path, append=False)
        st.session_state.current_report_path = path
        st.session_state.last_report = path
        save_history(path)
    else:
        path = st.session_state.current_report_path
        follow_ups = generator.generate_report(query_text, path, append=True)

    st.session_state.follow_ups = follow_ups

def display_full_report(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f.read())


user_query = st.text_area("Enter your question about the document:", value=st.session_state.user_query, key="query_input")

if st.button("Generate Report"):
    if not user_query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        st.session_state.user_query = user_query
        generate_and_append(user_query, is_new=True)
        st.rerun()


if st.session_state.current_report_path and os.path.exists(st.session_state.current_report_path):
    st.markdown("  Current Report")
    display_full_report(st.session_state.current_report_path)

    if st.session_state.follow_ups:
        st.markdown("  Expand Your Understanding")
        for question in st.session_state.follow_ups:
            if st.button(question, key=f"followup_{question}"):
                st.session_state.user_query = question
                generate_and_append(question, is_new=False)
                st.rerun()


st.markdown("---")
st.subheader("View Historical Reports")

history = load_history()
if history:
    selected_report = st.selectbox("Select a report to view or delete", [""] + history[::-1])
    if selected_report != "":
        with open(selected_report, "r", encoding="utf-8") as f:
            st.markdown(f"### Viewing: {selected_report}")
            st.markdown(f.read())

        if st.button("Delete This Report"):
            os.remove(selected_report)
            delete_from_history(selected_report)
            st.success(f"Deleted: {selected_report}")
            if st.session_state.last_report == selected_report:
                st.session_state.last_report = None
                st.session_state.current_report_path = None
                st.session_state.follow_ups = []
            st.rerun()
else:
    st.info("No historical reports found.")
