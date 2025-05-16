import streamlit as st
from query_service import EnhancedQueryService
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import re

st.set_page_config(page_title="Document Analysis Assistant", layout="wide")
st.title("📘 Document Analysis Assistant")

# Initialize session state
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "ppt_slides" not in st.session_state:
    st.session_state.ppt_slides = []

HISTORY_DIR = "reports"
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_report(content, filename):
    path = os.path.join(HISTORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def load_history():
    return [f for f in os.listdir(HISTORY_DIR) if f.endswith(".md")]

def generate_new_report(query_text):
    """Generate full report and cache it"""
    service = EnhancedQueryService(top_k=3)
    filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
    full_path = os.path.join(HISTORY_DIR, filename)
    report_path = service.query_to_ppt(query_text, output_path=full_path)

    with open(report_path, encoding="utf-8") as f:
        content = f.read()

    slides = content.split("---\n")[2:-1]
    st.session_state.ppt_slides = slides
    st.session_state.current_report = report_path
    st.session_state.query_history.insert(0, {
        "timestamp": datetime.now(),
        "query": query_text,
        "path": report_path
    })
    return report_path

def display_full_report(slides, query_text, report_path):
    """Display all PPT pages continuously, support asking questions and generating sub-pages"""
    service = EnhancedQueryService()

    try:
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        
        # More robust slide splitting logic
        parts = [p.strip() for p in content.split("---\n") if p.strip()]
        
        # Find actual slide starting position (skip title and metadata)
        start_idx = 0
        for i, part in enumerate(parts):
            if "<!-- PPT Slide" in part:
                start_idx = i
                break
        
        # Extract and organize slides
        organized_slides = []
        for part in parts[start_idx:]:
            if not part.startswith("<!-- PPT Slide"):
                continue

            # Extract slide number
            first_line = part.split("\n")[0]
            try:
                match = re.search(r"<!-- PPT Slide ([\d\.]+) -->", first_line)
                if match:
                    slide_num = match.group(1)
                else:
                    slide_num = str(len(organized_slides) + 1)
            except:
                slide_num = str(len(organized_slides) + 1)

            organized_slides.append({
                "content": part,
                "num": slide_num,
                "display_num": slide_num  # ✅ Display number = original number
            })

        # Sorting function supports any level of nested slide numbers
        def sort_key(item):
            try:
                return tuple(int(p) for p in item["num"].split("."))
            except:
                return (0,)

        organized_slides.sort(key=sort_key)

        slides = [slide["content"] for slide in organized_slides]
        st.session_state.ppt_slides = slides
        st.session_state.slide_metadata = organized_slides
        
    except Exception as e:
        st.error(f"Failed to load report: {str(e)}")
        return

    # Display slides
    for i, slide_content in enumerate(slides):
        slide_meta = st.session_state.slide_metadata[i]
        lines = slide_content.split("\n")
        title = ""
        image_data = ""
        explanation = ""

        for line in lines:
            if line.startswith("# "):
                title = line[2:]
            elif "base64" in line:
                image_data = line.split(",")[1]
            elif line.strip() and not line.startswith("<!--"):
                explanation += line + "\n"

        with st.container():
            # ✅ Use real number as display number
            st.subheader(f"{title} (Slide {slide_meta['display_num']})")

            col1, col2 = st.columns([1, 2])
            with col1:
                if image_data:
                    try:
                        img = Image.open(BytesIO(base64.b64decode(image_data)))
                        st.image(img, use_container_width=True)
                    except:
                        st.error("Failed to display image")
            with col2:
                st.markdown(explanation)

            with st.expander("💬 Ask a question (for follow-up on current slide)"):
                user_question = st.text_input(
                    f"Your question (Slide {slide_meta['display_num']})", 
                    key=f"q_{slide_meta['num']}"
                )
                if st.button("Submit question", key=f"btn_{slide_meta['num']}"):
                    if user_question.strip():
                        with st.spinner("Generating sub-page, please wait..."):
                            service.handle_followup(
                                slide_num=slide_meta["num"],
                                followup_question=user_question,
                                original_image=image_data,
                                image_description=title,
                                query_text=query_text,
                                ppt_path=report_path
                            )
                            st.success("Sub-slide generated ✅")
                            st.rerun()
                    else:
                        st.warning("Please enter a valid question")

        st.markdown("---")

# =======================
# 🎯 Streamlit UI Layout
# =======================

# Sidebar controls
with st.sidebar:
    st.header("📂 Query History")
    history = load_history()
    selected_report = st.selectbox("Select historical report", [""] + history)

    if selected_report:
        try:
            with open(os.path.join(HISTORY_DIR, selected_report), encoding="utf-8") as f:
                content = f.read()
                st.session_state.ppt_slides = content.split("---\n")[2:-1]
                st.session_state.current_report = os.path.join(HISTORY_DIR, selected_report)
        except Exception as e:
            st.error(f"Loading failed: {str(e)}")

    if st.button("🗑️ Clear current cache"):
        # Delete report file
        if st.session_state.current_report:
            try:
                os.remove(st.session_state.current_report)
            except Exception as e:
                st.warning(f"Failed to delete report file: {e}")

        # Clear state variables
        st.session_state.current_report = None
        st.session_state.ppt_slides = []
        st.session_state.query_history = []
        st.rerun()
    if st.button("🧹 Clear all history and reports"):
        # Delete all .md files in reports/
        deleted_files = []
        for fname in os.listdir(HISTORY_DIR):
            if fname.endswith(".md"):
                try:
                    os.remove(os.path.join(HISTORY_DIR, fname))
                    deleted_files.append(fname)
                except Exception as e:
                    st.warning(f"Failed to delete {fname}: {e}")
        
        # Clear session state
        st.session_state.current_report = None
        st.session_state.ppt_slides = []
        st.session_state.query_history = []
        st.rerun()

# Main interface: input question
query_text = st.text_input("🔍 Enter your analysis question (English):", 
                         placeholder="e.g.: how climate change affects agriculture")

if st.button("🚀 Generate analysis report"):
    if query_text.strip():
        with st.spinner("Generating analysis report..."):
            report_path = generate_new_report(query_text)
            st.success("Report generation complete ✅")
    else:
        st.warning("Please enter valid query content")

# Display full report
if st.session_state.ppt_slides:
    st.markdown("---")
    st.header("📑 Analysis Report Display")
    display_full_report(
        st.session_state.ppt_slides,
        query_text=query_text if query_text else "no query",
        report_path=st.session_state.current_report
    )

    # Download button
    if st.session_state.current_report:
        with open(st.session_state.current_report, "rb") as f:
            report_bytes = f.read()
        st.download_button(
            label="📥 Download full report",
            data=report_bytes,
            file_name=os.path.basename(st.session_state.current_report),
            mime="text/markdown"
        )

