import streamlit as st
import requests

API_ENDPOINT = "http://127.0.0.1:8080/v1/evaluate"

def configure_page():
    st.set_page_config(layout="wide", page_title="GenAI Output Validator")
    st.markdown("# AI Output Factual Consistency Checker")
    st.write("Compare generated text against its source context to detect hallucinations, contradictions, and structural degradation.")

def render_ui():
    configure_page()

    pane_left, pane_right = st.columns(2)

    with pane_left:
        st.markdown("### Ground Truth Context")
        source_data = st.text_area(
            "Insert the factual baseline (e.g., retrieved RAG documents)", 
            height=280,
            key="ctx_box"
        )

    with pane_right:
        st.markdown("### Generated Response")
        llm_output = st.text_area(
            "Insert the text produced by the generative model", 
            height=280,
            key="gen_box"
        )

    if st.button("Execute Validation Check", type="primary"):
        if not source_data.strip() or not llm_output.strip():
            st.warning("Both the context and the generated response are required to run the evaluation.")
            return

        with st.spinner("Analyzing semantic entailment and calculating lexical metrics..."):
            try:
                api_response = requests.post(
                    API_ENDPOINT, 
                    json={
                        "source_context": source_data,
                        "generated_text": llm_output
                    },
                    timeout=30
                )
                
                if api_response.status_code == 200:
                    payload = api_response.json()
                    display_results(payload)
                else:
                    st.error(f"Server rejected request: {api_response.text}")

            except requests.exceptions.RequestException as err:
                st.error(f"Failed to connect to the evaluation API. Ensure the backend is running. Details: {str(err)}")

def display_results(payload_data):
    st.divider()
    
    # Extract the summary and raw data
    narrative_summary = payload_data.get("summary", "No summary provided by API.")
    metrics_block = payload_data.get("data", {})
    
    # 1. Prominent Human-Readable Summary
    st.markdown("## Diagnostic Summary")
    
    # Style the box based on the dominant NLI label to give immediate visual feedback
    nli_status = metrics_block.get("nli_inference", {}).get("dominant_label", "")
    if nli_status == "Contradiction":
        st.error(narrative_summary)
    elif nli_status == "Entailment":
        st.success(narrative_summary)
    else:
        st.info(narrative_summary)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Metric Breakdown")
    
    # 2. Key Metrics Display
    col_a, col_b, col_c = st.columns(3)
    
    inference_stats = metrics_block.get("nli_inference", {})
    col_a.metric(
        label="NLI Classification", 
        value=inference_stats.get("dominant_label", "Unknown")
    )
    col_a.caption(f"Confidence: {max(inference_stats.get('entailment_prob', 0), inference_stats.get('contradiction_prob', 0)):.2f}")
    
    overlap_stats = metrics_block.get("factual_alignment", {})
    col_b.metric(
        label="Entity Retention", 
        value=f"{overlap_stats.get('entity_overlap', 0) * 100:.1f}%"
    )
    col_b.caption(f"Lexical Overlap: {overlap_stats.get('token_overlap', 0) * 100:.1f}%")
    
    structure_stats = metrics_block.get("structural_metrics", {})
    col_c.metric(
        label="Gibberish Factor", 
        value=round(structure_stats.get("gibberish_ratio", 0), 3)
    )
    col_c.caption(f"Coherence Index: {structure_stats.get('coherence_score', 0):.2f}")
    
    # 3. Raw Data Inspector
    with st.expander("Inspect Raw JSON Payload"):
        st.json(metrics_block)

if __name__ == "__main__":
    render_ui()
