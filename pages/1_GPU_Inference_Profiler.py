import streamlit as st

from services.local_llm import LocalLLMRunner
from services.device_metrics import get_device_snapshot


st.set_page_config(
    page_title="GPU Inference Profiler",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GPU-Aware LLM Inference Profiler")

st.write(
    "This page runs a local LLM and measures how inference behaves on your machine, "
    "including device type, token counts, latency, tokens per second, RAM usage, and GPU memory."
)

st.divider()

with st.sidebar:
    st.header("Local LLM Settings")

    model_name = st.selectbox(
        "Choose local model",
        ["distilgpt2"],
        index=0
    )

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=10,
        max_value=150,
        value=50,
        step=10
    )

    st.caption("For the first version, we use distilgpt2 because it is small and runs fast on your Mac.")


device_snapshot = get_device_snapshot()

st.subheader("Current Device Snapshot")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Device", device_snapshot.get("device", "unknown"))

with col2:
    st.metric("GPU", device_snapshot.get("gpu_name", "unknown"))

with col3:
    st.metric("RAM Used GB", device_snapshot.get("ram_used_gb", "unknown"))

with col4:
    st.metric("RAM Total GB", device_snapshot.get("ram_total_gb", "unknown"))


if "local_runner" not in st.session_state:
    st.session_state.local_runner = None

if "loaded_model_name" not in st.session_state:
    st.session_state.loaded_model_name = None


st.divider()

st.subheader("Run Local LLM Inference")

prompt = st.text_area(
    "Enter your prompt",
    value="Explain GPU memory in LLM inference in simple words:",
    height=120
)

run_button = st.button("Run Local LLM", type="primary")

if run_button:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        try:
            with st.spinner("Loading model and generating response..."):
                if (
                    st.session_state.local_runner is None
                    or st.session_state.loaded_model_name != model_name
                ):
                    st.session_state.local_runner = LocalLLMRunner(model_name)
                    load_info = st.session_state.local_runner.load_model()
                    st.session_state.loaded_model_name = model_name
                else:
                    load_info = {
                        "model_name": model_name,
                        "device": st.session_state.local_runner.device,
                        "model_load_time_sec": "Already loaded"
                    }

                result = st.session_state.local_runner.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens
                )

            st.success("Inference completed successfully.")

            st.subheader("Generated Response")
            st.write(result["response"])

            st.subheader("Inference Metrics")

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.metric("Input Tokens", result["input_tokens"])

            with m2:
                st.metric("Output Tokens", result["output_tokens"])

            with m3:
                st.metric("Total Tokens", result["total_tokens"])

            with m4:
                st.metric("Latency Sec", result["latency_sec"])

            with m5:
                st.metric("Tokens / Sec", result["tokens_per_second"])

            st.subheader("Model Load Info")
            st.json(load_info)

            st.subheader("Before Inference Snapshot")
            st.json(result["before_snapshot"])

            st.subheader("After Inference Snapshot")
            st.json(result["after_snapshot"])

        except Exception as error:
            st.error("Something went wrong while running local inference.")
            st.exception(error)
