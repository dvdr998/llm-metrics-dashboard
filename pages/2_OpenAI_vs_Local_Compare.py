import os
import streamlit as st

from services.openai_runner import run_openai_response
from services.local_llm import LocalLLMRunner
from services.device_metrics import get_device_snapshot


st.set_page_config(
    page_title="OpenAI vs Local LLM Compare",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 OpenAI API vs Local LLM Inference Compare")

st.write(
    "This page compares a hosted OpenAI model with a local LLM running in the current runtime environment."
)

st.info(
    "OpenAI metrics show hosted API behavior such as tokens and latency. "
    "Local LLM metrics show runtime behavior such as device type, RAM usage, GPU availability, latency, and tokens per second."
)

st.divider()


def get_api_key():
    key = os.getenv("OPENAI_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None


with st.sidebar:
    st.header("Comparison Settings")

    openai_model = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini"],
        index=0
    )

    local_model = st.selectbox(
        "Local model",
        ["distilgpt2"],
        index=0
    )

    max_output_tokens = st.slider(
        "Max output/new tokens",
        min_value=20,
        max_value=200,
        value=80,
        step=20
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.2,
        value=0.7,
        step=0.1
    )


api_key = get_api_key()

if not api_key:
    st.warning(
        "OpenAI API key was not found. Add OPENAI_API_KEY in Streamlit secrets or your local environment to run OpenAI comparison."
    )


device_snapshot = get_device_snapshot()

st.subheader("Current Runtime Environment")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Runtime Device", device_snapshot.get("device", "unknown"))

with c2:
    st.metric("Host GPU / Accelerator", device_snapshot.get("gpu_name", "unknown"))

with c3:
    st.metric("Host RAM Used GB", device_snapshot.get("ram_used_gb", "unknown"))

with c4:
    st.metric("Host RAM Total GB", device_snapshot.get("ram_total_gb", "unknown"))


st.divider()

prompt = st.text_area(
    "Enter the same prompt for both models",
    value="Explain why GPU memory matters in LLM inference in simple words.",
    height=130
)

run_compare = st.button("Run OpenAI vs Local Compare", type="primary")


if "compare_local_runner" not in st.session_state:
    st.session_state.compare_local_runner = None

if "compare_loaded_model_name" not in st.session_state:
    st.session_state.compare_loaded_model_name = None


if run_compare:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    elif not api_key:
        st.error("OpenAI API key is missing, so comparison cannot run.")
    else:
        try:
            with st.spinner("Running OpenAI hosted model..."):
                openai_result = run_openai_response(
                    prompt=prompt,
                    model=openai_model,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    api_key=api_key
                )

            with st.spinner("Running local LLM in current runtime..."):
                if (
                    st.session_state.compare_local_runner is None
                    or st.session_state.compare_loaded_model_name != local_model
                ):
                    st.session_state.compare_local_runner = LocalLLMRunner(local_model)
                    local_load_info = st.session_state.compare_local_runner.load_model()
                    st.session_state.compare_loaded_model_name = local_model
                else:
                    local_load_info = {
                        "model_name": local_model,
                        "device": st.session_state.compare_local_runner.device,
                        "model_load_time_sec": "Already loaded"
                    }

                local_result = st.session_state.compare_local_runner.generate(
                    prompt=prompt,
                    max_new_tokens=max_output_tokens
                )

            st.success("Comparison completed successfully.")

            st.subheader("Side-by-Side Metrics")

            left, right = st.columns(2)

            with left:
                st.markdown("### OpenAI Hosted API")

                st.metric("Model", openai_result["model_name"])
                st.metric("Input Tokens", openai_result["input_tokens"])
                st.metric("Output Tokens", openai_result["output_tokens"])
                st.metric("Total Tokens", openai_result["total_tokens"])
                st.metric("Latency Sec", openai_result["latency_sec"])
                st.metric("Tokens / Sec", openai_result["tokens_per_second"])

                st.markdown("#### OpenAI Response")
                st.write(openai_result["response"])

            with right:
                st.markdown("### Local LLM Runtime")

                st.metric("Model", local_result["model_name"])
                st.metric("Runtime Device", local_result["device"])
                st.metric("Input Tokens", local_result["input_tokens"])
                st.metric("Output Tokens", local_result["output_tokens"])
                st.metric("Total Tokens", local_result["total_tokens"])
                st.metric("Latency Sec", local_result["latency_sec"])
                st.metric("Tokens / Sec", local_result["tokens_per_second"])

                st.markdown("#### Local LLM Response")
                st.write(local_result["response"])

            st.divider()

            st.subheader("Comparison Summary")

            st.table([
                {
                    "Metric": "Execution location",
                    "OpenAI API": "OpenAI hosted infrastructure",
                    "Local LLM": "Current app runtime environment"
                },
                {
                    "Metric": "Token accuracy",
                    "OpenAI API": "API usage field when available",
                    "Local LLM": "Tokenizer-based local count"
                },
                {
                    "Metric": "Hardware visibility",
                    "OpenAI API": "Not visible to this app",
                    "Local LLM": "Runtime device, RAM, GPU/MPS signals"
                },
                {
                    "Metric": "Best use",
                    "OpenAI API": "Production-quality hosted model response",
                    "Local LLM": "Local inference performance profiling"
                }
            ])

            st.subheader("Local Model Load Info")
            st.json(local_load_info)

            st.subheader("Local Before Inference Snapshot")
            st.caption("Host/runtime snapshot before local inference.")
            st.json(local_result["before_snapshot"])

            st.subheader("Local After Inference Snapshot")
            st.caption("Host/runtime snapshot after local inference.")
            st.json(local_result["after_snapshot"])

        except Exception as error:
            st.error("Something went wrong while running the comparison.")
            st.exception(error)
