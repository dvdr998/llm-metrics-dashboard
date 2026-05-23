# LLM Metrics Dashboard - Streamlit App
# A simple dashboard to track mock LLM prompt metrics.

import csv
import os
import time
from io import StringIO

import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator

from src.charts import (
    create_cost_chart_data,
    create_latency_chart_data,
    create_model_usage_chart_data,
    create_summary_metrics_display,
    create_token_usage_chart_data,
)
from src.config import DEFAULT_MODEL, USE_MOCK_LLM, is_api_key_valid
from src.database import (
    clear_prompt_logs,
    get_all_prompt_logs,
    initialize_database,
    save_prompt_log,
)
from src.llm_client import run_mock_llm
from src.metrics import estimate_cost
from src.metrics import estimate_tokens


APP_VERSION = "0.1.0"
REAL_OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-4o-mini",
]
MOCK_MODE_MODELS = [
    "gpt-5.2",
    "gpt-4o-mini",
    "mock-fast-model",
    "mock-quality-model",
]


# Configure the page before building the UI.
st.set_page_config(
    page_title="LLM Metrics Dashboard",
    page_icon="📊",
    layout="wide",
)

# Load environment variables from .env file.
load_dotenv()

# Initialize the SQLite database on app startup.
database_connected = initialize_database()


def convert_rows_to_csv(rows: list) -> str:
    """
    Convert a list of dictionaries into CSV text for downloading.
    """
    if not rows:
        return ""

    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    return csv_buffer.getvalue()


def get_recent_prompt_logs(prompt_logs: list, limit: int = 20) -> list:
    """
    Return only the most recent prompt logs for cleaner dashboard charts.
    """
    return prompt_logs[:limit]


def create_latency_figure(rows: list):
    """
    Create a matplotlib chart for latency over time.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    run_numbers = list(range(1, len(rows) + 1))
    latency_values = [row["latency_seconds"] for row in rows]

    ax.plot(run_numbers, latency_values, marker="o", linewidth=2, color="#2563eb")
    ax.set_title("Latency Trend (Recent Prompt Runs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recent prompt run")
    ax.set_ylabel("Latency (seconds)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return fig


def create_cost_figure(rows: list):
    """
    Create a matplotlib chart for total cost over time.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    run_numbers = list(range(1, len(rows) + 1))
    cost_values = [row["total_cost"] for row in rows]

    ax.plot(run_numbers, cost_values, marker="o", linewidth=2, color="#16a34a")
    ax.set_title("Estimated Cost Trend (Recent Prompt Runs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recent prompt run")
    ax.set_ylabel("Estimated Cost (USD)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return fig


def create_token_usage_figure(rows: list):
    """
    Create a matplotlib chart for estimated token usage over time.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    run_numbers = list(range(1, len(rows) + 1))
    input_tokens = [row["input_tokens"] for row in rows]
    output_tokens = [row["output_tokens"] for row in rows]

    ax.plot(run_numbers, input_tokens, marker="o", linewidth=2, label="Billed input tokens")
    ax.plot(run_numbers, output_tokens, marker="o", linewidth=2, label="Output tokens")
    ax.set_title("Estimated Token Usage (Recent Prompt Runs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recent prompt run")
    ax.set_ylabel("Estimated Tokens")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return fig


def create_model_usage_figure(rows: list):
    """
    Create a matplotlib chart for model usage counts.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    models = [row["model"] for row in rows]
    prompt_runs = [row["prompt_runs"] for row in rows]

    ax.bar(models, prompt_runs, color="#9333ea")
    ax.set_title("Prompt Runs by Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Prompt Runs")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    return fig


def show_mock_mode_notice():
    """
    Show a clear notice that this app is not calling any real API.
    """
    st.info(
        "Mock Mode is active. No real OpenAI API calls will be made, "
        "and no API key is required."
    )


def get_openai_api_key() -> str | None:
    """
    Read OPENAI_API_KEY from Streamlit secrets first, then environment variables.
    
    Streamlit Cloud stores secrets in st.secrets. Local development often uses
    a .env file or shell environment variable.
    """
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
        if secret_key:
            return str(secret_key)
    except Exception:
        # st.secrets may not be configured in local development.
        pass

    return os.getenv("OPENAI_API_KEY")


def get_usage_value(usage, field_name: str) -> int | None:
    """
    Safely read token usage from an OpenAI Responses API usage object.
    """
    if usage is None:
        return None

    if isinstance(usage, dict):
        return usage.get(field_name)

    return getattr(usage, field_name, None)


def run_real_openai_llm(prompt: str, model: str, temperature: float, api_key: str) -> dict:
    """
    Send one prompt to OpenAI using the Responses API.
    
    This function is only called when the user explicitly selects Real API Mode
    and a valid API key is available.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    start_time = time.time()

    try:
        prompt_token_count = client.responses.input_tokens.count(
            model=model,
            input=prompt,
        )
        prompt_tokens = get_usage_value(prompt_token_count, "input_tokens")
        if prompt_tokens is None:
            prompt_tokens = get_usage_value(prompt_token_count, "total_tokens")
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt, model)
    except Exception:
        # Keep Real API Mode usable if token counting is unavailable in the
        # installed SDK version.
        prompt_tokens = estimate_tokens(prompt, model)

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=180,
    )

    latency_ms = (time.time() - start_time) * 1000
    response_text = response.output_text

    billed_input_tokens = get_usage_value(response.usage, "input_tokens")
    output_tokens = get_usage_value(response.usage, "output_tokens")
    total_tokens = get_usage_value(response.usage, "total_tokens")

    # If usage details are ever unavailable, fall back to local estimates so
    # the dashboard and database still have usable values.
    if billed_input_tokens is None:
        billed_input_tokens = estimate_tokens(prompt, model)
    if output_tokens is None:
        output_tokens = estimate_tokens(response_text, model)
    if total_tokens is None:
        total_tokens = billed_input_tokens + output_tokens

    return {
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "billed_input_tokens": billed_input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "model": model,
    }


# Read logs once near the top so the sidebar and tabs share the same data.
prompt_logs = get_all_prompt_logs()


# Sidebar: quick project status for portfolio viewers.
with st.sidebar:
    st.title("LLM Metrics Dashboard")
    st.caption("Portfolio project status")
    selected_mode = st.radio(
        "Mode",
        options=["Mock Mode", "Real API Mode"],
        index=0,
    )
    # Only check secrets or environment variables when Real API Mode is chosen.
    # This prevents Streamlit from showing a secrets warning in Mock Mode.
    openai_api_key = None
    if selected_mode == "Real API Mode":
        openai_api_key = get_openai_api_key()

    st.divider()
    st.write(f"**Current mode:** {selected_mode}")
    if selected_mode == "Mock Mode":
        st.write("**API status:** Not required in Mock Mode")
    elif is_api_key_valid(openai_api_key):
        st.write("**API status:** Configured")
    else:
        st.write("**API status:** Missing")
    st.write(f"**Database status:** {'Connected' if database_connected else 'Not connected'}")
    st.write(f"**Total prompt runs:** {len(prompt_logs)}")
    st.write(f"**Version:** {APP_VERSION}")


st.title("LLM Metrics Dashboard")
st.caption("Cost, latency, token usage, and response tracking for LLM prompt runs.")

if selected_mode == "Mock Mode":
    show_mock_mode_notice()
elif not is_api_key_valid(openai_api_key):
    st.warning(
        "Real API Mode is selected, but no valid OPENAI_API_KEY was found. "
        "Add it to Streamlit secrets or your local environment before running real calls."
    )
else:
    st.info("Real API Mode is selected. Prompt runs will call OpenAI and may incur API costs.")


prompt_tab, dashboard_tab, history_tab, info_tab = st.tabs(
    ["Prompt Runner", "Dashboard", "Prompt History", "Project Info"]
)


with prompt_tab:
    st.header("Prompt Runner")
    st.caption("Run a prompt and save the result to SQLite.")
    if selected_mode == "Mock Mode":
        show_mock_mode_notice()
        model_options = MOCK_MODE_MODELS
    else:
        st.info("Real API Mode uses the OpenAI Responses API and may incur API costs.")
        model_options = REAL_OPENAI_MODELS

    with st.form(key="prompt_runner_form"):
        user_prompt = st.text_area(
            label="Enter your prompt:",
            placeholder="Example: explain binary search",
            height=140,
        )

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            selected_model = st.selectbox(
                label="Select a model:",
                options=model_options,
            )

        with input_col2:
            temperature = st.slider(
                label="Temperature (Creativity):",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
            )

        submit_button = st.form_submit_button(label="Run Prompt", use_container_width=True)

    if submit_button:
        if not user_prompt.strip():
            st.error("Please enter a prompt before running.")
        elif selected_mode == "Mock Mode":
            with st.spinner("Generating mock response..."):
                result = run_mock_llm(user_prompt, selected_model)
            run_mode = "mock"
        else:
            if not is_api_key_valid(openai_api_key):
                st.warning(
                    "Real API Mode is selected, but no valid OPENAI_API_KEY was found. "
                    "Add it to Streamlit secrets or your local environment and try again."
                )
                result = None
                run_mode = None
            else:
                try:
                    with st.spinner("Calling OpenAI Responses API..."):
                        result = run_real_openai_llm(
                            prompt=user_prompt,
                            model=selected_model,
                            temperature=temperature,
                            api_key=openai_api_key,
                        )
                    run_mode = "real"
                except Exception as error:
                    st.error(f"OpenAI API call failed: {error}")
                    result = None
                    run_mode = None

        if result:
            latency_seconds = result["latency_ms"] / 1000
            prompt_tokens = result["prompt_tokens"]
            billed_input_tokens = result.get("billed_input_tokens", result["prompt_tokens"])
            output_tokens = result["completion_tokens"]
            total_tokens = result["total_tokens"]
            cost_info = estimate_cost(selected_model, billed_input_tokens, output_tokens)

            st.divider()
            st.subheader("Response")
            st.write(result["response"])

            st.subheader("Run Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric(label="Prompt Tokens", value=prompt_tokens)
            with metric_col2:
                input_token_label = (
                    "Billed Input Tokens"
                    if selected_mode == "Real API Mode"
                    else "Estimated Input Tokens"
                )
                st.metric(label=input_token_label, value=billed_input_tokens)
            with metric_col3:
                st.metric(label="Output Tokens", value=output_tokens)
            with metric_col4:
                st.metric(label="Total Tokens", value=total_tokens)

            latency_col, note_col = st.columns([1, 3])
            with latency_col:
                st.metric(label="Latency", value=f"{latency_seconds:.2f}s")
            with note_col:
                if selected_mode == "Real API Mode":
                    st.caption(
                        "Prompt Tokens counts only the text you typed. "
                        "Billed Input Tokens comes from OpenAI usage data and is used for cost calculation."
                    )
                else:
                    st.caption(
                        "Token counts are estimated in Mock Mode. "
                        "Prompt Tokens and Estimated Input Tokens use the same local estimate."
                    )

            cost_col1, cost_col2, cost_col3 = st.columns(3)

            with cost_col1:
                st.metric(label="Input Cost", value=f"${cost_info['input_cost']:.6f}")
            with cost_col2:
                st.metric(label="Output Cost", value=f"${cost_info['output_cost']:.6f}")
            with cost_col3:
                st.metric(label="Total Cost", value=f"${cost_info['total_cost']:.6f}")

            save_success = save_prompt_log(
                prompt=user_prompt,
                response=result["response"],
                model=selected_model,
                temperature=temperature,
                latency_seconds=latency_seconds,
                input_tokens=billed_input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_cost=cost_info["input_cost"],
                output_cost=cost_info["output_cost"],
                total_cost=cost_info["total_cost"],
                mode=run_mode,
            )

            if save_success:
                st.success("Prompt run saved to database.")
                # Reload logs so the Dashboard and Prompt History tabs include
                # this new run during the same app render.
                prompt_logs = get_all_prompt_logs()
                st.caption("Sidebar totals update on the next refresh.")
            else:
                st.error("Prompt run could not be saved to database.")

            if selected_mode == "Mock Mode":
                show_mock_mode_notice()


with dashboard_tab:
    st.header("Dashboard")
    st.caption(
        "This dashboard summarizes saved LLM prompt activity, including "
        "response latency, token estimates, and simulated cost trends."
    )

    if not prompt_logs:
        st.info("No prompt runs saved yet. Run a prompt first to populate the dashboard.")
    else:
        summary_metrics = create_summary_metrics_display(prompt_logs)

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric(label="Total Prompt Runs", value=summary_metrics["total_prompt_runs"])
        with summary_col2:
            st.metric(label="Average Latency", value=f"{summary_metrics['average_latency']:.2f}s")
        with summary_col3:
            st.metric(label="Total Tokens Used", value=summary_metrics["total_tokens"])
        with summary_col4:
            st.metric(label="Total Estimated Cost", value=f"${summary_metrics['total_cost']:.6f}")

        st.divider()

        chart_limit = st.slider(
            "Show recent prompt runs",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
        )
        st.caption("This only limits chart display. Full history remains saved in SQLite.")

        if len(prompt_logs) < 2:
            st.info("Run a few more prompts to generate more useful charts.")
        else:
            recent_prompt_logs = get_recent_prompt_logs(prompt_logs, chart_limit)

            latency_chart_data = create_latency_chart_data(recent_prompt_logs)
            cost_chart_data = create_cost_chart_data(recent_prompt_logs)
            token_chart_data = create_token_usage_chart_data(recent_prompt_logs)
            model_usage_chart_data = create_model_usage_chart_data(recent_prompt_logs)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Latency Trend")
                latency_fig = create_latency_figure(latency_chart_data)
                st.pyplot(latency_fig)
                plt.close(latency_fig)

            with chart_col2:
                st.subheader("Estimated Cost Trend")
                cost_fig = create_cost_figure(cost_chart_data)
                st.pyplot(cost_fig)
                plt.close(cost_fig)

            st.divider()

            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                st.subheader("Token Usage Trend")
                token_fig = create_token_usage_figure(token_chart_data)
                st.pyplot(token_fig)
                plt.close(token_fig)

            with chart_col4:
                st.subheader("Prompt Runs by Model")
                model_usage_fig = create_model_usage_figure(model_usage_chart_data)
                st.pyplot(model_usage_fig)
                plt.close(model_usage_fig)


with history_tab:
    st.header("Prompt History")
    st.caption("This table stores every mock prompt run saved in the local SQLite database.")

    if st.button("Refresh History"):
        st.rerun()

    if st.session_state.pop("prompt_history_cleared", False):
        st.success("Prompt history cleared.")

    if not prompt_logs:
        st.info("No prompt history yet. Run a prompt first.")
    else:
        history_columns = [
            "timestamp",
            "model",
            "prompt",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "total_cost",
            "mode",
        ]

        history_rows = []
        for log in prompt_logs:
            history_rows.append({column: log[column] for column in history_columns})

        if not history_rows:
            st.info("No prompt history yet. Run a prompt first.")
        else:
            history_action_col1, history_action_col2 = st.columns(2)

            with history_action_col1:
                confirm_clear_history = st.checkbox(
                    "I understand this will delete all saved prompt logs."
                )

            with history_action_col2:
                clear_history_clicked = st.button(
                    "Clear Prompt History",
                    disabled=not confirm_clear_history,
                )

            if clear_history_clicked:
                if clear_prompt_logs():
                    st.session_state["prompt_history_cleared"] = True
                    st.rerun()
                else:
                    st.error("Prompt history could not be cleared.")

            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                selected_history_model = st.selectbox(
                    "Filter by model",
                    options=["All"] + sorted({row["model"] for row in history_rows}),
                )

            with filter_col2:
                selected_history_mode = st.selectbox(
                    "Filter by mode",
                    options=["All"] + sorted({row["mode"] for row in history_rows}),
                )

            search_col, recent_col = st.columns(2)

            with search_col:
                prompt_search_text = st.text_input(
                    "Search prompt text",
                    placeholder="Search saved prompts...",
                )

            # Only render the slider when there is at least one history row.
            # This prevents Streamlit Cloud from receiving invalid min/max
            # values when a fresh database has no prompt records.
            with recent_col:
                recent_record_limit = st.slider(
                    "Show recent N records",
                    min_value=1,
                    max_value=len(history_rows),
                    value=min(10, len(history_rows)),
                    step=1,
                )

            filtered_history_rows = history_rows

            if selected_history_model != "All":
                filtered_history_rows = [
                    row for row in filtered_history_rows
                    if row["model"] == selected_history_model
                ]

            if selected_history_mode != "All":
                filtered_history_rows = [
                    row for row in filtered_history_rows
                    if row["mode"] == selected_history_mode
                ]

            if prompt_search_text.strip():
                search_text = prompt_search_text.lower().strip()
                filtered_history_rows = [
                    row for row in filtered_history_rows
                    if search_text in row["prompt"].lower()
                ]

            # The database query returns newest rows first, so slicing here
            # keeps the most recent matching records.
            filtered_history_rows = filtered_history_rows[:recent_record_limit]

            st.download_button(
                label="Download Filtered History as CSV",
                data=convert_rows_to_csv(filtered_history_rows),
                file_name="llm_prompt_history.csv",
                mime="text/csv",
                disabled=not filtered_history_rows,
            )

            if not filtered_history_rows:
                st.info("No prompt history matches the selected filters.")
            else:
                st.dataframe(filtered_history_rows, use_container_width=True)


with info_tab:
    st.header("Project Info")
    st.write(
        "LLM Metrics Dashboard is a portfolio-friendly Streamlit app for exploring "
        "how prompt runs can be tracked, logged, and summarized. It records mock "
        "or real LLM responses, token usage, estimated cost, and latency in a "
        "local SQLite database."
    )
    st.write(
        "Mock Mode is the default safe mode and does not require an API key. "
        "Real API Mode is optional and only runs when selected and a valid "
        "OPENAI_API_KEY is available."
    )

    st.subheader("Technologies")
    st.markdown(
        """
        - Python
        - Streamlit
        - SQLite
        - Matplotlib
        - Environment variables
        - Modular project structure
        """
    )

    st.divider()
    st.caption(f"Built with Streamlit | Version {APP_VERSION}")
