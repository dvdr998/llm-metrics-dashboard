# LLM Metrics Dashboard with GPU-Aware Local LLM Inference Profiler

Live App: https://llm-metrics-dashboard-dvdr998.streamlit.app/

GitHub Repository: https://github.com/dvdr998/llm-metrics-dashboard

## Overview

LLM Metrics Dashboard is a Streamlit-based AI observability project for tracking how LLM prompts behave across API-based inference and local model inference.

The project started as a simple dashboard to log prompts, latency, token usage, estimated cost, model usage, and prompt history. I later expanded it into a more transparent LLM profiling tool by adding optional OpenAI API support, accurate token estimation with `tiktoken`, SQLite logging, visual analytics, and a GPU-aware local LLM inference profiler using Hugging Face Transformers and PyTorch.

The main goal of this project is not just to generate text. The goal is to make LLM behavior easier to understand by showing useful metrics such as:

- Latency
- Input tokens
- Output tokens
- Total tokens
- Estimated cost
- Tokens per second
- Prompt history
- Model usage
- Local RAM usage
- Local GPU or backend information
- Device-level inference behavior

This project is built as a practical portfolio project for learning LLM observability, prompt tracking, local inference profiling, and AI dashboard development.

---

## Why I Built This

When developers experiment with LLM prompts, it is easy to lose track of important details such as:

- Which prompts were tested
- Which model was used
- How long the response took
- How many tokens were used
- How much the run may cost
- How responses changed over time
- Whether the prompt was tested in mock mode or real API mode
- How local inference affects system memory and hardware resources

This dashboard solves that by giving a simple interface to run prompts, store results, and visualize the behavior of LLM requests over time.

I also wanted to understand the difference between cloud API inference and local LLM inference:

- API-based inference is usually higher quality and easier to use, but it has usage cost and depends on cloud services.
- Local inference avoids API cost and can run on personal hardware, but performance depends heavily on the device, memory, backend, and model size.

This project helped me explore both sides in one dashboard.

---

## Main Features

### 1. Prompt Runner

The Prompt Runner allows users to enter a prompt and run it through the dashboard.

It supports:

- Model selection
- Temperature control
- Mock Mode
- Optional Real API Mode
- Response display
- Latency tracking
- Token estimation
- Estimated cost calculation
- Tokens per second calculation
- Prompt logging into SQLite

---

## Operating Modes

### Mock Mode

Mock Mode is the default safe mode.

In Mock Mode:

- No real OpenAI API call is made
- No API key is required
- The app generates a mock response locally
- Token usage is estimated
- Cost is estimated
- The dashboard can be tested safely without spending API credits

Mock Mode is useful for testing the dashboard, UI, database logging, and analytics workflow without depending on paid API usage.

---

### Real API Mode

Real API Mode is optional.

When enabled, the app uses the OpenAI Python SDK and the OpenAI Responses API to send the prompt to a selected OpenAI model.

In Real API Mode, the app tracks:

- Real response latency
- Estimated input tokens
- Estimated output tokens
- Total tokens
- Estimated cost
- Tokens per second
- Model used
- Prompt and response history

The app looks for the OpenAI API key from:

1. Streamlit secrets
2. Local environment variables

The API key is never committed to GitHub.

---

## Token Counting Improvements

I added `tiktoken` to improve token counting accuracy.

The dashboard separates:

- Plain prompt token estimates
- API-style billed token estimates

This is important because OpenAI chat-style or response-style APIs may include formatting overhead, system/message structure, or additional metadata. That means a simple prompt like `hi` can show more billed tokens than just the visible text.

To avoid confusion, I added helper notes in the UI so users understand why short prompts can still produce higher billed token counts.

---

## Metrics Tracked

Each prompt run can track and store:

- Timestamp
- Prompt text
- Response text
- Model name
- Temperature
- Mode used
- Latency
- Input tokens
- Output tokens
- Total tokens
- Tokens per second
- Estimated input cost
- Estimated output cost
- Estimated total cost

These metrics are saved into a local SQLite database and later used for dashboards, charts, filtering, and history review.

---

## Dashboard Metrics

The dashboard includes summary cards for:

- Total prompt runs
- Average latency
- Total tokens used
- Total estimated cost

It also includes charts for:

- Latency trends
- Estimated cost trends
- Token usage trends
- Prompt runs by model

These charts help visualize how prompts behave over time.

---

## Prompt History

The Prompt History page allows users to review previous prompt runs.

It includes:

- Stored prompt logs
- Prompt text
- Model used
- Response
- Latency
- Token usage
- Estimated cost
- Timestamp
- Search and filtering
- CSV export
- Clear history option

This makes the dashboard useful as a lightweight LLM experiment tracker.

---

## GPU-Aware Local LLM Inference Profiler

I added a second major feature: a GPU-aware local LLM inference profiler.

This page allows users to run a small local LLM using:

- Hugging Face Transformers
- PyTorch
- Local machine resources

The profiler automatically detects the best available backend:

- CUDA, if available
- Apple Silicon MPS, if available
- CPU, if no GPU backend is available

During local inference, the app shows:

- Device type
- GPU or backend name
- RAM used
- Total RAM
- MPS memory usage, when available
- Before-inference memory snapshot
- After-inference memory snapshot
- Input tokens
- Output tokens
- Total tokens
- Latency
- Tokens per second

This helps users understand how local LLM inference affects their machine.

For example, on an Apple Silicon Mac, the profiler can detect MPS and show memory usage during inference. On Streamlit Cloud, the app may show CPU because hosted cloud environments usually do not expose Apple MPS or a local GPU.

---

## OpenAI API vs Local LLM Comparison

This project is also moving toward comparing OpenAI API inference and local LLM inference more transparently.

The comparison idea is:

### API-Based Inference

- Usually better response quality
- Cloud-dependent
- Requires API key
- Has usage cost
- Easier to run large models
- Less hardware burden on the user device

### Local LLM Inference

- No API cost
- Can run locally
- Hardware-dependent
- Smaller models may produce lower-quality responses
- Performance depends on RAM, GPU, backend, and model size
- Useful for learning local AI deployment and profiling

The purpose is to help users understand not only the output of an LLM, but also the cost, speed, hardware impact, and deployment tradeoffs behind it.

---

## Database Logging

The app uses SQLite to store prompt history locally.

Database file:

```text
llm_metrics.db