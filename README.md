# LLM Metrics Dashboard

Live App: https://llm-metrics-dashboard-dvdr998.streamlit.app/

GitHub Repository: https://github.com/dvdr998/llm-metrics-dashboard

A Streamlit dashboard for running LLM prompts and tracking useful metrics such as latency, token usage, estimated cost, prompt history, and model usage.

The app supports both **Mock Mode** and optional **Real API Mode**. Mock Mode is the default safe mode and does not make real OpenAI API calls.

## Why I Built This

I built this project to practice the core pieces of an LLM observability workflow: running prompts, collecting metrics, saving logs, and visualizing usage over time.

The goal was to create a realistic dashboard experience without requiring paid API usage during development. Mock Mode lets the app behave like an LLM metrics tool while keeping the project easy to run locally, while Real API Mode can be enabled later for live OpenAI usage.

## Features

- Prompt runner with model selection and temperature control
- Mock LLM responses for local testing
- Optional Real API Mode using the OpenAI Responses API
- SQLite logging for every prompt run
- Prompt history table with filters
- Search prompt history by prompt text
- CSV export for filtered prompt history
- Clear prompt history option
- Summary metric cards for total runs, average latency, tokens, and cost
- Matplotlib dashboard charts for:
  - Latency trends
  - Estimated cost trends
  - Estimated token usage
  - Prompt runs by model
- Sidebar with app status, mode, database status, and run count
- Tabbed UI for cleaner navigation

## Tech Stack

- Python
- Streamlit
- OpenAI Python SDK
- SQLite
- Matplotlib
- python-dotenv
- Environment variables
- Modular project structure

## Project Structure

```text
llm-metrics-dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── llm_metrics.db      # Local SQLite database, created when the app runs
└── src/
    ├── __init__.py
    ├── config.py       # App configuration and environment settings
    ├── llm_client.py   # Mock LLM response logic
    ├── metrics.py      # Token and cost estimation helpers
    ├── database.py     # SQLite database helpers
    └── charts.py       # Dashboard data preparation helpers
```

## How To Run Locally

1. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in your terminal, usually:

   ```text
   http://localhost:8501
   ```

## Mock Mode

Mock Mode is the default operating mode for this project.

In Mock Mode:

- No real OpenAI API calls are made.
- No API key is required to test the app.
- Responses are generated locally from mock response templates.
- Token counts and costs are estimates, not real API billing data.

This keeps the project simple to run and safe to test while the dashboard features are being developed.

## Real API Mode

Real API Mode is optional. When selected in the sidebar, the app uses the OpenAI Python SDK and the Responses API to send the prompt to OpenAI.

The app looks for `OPENAI_API_KEY` in this order:

1. Streamlit secrets
2. Local environment variables

If no API key is found, the app shows a warning and does not call the API.

### Set `OPENAI_API_KEY` locally

Create a local `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is ignored by git and should never be committed.

### Set `OPENAI_API_KEY` on Streamlit Cloud

In Streamlit Cloud:

1. Open the app settings.
2. Go to **Secrets**.
3. Add:

   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

API keys are never displayed in the app and should not be committed to GitHub.

## Database Logging

The app uses SQLite to store prompt history locally.

Database file:

```text
llm_metrics.db
```

Main table:

```text
prompt_logs
```

Each saved prompt run includes:

- Timestamp
- Prompt
- Mock or real response
- Model
- Temperature
- Latency
- Input tokens
- Output tokens
- Total tokens
- Estimated input cost
- Estimated output cost
- Estimated total cost
- Mode

The dashboard uses this saved history to track latency, tokens, estimated cost, and model usage over time.

## Screenshots

Screenshots should be added after running the app locally with:

```bash
streamlit run app.py
```

Suggested screenshot files:

![Dashboard overview](screenshots/dashboard-overview.png)

![Prompt runner](screenshots/prompt-runner.png)

![Prompt history](screenshots/prompt-history.png)

![Metrics charts](screenshots/metrics-charts.png)

## Future Improvements

- More robust OpenAI model configuration
- User authentication
- Better token counting with tiktoken
- More advanced analytics
- Deployment on Streamlit Cloud

## What I Learned

This project helped me practice:

- Building a multi-tab Streamlit app
- Designing a cleaner dashboard-style UI
- Saving and reading data with SQLite
- Structuring a Python project into reusable modules
- Creating charts with Matplotlib
- Exporting filtered data as CSV
- Thinking through how LLM usage metrics can be tracked in both mock and optional real API workflows

The project is intentionally kept honest and lightweight. It is not production-ready, but it demonstrates the foundation of an LLM metrics dashboard that can be extended later.

## License and Usage

Copyright (c) 2026 Vamshidhar Reddy Devulapally. All rights reserved.

This project is shared publicly as a portfolio and learning project. You may view the code for review purposes, but you may not copy, modify, distribute, or reuse this project without written permission from the author.

