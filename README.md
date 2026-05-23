# LLM Metrics Dashboard

A Streamlit dashboard for running mock LLM prompts and tracking useful metrics such as latency, estimated token usage, estimated cost, prompt history, and model usage.

This project currently runs in **Mock Mode**. It does **not** make real OpenAI API calls yet.

## Why I Built This

I built this project to practice the core pieces of an LLM observability workflow: running prompts, collecting metrics, saving logs, and visualizing usage over time.

The goal was to create a realistic dashboard experience without requiring paid API usage during development. Mock Mode lets the app behave like an LLM metrics tool while keeping the project easy to run locally.

## Features

- Prompt runner with model selection and temperature control
- Mock LLM responses for local testing
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

Mock Mode is the current operating mode for this project.

In Mock Mode:

- No real OpenAI API calls are made.
- No API key is required to test the app.
- Responses are generated locally from mock response templates.
- Token counts and costs are estimates, not real API billing data.

This keeps the project simple to run and safe to test while the dashboard features are being developed.

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
- Mock response
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

- Real OpenAI API integration
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
- Thinking through how LLM usage metrics can be tracked before adding real API calls

The project is intentionally kept honest and lightweight. It is not production-ready, but it demonstrates the foundation of an LLM metrics dashboard that can be extended later.
