# 🔬 Public Health Data Insights Agent (HIA)

AI-powered analysis tool for public health datasets. Upload your data, get instant insights, interactive visualizations, and follow-up Q&A.

Built by **Brook Eshete, MD, MPH** — Johns Hopkins Bloomberg School of Public Health.

## Features

- **📊 Auto-Analysis** — Upload CSV/Excel → instant overview, statistics, column type detection, health indicator identification
- **🧠 AI-Powered Insights** — Trends, disparities, correlations, and public health recommendations via local AI
- **📈 Interactive Visualizations** — Auto-generated Plotly charts (histograms, bar charts, correlation heatmaps, custom scatter plots)
- **💬 Follow-up Chat** — Ask questions about your data, get data-driven answers
- **📄 Export Reports** — Download Markdown or HTML/PDF analysis reports

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Dr-Brook/public-health-hia.git
cd public-health-hia
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## AI Setup

The app uses **Ollama** for local AI (no API keys needed):

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull glm-5.1:cloud

# Start Ollama
ollama serve
```

## Sample Dataset

A built-in sample dataset is included with US regional public health indicators (2018-2022):
- Region, Year, Population, Diabetes Rate, Obesity Rate, Income Level, Insurance Coverage, Life Expectancy, Hospital Beds

Click "Load Sample Dataset" in the sidebar to try it instantly.

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select this repo → Deploy
5. Live at `your-app.streamlit.app`

**Note:** AI features require Ollama, which isn't available on Streamlit Cloud. For cloud deployment with AI, you'd need to configure an OpenAI or Groq API key in `.env`.

## Tech Stack

- **Frontend:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **AI:** Ollama (glm-5.1:cloud) via OpenAI-compatible API
- **RAG:** FAISS + sentence-transformers
- **ML:** scikit-learn, scipy

## License

MIT