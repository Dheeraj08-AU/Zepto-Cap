# Zepto Capstone — End-to-End E-Commerce & GenAI Platform

A unified platform combining automated data engineering, predictive machine learning, and an offline-first GenAI customer support service. Built to handle real-world quick-commerce workflows—from catalog database operations and customer analytics to routing complex policy queries using grounded RAG pipelines.

---

## Project Structure

```text
Zepto-Cap/
├── data_pipeline/
│   ├── pipeline.ipynb        # Data cleaning, ETL execution & database setup
│   ├── zepto_catalogue.db    # SQLite relational product catalogue
│   ├── requirements.txt      # Module dependencies
│   └── README.md             # Data pipeline guide
├── analytics/
│   ├── 01_eda.ipynb          # Exploratory data analysis & feature insights
│   ├── 02_modeling.ipynb     # Preprocessing, SMOTE, grid search, model evaluation
│   ├── best_pipeline.pkl     # Exported Random Forest model pipeline
│   ├── titanic.csv           # Baseline dataset
│   ├── requirements.txt      # Analytics dependencies
│   └── README.md             # Analytics module guide
├── support_assistant/
│   ├── docs/                 # 8 Zepto policy text documents (doc_01 to doc_08)
│   ├── chroma_db/            # Local vector storage
│   ├── ingest.py             # Document embedding & vector indexing script
│   ├── main.py               # LangGraph workflow & FastAPI application
│   ├── Dockerfile            # Container configuration
│   ├── requirements.txt      # Support Assistant dependencies
│   └── README.md             # GenAI service guide
└── README.md                 # Root repository overview

Architecture & Modules
1. Data Pipeline (/data_pipeline)
Cleans and transforms raw catalog data, creating a structured relational SQLite database (zepto_catalogue.db). Ensures data integrity, proper type casting, and schema enforcement across product listings.

2. Predictive Analytics (/analytics)
Explores underlying patterns in user data and constructs leak-free machine learning pipelines using ColumnTransformer and scikit-learn Pipeline objects. Implements SMOTE for class imbalance, runs GridSearchCV for hyperparameter optimization, and exports the tuned Random Forest classifier (best_pipeline.pkl).

3. GenAI Support Assistant (/support_assistant)
-> Builds a Retrieval-Augmented Generation (RAG) assistant for Zepto's policy documentation.
-> Ingestion & Embedding: Embeds policy text files locally using all-MiniLM-L6-v2 and indexes them in ChromaDB.
-> Orchestration: Employs a 3-node LangGraph state graph (classify_intent, retrieve_and_answer, direct_answer) to intelligently route incoming queries based on intent.
-> Guaranteed Output: Enforces strict Pydantic JSON schemas with response retries on validation errors.
-> Offline & Cloud Ready: Operates deterministically offline by default (MOCK_LLM=1) without requiring API keys, while supporting live LLM providers when toggled (MOCK_LLM=0).
-> Deployment: Served via FastAPI and fully containerized with Docker.
