# AWS SBG Tasks & Projects Portfolio

A structured repository tracking weekly tasks, cloud exercises, and data-processing projects completed as part of the AWS SBG program.

---

## Repository Structure

```text
aws-sbg-tasks/
├── README.md                   # Portfolio overview & roadmap
├── requirements.txt            # Shared project dependencies
├── .gitignore                  # Git ignore rules for environments and cache
│
├── week-01/                    # Week 01: Small Python Data-Processing App
│   ├── README.md               # Detailed task documentation & findings
│   ├── data_processing_app.py  # Standalone CLI data processing script
│   ├── data_processing_app.ipynb # Interactive Jupyter & Google Colab notebook
│   └── data/
│       └── StudentsPerformance.csv # Raw dataset (1,000 exam records)
│
└── week-02/                    # Week 02: Mini Semantic Search Engine
    ├── README.md               # Detailed documentation & query examples
    ├── requirements.txt        # Week 02 dependencies (chromadb, pandas, numpy)
    ├── task.py                 # Standalone CLI semantic search engine
    ├── task.ipynb              # Interactive Jupyter & Google Colab notebook
    └── data/
        ├── faqs.json           # AASTU Knowledge Base (JSON)
        └── faqs.csv            # AASTU Knowledge Base (CSV)
```

---

## Weekly Roadmap

| Week | Focus Area / Task | Core Technologies | Deliverables | Status |
| :---: | :--- | :--- | :--- | :---: |
| **01** | **Small Python Data-Processing App** | Python, NumPy, Jupyter, Colab | [Script](week-01/data_processing_app.py) • [Notebook](week-01/data_processing_app.ipynb) • [Docs](week-01/README.md) | ✅ Completed |
| **02** | **Mini Semantic Search Engine** | Pandas, ChromaDB, Cosine Similarity, Embeddings | [Script](week-02/task.py) • [Notebook](week-02/task.ipynb) • [Docs](week-02/README.md) | ✅ Completed |

---

## Environment Setup

### 1. Clone & Activate Virtual Environment
```powershell
git clone https://github.com/AbellTT/aws-sbg-tasks.git
cd aws-sbg-tasks

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.venv\Scripts\Activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
pip install -r week-02/requirements.txt
```

---

## Running Tasks

### Week 01: Data-Processing App
```powershell
# Run standalone script
python week-01/data_processing_app.py

# Or launch notebook
jupyter lab
```

### Week 02: Mini Semantic Search Engine
```powershell
# Run semantic search script with demo queries
python week-02/task.py

# Query with custom text
python week-02/task.py --query "How do I connect to campus wifi?"

# Launch notebook
jupyter notebook week-02/task.ipynb
```

