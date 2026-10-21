# Week 01: Small Python Data-Processing App

## Objective
Build a Python data-processing application and notebook that:
1. Loads a raw dataset (`data/StudentsPerformance.csv`)
2. Cleans and validates data using modular Python functions
3. Processes it using NumPy vectorized operations and statistics
4. Runs interactively in Jupyter / Google Colab
5. Uses a virtual environment (`.venv`) and `requirements.txt`

---

## Deliverables

```text
week-01/
├── README.md                   # Week 01 documentation & run guide
├── data_processing_app.py      # Standalone Python script with CLI runner
├── data_processing_app.ipynb   # Interactive Jupyter / Google Colab notebook
└── data/
    └── StudentsPerformance.csv # Raw exam performance dataset (1,000 records)
```

---

## How to Run

### 1. Run the CLI Script
From the repository root or the `week-01/` folder:

```powershell
# From project root
python week-01/data_processing_app.py

# Or inside week-01/
cd week-01
python data_processing_app.py
```

### 2. Run the Jupyter Notebook
Open `week-01/data_processing_app.ipynb` in VS Code or JupyterLab, select your Python kernel, and run all cells.

### 3. Run in Google Colab
Upload `week-01/data_processing_app.ipynb` to [Google Colab](https://colab.research.google.com/). The notebook automatically retrieves the dataset from the GitHub repository if not present locally.

---

## Key Results
- **Valid records processed:** 1,000 / 1,000 (0 dropped)
- **Passing rate ($\ge 60$ in Math, Reading, and Writing):** 60.3%
- **Top 10% cutoff score:** 86.00 average
- **Test prep course impact:** Students with test preparation scored on average **+7.63 points** higher.
