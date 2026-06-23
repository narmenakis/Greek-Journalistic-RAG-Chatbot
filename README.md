# Greek-Journalistic-RAG-Chatbot

Open-source RAG chatbot for Greek newsrooms using Llama-KriKri-8B-Instruct, Chroma, and multilingual embeddings to provide source-grounded question answering over Greek news archives.

## Overview

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) chatbot designed for Greek newsroom use. The system supports source-grounded question answering by combining semantic retrieval, metadata filtering, two-stage query expansion, and local inference using open-source language models.

The application is designed to help journalists and researchers retrieve information from Greek news archives, generate fact-based answers, and inspect the sources used by the system.

## Dataset

The full evaluation dataset contains approximately **73,000 articles** collected from four Greek news websites:

- kathimerini.gr
- efsyn.gr
- skai.gr
- zougla.gr

The dataset is **not included** in the repository due to size and licensing restrictions. You must scrape the articles yourself using the provided scraper (see step 3 below).

The repository includes a lightweight sample dataset containing **200 Greek news articles** for quick testing of the vector database creation and the application without scraping the full corpus.

> Note: Results produced using the sample dataset may differ from those reported in the evaluation study, which used the complete corpus of approximately 73K news articles.

## Installation and Usage

### 1. Create and activate a virtual environment

It is recommended to run the project inside a virtual environment.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

If using Anaconda on macOS, activate the base environment with:

```bash
source /opt/anaconda3/bin/activate
```

### 2. Install dependencies

After activating the environment, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Scrape the article dataset

Collect article URLs and save them in `links.txt` (one URL per line), then run the scraper:

```bash
python3 scraper.py
```

This will generate `new_sample_dataset.csv` containing the scraped articles. To use the full 73K-article dataset, populate `links.txt` with the complete list of URLs from the four news websites.

To skip scraping and use the included 200-article sample dataset instead, proceed directly to step 4.

### 4. Create the vector database

Before running the application, create the vector database:

```bash
python3 create_vector_db.py
```

On Windows, use:

```bash
python create_vector_db.py
```

### 5. Run the application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in a web browser. If it does not open automatically, copy the local URL displayed in the terminal and paste it into your browser.

## Notes

- Make sure the required dataset CSV file is placed in the expected directory before creating the vector database. Either use the included sample dataset or scrape your own by running `scraper.py`.
- A GPU is recommended for local LLM inference, although the application may run on CPU with reduced performance.
- The first execution may take longer because models and resources need to be loaded and cached.
- The included sample dataset is intended for demonstration and testing. Full-scale evaluation requires the complete article corpus.

## Example Queries

The following example questions can be used to explore the chatbot’s retrieval and answer-generation capabilities.

---

### Example 1 – Fact Retrieval

**Greek**

> Πόσοι άνθρωποι έχασαν τη ζωή τους στη σύγκρουση των τρένων στα Τέμπη και ποια ήταν η αιτία σύμφωνα με τα επίσημα στοιχεία;

**English**

> How many people lost their lives in the Tempi train collision and what was identified as the cause according to official reports?

---

### Example 2 – Information Synthesis

**Greek**

> Ποιες ήταν οι σημαντικότερες αδυναμίες στο σιδηροδρομικό δίκτυο που αναδείχθηκαν μετά το δυστύχημα των Τεμπών και τι μέτρα ανακοινώθηκαν για την πρόληψη παρόμοιων περιστατικών;

**English**

> What were the most significant weaknesses in the railway network highlighted after the Tempi accident, and what measures were announced to prevent similar incidents?

---

### Example 3 – Reasoning / Counterfactual Analysis

**Greek**

> Εάν οι προειδοποιήσεις των εργαζομένων για την ασφάλεια είχαν εισακουστεί εγκαίρως, και εάν είχαν εγκατασταθεί τα συστήματα τηλεδιοίκησης, τότε θα μπορούσε να αποφευχθεί το δυστύχημα;

**English**

> If the workers’ safety warnings had been taken into account in time, and if remote traffic control systems had been installed, could the accident have been prevented?

---

> Note: Since the repository includes a 200-article sample dataset, the answers generated during local testing may differ from those obtained using the complete evaluation corpus.
