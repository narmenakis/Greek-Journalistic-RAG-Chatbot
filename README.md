# Greek-Journalistic-RAG-Chatbot
Open-source RAG chatbot for Greek newsrooms using Llama-KriKri-8B-Instruct, Chroma, and multilingual embeddings to provide source-grounded question answering over Greek news archives.

````markdown
## Installation and Usage

### 1. Create and activate a virtual environment

It is recommended to run the project inside a virtual environment.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
````

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

### 3. Create the vector database

Before running the application, create the vector database:

```bash
python3 create_vector_db.py
```

On Windows, use:

```bash
python create_vector_db.py
```

### 4. Run the application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in a web browser. If it does not open automatically, copy the local URL displayed in the terminal and paste it into your browser.

````

```markdown
## Notes

- Make sure the required dataset files are placed in the expected directory before creating the vector database.
- A GPU is recommended for local LLM inference, although the application may run on CPU with reduced performance.
- The first execution may take longer because models and resources need to be loaded and cached.
````
