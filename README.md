# Video-Classifier

## 1. Setup
### 1.1. Clone repository
```
git clone https://github.com/rht27/video-classifier.git
```
### 1.2. Initialize project
- ### [uv](https://docs.astral.sh/uv/)
    ```bash
    uv sync
    ```

- ### venv (this repo requires Python >= 3.12)
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## 2. Annotation tool
```bash
uv run streamlit run tool/annotation.py

# with optional arguments
uv run streamlit run tool/annotation.py -- -v VIDEO_DIR_PATH -l LABEL_CSV_PATH

# without uv
streamlit run tool/annotation.py -- -v VIDEO_DIR_PATH -l LABEL_CSV_PATH
```
