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

## 2. Annotation tool
```bash
uv run streamlit run tool/annotation.py

# with optional arguments
uv run streamlit run tool/annotation.py -- -v VIDEO_DIR_PATH -l LABEL_CSV_PATH
```
