# AI vs Human Hybrid Model Streamlit App

This project is a Streamlit web app for analyzing and classifying text as AI-generated, human-written, or hybrid, using statistical, linguistic, and embedding-based models.

## Features

- Upload or paste text for analysis
- Hybrid ensemble of three models
- Downloadable and copyable reports
- Fast, production-ready deployment

## Project Structure

- `production_streamlit.py` — Main Streamlit app
- `production_models/` — Pretrained model files (joblib)
- `requirements.txt` — Python dependencies
- `.gitignore` — Standard ignores

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run production_streamlit.py
```

## Deployment

- Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) or Heroku for global access.
- All model files are included for easy deployment (total <20MB).

## Model Training

If you need to retrain models, use your own scripts and place the resulting `.joblib` files in `production_models/`.

## Credits

- Built with Streamlit, scikit-learn, spaCy, PyTorch, and Hugging Face Transformers.

## License

This project is licensed under the [Creative Commons BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

- **You are free to:**

  - View, share, and adapt this work for non-commercial purposes.
  - Use the code, data, and research for personal projects, academic papers, and learning.

- **You must:**

  - Provide proper attribution.
  - Share any adaptations under the same license terms.

- **You may not:**
  - Use this project, its code, or its datasets in a commercial (paid) product or service without explicit written permission.

For full license details, see the [LICENSE](LICENSE) file.
