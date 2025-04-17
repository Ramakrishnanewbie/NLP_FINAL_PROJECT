# Multi-Label Petition Classification Application

This Streamlit application allows users to classify petition texts into EUROVOC categories using various NLP models, including traditional machine learning approaches and deep learning techniques.

## Installation

1. Clone this repository:
```
git clone github.com/Ramakrishnanewbie/NLP_FINAL_PROJECT
cd petition-classification
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Export your trained models from your Jupyter notebook (Most important, otherwise streamlit won't run):

* Github doesn't allow us toa dd pkl and pt files where the weights are saved due to storage limit. So, do make sure to update these to make sure the UI works. 

```python
# Save the TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Save the MultiLabelBinarizer
joblib.dump(mlb, "multilabel_binarizer.pkl")

# Save traditional models
joblib.dump(model_nb, "naive_bayes_model.pkl")
joblib.dump(model_pa, "passive_aggressive_model.pkl")

# Save BERT models
torch.save(model_bert_gru, "bert_gru_model.pt")
torch.save(model_bilstm, "bert_bilstm_model.pt")
```

4. Ensure all model files are in the same directory as the application.

## Usage

1. Run the Streamlit application:
```
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Enter petition text or load a sample

4. Select a model from the sidebar

5. Click "Classify Petition" to see results

## Example Model Metrics

Based on the EURLEX dataset:

| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Naive Bayes | 0.290 | 0.833 | 0.176 |
| Passive Aggressive | 0.688 | 0.765 | 0.625 |
| BERT+GRU | 0.529 | 0.860 | 0.382 |
| BERT+BiLSTM | 0.298 | 0.855 | 0.181 |


