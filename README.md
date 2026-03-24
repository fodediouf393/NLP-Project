# ASRS Extractive Summarization

Application NLP de **résumé extractif** sur des rapports d’incidents aéronautiques **ASRS**.  
Le projet implémente et compare deux approches extractives pour la **Task 1** :

- **Top-k sentence selection**
- **TextRank**

L’objectif est de transformer un **narrative** long en un **synopsis** court, fidèle et exploitable.

---

## Features

- Résumé extractif **Top-k**
- Résumé extractif **TextRank**
- Évaluation sur base de test avec :
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L
  - runway coverage
  - altitude coverage
  - airport code coverage
  - compression ratio mean
  - latency mean
- Application **Streamlit** avec :
  - exemples réels du dataset
  - synopsis de référence
  - résumés prédits par Top-k et TextRank
  - démo interactive
  - affichage des métriques

---

## Project structure

```text
NLP-Project/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── logs/
├── outputs/
│   ├── predictions/
│   └── reports/
├── scripts/
│   └── download_data.py
├── src/
│   └── asrs_sum/
│       ├── core/
│       │   ├── preprocessing.py
│       │   ├── sentence_splitter.py
│       │   ├── sentence_ranker.py
│       │   ├── topk_summarizer.py
│       │   └── textrank_summarizer.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── evaluate.py
│       ├── pipeline/
│       │   ├── predict.py
│       │   └── batch_predict.py
│       └── utils/
│           ├── io.py
│           └── logger.py
├── tests/
├── requirements.txt
└── README.md