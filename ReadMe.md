# IFRESI corpus analysis

Analyse textométrique et distant viewing du fonds numérisé de l'IFRESI


# Description

[Présentation du projet à venir]


# Installation
1. Cloner ce dépôt :

`git clone`\
`cd nom-du-projet`

2. Créer un environnement virtuel (recommandé) :

`python -m venv venv`\
`source venv/bin/activate`

3. Installer les dépendances :

`pip install -r requirements.txt`

4. Télécharger les ressources nécessaires (modèle Spacy français (small), non entraîné) :

`python -m spacy download fr_core_news_sm`

# Fonctionnalités
## Script pdf2txt.py
- Script d'extraction de la couche texte d'un ensemble de fichiers PDF océrisés par l'ANRT
- Utilisation de PDFplumber pour l'extraction car l'OCR est de très bonne qualité
  - Possibilité de fonctionner avec Tesseract si l'OCR n'est pas satisfaisant
- Input: fichiers .pdf > output: fichiers .txt avec mirroir de l'arborescence

## Script textometry.py
- Script d'analyse textométrique à partir du corpus océrisé et de l'extraction au format texte brut
- Résultats: analyse de fréquences de mots, de richesse lexicale, N-grams (bi et tri-grams), wordcloud enregistrés dans ./data_analysis
## Script clustering_t-SNE.py
- Script de clustering à partir des méthodes K-Means et T-SNE (input: fichiers .txt)
- Résultats: visualisation sous forme d'un graphe de la proximité sémantique des documents et de leurs clusters
## Script NER.py
- Script de reconnaissance et extraction des entités nommées (input: fichiers .txt / output: CSV de fréquences décroissantes des entités nommées)
- Modèle: Spacy french small

# Structure du projet (v. 1.1)
```bash
.
├── clustering_tSNE_kMeans.py
├── data
│   ├── extraction_log.txt
│   ├── input
│   │   ├── A5
│   │   │   ├── IFRESI_029.pdf
│   │   │   ├── IFRESI_030.pdf
│   │   │   ├── IFRESI_031.pdf
│   │   │   ├── IFRESI_032.pdf
│   │   │   ├── IFRESI_033.pdf
│   │   │   └── IFRESI_034.pdf
│   │   ├── A6
│   │   │   ├── IFRESI_035.pdf
│   │   │   ├── IFRESI_036.pdf
│   │   │   ├── IFRESI_037.pdf
│   │   │   └── IFRESI_038.pdf
│   │   └── A8
│   │       └── IFRESI_040.pdf
│   └── output
│       ├── A5
│       │   ├── IFRESI_029.txt
│       │   ├── IFRESI_030.txt
│       │   ├── IFRESI_031.txt
│       │   ├── IFRESI_032.txt
│       │   ├── IFRESI_033.txt
│       │   └── IFRESI_034.txt
│       ├── A6
│       │   ├── IFRESI_035.txt
│       │   ├── IFRESI_036.txt
│       │   ├── IFRESI_037.txt
│       │   └── IFRESI_038.txt
│       └── A8
│           └── IFRESI_040.txt
├── data_analysis
│   ├── bigram_frequencies.csv
│   ├── clustering_results.csv
│   ├── entities_with_types_and_count.csv
│   ├── nuage_de_mots.png
│   ├── statistiques_corpus.csv
│   ├── trigram_frequencies.csv
│   └── word_frequencies.csv
├── NER.py
├── pdf2txt.py
├── ReadMe.md
├── requirements.txt
└── textometry.py


