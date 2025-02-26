import os
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ---------------------------- CONFIGURATION --------------------------------
SPACY_MODEL = "fr_core_news_sm"
DATA_DIR = "data/"
OUTPUT_FILE = "data_analysis/clustering_results.csv"
NUM_CLUSTERS = 5  # Ajustable selon la taille du corpus
TSNE_PERPLEXITY = 5  # Valeur minimale pour éviter les erreurs
TSNE_ITERATIONS = 300
TFIDF_MAX_FEATURES = 1000


# ---------------------------- CHARGEMENT DU MODÈLE SPACY -------------------
def load_spacy_model(model_name: str) -> spacy.language.Language:
    """Charge le modèle linguistique SpaCy avec une gestion des erreurs."""
    try:
        nlp = spacy.load(model_name)
        nlp.max_length = 2_000_000  # Augmentation de la limite pour éviter les erreurs de taille
        return nlp
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle SpaCy : {e}")


nlp = load_spacy_model(SPACY_MODEL)


# ---------------------------- CHARGEMENT DES TEXTES ------------------------
def load_text_files(directory: str) -> Dict[str, str]:
    """Charge tous les fichiers .txt d'un répertoire et ses sous-dossiers, en excluant les logs."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le dossier '{directory}' n'existe pas. Vérifiez le chemin.")

    text_data = {}
    files_found = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") and "log" not in file.lower():
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text_data[file_path] = f.read()
                    files_found += 1

    print(f"✅ Nombre de fichiers trouvés (hors logs) : {files_found}")
    return text_data


corpus = load_text_files(DATA_DIR)
print(f"📂 Nombre de documents chargés : {len(corpus)}")


# ---------------------------- PRÉTRAITEMENT DES TEXTES ---------------------
def preprocess_texts(corpus: Dict[str, str]) -> List[str]:
    """Prétraite les textes : suppression des stopwords et lemmatisation."""
    preprocessed_texts = []

    for text in tqdm(corpus.values(), desc="🔄 Prétraitement des documents", unit="doc"):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        preprocessed_texts.append(" ".join(tokens))

    return preprocessed_texts


preprocessed_texts = preprocess_texts(corpus)


# ---------------------------- VECTORISATION TF-IDF -------------------------
def vectorize_texts(texts: List[str], max_features: int) -> np.ndarray:
    """Vectorise les textes avec TF-IDF et limite le nombre de features."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform(texts)


X = vectorize_texts(preprocessed_texts, TFIDF_MAX_FEATURES)


# ---------------------------- CLUSTERING K-MEANS ---------------------------
def cluster_texts(X: np.ndarray, num_clusters: int) -> np.ndarray:
    """Applique K-Means sur les données vectorisées."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


clusters = cluster_texts(X, NUM_CLUSTERS)


# ---------------------------- RÉDUCTION DE DIMENSION T-SNE ----------------
def reduce_dimensions(X: np.ndarray, perplexity: int, iterations: int) -> np.ndarray:
    """Réduit les dimensions du corpus avec t-SNE pour la visualisation."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=iterations)
    return tsne.fit_transform(X.toarray())


X_2d = reduce_dimensions(X, TSNE_PERPLEXITY, TSNE_ITERATIONS)


# ---------------------------- VISUALISATION DES CLUSTERS -------------------
def plot_clusters(X_2d: np.ndarray, clusters: np.ndarray, corpus: Dict[str, str], num_clusters: int):
    """Affiche la visualisation des clusters en 2D avec t-SNE."""
    colors = plt.colormaps["tab10"]

    plt.figure(figsize=(12, 8))

    # Ajout des points par cluster
    for i in range(num_clusters):
        cluster_points = X_2d[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f"Cluster {i}", color=colors(i % colors.N), alpha=0.7)

    # Ajout des noms des documents sur le graphe
    for i, txt in enumerate(corpus.keys()):
        plt.text(X_2d[i, 0], X_2d[i, 1], txt.split("/")[-1], fontsize=8, alpha=0.6)

    plt.title("Proximité entre les documents (t-SNE + K-Means)")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.legend()
    plt.show()


plot_clusters(X_2d, clusters, corpus, NUM_CLUSTERS)


# ---------------------------- ENREGISTREMENT DES RÉSULTATS -----------------
def save_results(corpus: Dict[str, str], clusters: np.ndarray, output_file: str):
    """Enregistre les résultats du clustering dans un fichier CSV."""
    df_results = pd.DataFrame({
        "Document": list(corpus.keys()),
        "Cluster": clusters
    })
    df_results.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n📂 Clustering et visualisation enregistrés dans '{output_file}'.")


save_results(corpus, clusters, OUTPUT_FILE)