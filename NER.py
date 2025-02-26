import os
import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
from typing import Dict, Tuple, List

# ---------------------------- CONFIGURATION --------------------------------
SPACY_MODEL = "fr_core_news_sm"  # Modèle linguistique pour le traitement en français
DATA_DIR = "data/"
OUTPUT_FILE = "data_analysis/entities_with_types_and_count.csv"


# ---------------------------- CHARGEMENT DU MODÈLE SPACY --------------------
def load_spacy_model(model_name: str) -> spacy.language.Language:
    """Charge le modèle SpaCy et gère les erreurs potentielles."""
    try:
        nlp = spacy.load(model_name)
        nlp.max_length = 2_000_000  # Augmentation de la limite pour éviter les erreurs
        return nlp
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle SpaCy : {e}")


nlp = load_spacy_model(SPACY_MODEL)


# ---------------------------- CHARGEMENT DES TEXTES ------------------------
def load_text_files(directory: str) -> Dict[str, str]:
    """Charge les fichiers .txt d'un répertoire et ses sous-dossiers, en excluant les logs."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le dossier '{directory}' n'existe pas. Vérifiez le chemin.")

    text_data = {}
    files_found = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") and "log" not in file.lower():  # Exclure les fichiers logs
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text_data[file_path] = f.read()
                    files_found += 1

    print(f"✅ Nombre de fichiers trouvés (hors logs) : {files_found}")
    return text_data


# ---------------------------- EXTRACTION DES ENTITÉS NOMMÉES --------------
def extract_named_entities(corpus: Dict[str, str]) -> Counter:
    """Extrait les entités nommées du corpus et les compte."""
    entity_data = Counter()

    for filename, text in tqdm(corpus.items(), desc="Extraction des entités nommées", unit="doc"):
        doc = nlp(text)
        for ent in doc.ents:
            # Utilisation d'une clé combinée pour éviter les doublons
            entity_data[(ent.text, ent.label_)] += 1

    return entity_data


# ---------------------------- ENREGISTREMENT DES RÉSULTATS ----------------
def save_results(entity_data: Counter, output_file: str):
    """Enregistre les entités extraites et leur fréquence dans un fichier CSV."""
    # Classement des entités par fréquence
    sorted_entities = entity_data.most_common()

    # Préparation des données pour le DataFrame
    data_for_df = [(entity[0][0], entity[0][1], entity[1]) for entity in sorted_entities]

    # Enregistrement dans un fichier CSV
    df = pd.DataFrame(data_for_df, columns=["Entité", "Type d'entité", "Fréquence"])
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n📂 Résultats enregistrés dans '{output_file}'.")


# ---------------------------- EXÉCUTION PRINCIPALE ------------------------
def main():
    """Fonction principale pour charger les données, extraire les entités et enregistrer les résultats."""
    # Chargement du corpus
    corpus = load_text_files(DATA_DIR)
    print(f"📂 Nombre de documents chargés : {len(corpus)}")

    # Extraction des entités nommées
    entities = extract_named_entities(corpus)

    # Enregistrement des résultats
    save_results(entities, OUTPUT_FILE)

    # Affichage des 10 entités les plus fréquentes
    print("\n🔍 10 entités nommées les plus fréquentes avec leur type:")
    for entity in list(entities.most_common())[:10]:
        print(f"{entity[0][0]} ({entity[0][1]}): {entity[1]} fois")


# ---------------------------- EXÉCUTION DU SCRIPT -------------------------
if __name__ == "__main__":
    main()
