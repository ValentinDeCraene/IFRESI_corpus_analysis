import os
import spacy
import pandas as pd
from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Chargement du modèle SpaCy (Français)
nlp = spacy.load("fr_core_news_sm")
nlp.max_length = 2_000_000  # Évite les erreurs sur les grands fichiers


def load_text_files(directory):
    """Charge les fichiers .txt d'un répertoire (hors logs)."""
    text_data = {}

    if not os.path.exists(directory):
        raise FileNotFoundError(f"❌ Le dossier '{directory}' n'existe pas.")

    files_found = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") and "log" not in file.lower():
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text_data[file_path] = f.read()
                files_found += 1

    print(f"📂 {files_found} fichiers chargés depuis '{directory}'.")
    return text_data


def analyze_text_statistics(corpus, max_length=1_000_000):
    """Analyse un corpus et extrait les statistiques + fréquences de mots."""
    stats = []
    word_freq = Counter()
    bigram_freq = Counter()
    trigram_freq = Counter()

    for filename, text in tqdm(corpus.items(), desc="📊 Analyse des documents", unit="doc"):
        num_words, num_chars, num_sents = 0, 0, 0
        unique_lemmas = set()

        for i in range(0, len(text), max_length):
            segment = text[i:i + max_length]
            doc = nlp(segment)

            tokens = [token for token in doc if token.is_alpha and not token.is_stop]
            lemmatized_tokens = [token.lemma_ for token in tokens]

            num_words += len(tokens)
            num_chars += len(segment)
            num_sents += len(list(doc.sents))
            unique_lemmas.update(lemmatized_tokens)

            # Mise à jour des fréquences
            word_freq.update(lemmatized_tokens)
            bigram_freq.update(
                [f"{tokens[i].lemma_} {tokens[i + 1].lemma_}" for i in range(len(tokens) - 1)]
            )
            trigram_freq.update(
                [f"{tokens[i].lemma_} {tokens[i + 1].lemma_} {tokens[i + 2].lemma_}" for i in range(len(tokens) - 2)]
            )

        richness = len(unique_lemmas) / num_words if num_words > 0 else 0
        stats.append({
            "Document": filename,
            "Nombre de mots": num_words,
            "Nombre de caractères": num_chars,
            "Nombre de phrases": num_sents,
            "Richesse lexicale": richness
        })

    return pd.DataFrame(stats), word_freq, bigram_freq, trigram_freq


def save_csv(data, filename, output_dir):
    """Sauvegarde un DataFrame en CSV."""
    file_path = os.path.join(output_dir, filename)
    data.to_csv(file_path, index=False, encoding="utf-8")
    print(f"✅ Fichier enregistré : {file_path}")


def generate_wordcloud(word_freq, output_dir):
    """Génère et enregistre un nuage de mots."""
    if not word_freq:
        print("⚠️ Aucune donnée pour générer un nuage de mots.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white') \
        .generate_from_frequencies(dict(word_freq))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nuage de mots du corpus")

    wordcloud_path = os.path.join(output_dir, "nuage_de_mots.png")
    plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Nuage de mots enregistré : {wordcloud_path}")


if __name__ == "__main__":
    # 📂 Définition des chemins
    input_dir = "data"
    output_dir = "data_analysis"
    os.makedirs(output_dir, exist_ok=True)  # Création du dossier si nécessaire

    # 📥 Chargement du corpus
    corpus = load_text_files(input_dir)

    # 📊 Analyse du corpus
    stats_df, word_freq, bigram_freq, trigram_freq = analyze_text_statistics(corpus)

    # 📂 Enregistrement des résultats
    save_csv(stats_df, "statistiques_corpus.csv", output_dir)
    save_csv(pd.DataFrame(word_freq.most_common(), columns=["Mot", "Fréquence"]), "word_frequencies.csv", output_dir)
    save_csv(pd.DataFrame(bigram_freq.most_common(), columns=["Bigramme", "Fréquence"]), "bigram_frequencies.csv",
             output_dir)
    save_csv(pd.DataFrame(trigram_freq.most_common(), columns=["Trigramme", "Fréquence"]), "trigram_frequencies.csv",
             output_dir)

    # 🌥️ Génération du nuage de mots
    generate_wordcloud(word_freq, output_dir)