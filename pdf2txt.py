# ==========
# To Do:
#   - Git le projet
#   - Ajout dans le log des types d'erreurs si possible
#   - Structuration du projet et modules
# ==========

import os
import pdfplumber
from tqdm import tqdm
from datetime import datetime

# Variables des paths
pdf_root_folder = "data/input/"
output_folder = "data/output"
log_file = "data/extraction_log.txt"

# Création du dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Ajout d'un en-tête horodaté dans le log (sans l'écraser)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(log_file, "a", encoding="utf-8") as log:
    log.write(f"\n=== Début de l'extraction ({timestamp}) ===\n")

# Recherche de tous les fichiers PDF dans `data/input/` et ses sous-dossiers
pdf_files = []
for root, _, files in os.walk(pdf_root_folder):
    for filename in files:
        # Utilisation de l'extension .pdf
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(root, filename))

# Vérification s'il y a des fichiers à traiter
if not pdf_files:
    message = "❌ Aucun fichier PDF trouvé dans data/input/. Vérifiez vos dossiers.\n"
    print(message)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(message)
    exit(1)
else:
    print(f"🔍 {len(pdf_files)} fichiers PDF trouvés, début de l'extraction...")
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"🔍 {len(pdf_files)} fichiers PDF trouvés, début de l'extraction...\n")

# Variables de comptage pour le log
success_count = 0
error_count = 0

# Boucle d'extraction de l'OCR avec barre de progression (package tqdm) et log
for pdf_path in tqdm(pdf_files, desc="Extraction des PDF", unit="pdf"):
    try:
        # Génération du chemin de sortie
        relative_path = os.path.relpath(pdf_path, pdf_root_folder)
        text_output_path = os.path.join(output_folder, relative_path).replace(".pdf", ".txt")

        # Création des sous-dossiers de sortie si nécessaire
        os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

        # Extraction du texte avec pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text() is not None])

        # Écriture du texte dans un fichier .txt
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text)

        success_msg = f"✅ {pdf_path} → {text_output_path}\n"
        print(success_msg.strip())
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(success_msg)

        success_count += 1

    except Exception as e:
        error_msg = f"❌ Erreur lors de l'extraction de {pdf_path} : {e}\n"
        print(error_msg.strip())
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(error_msg)

        error_count += 1

# Fin du script avec résumé
final_msg = f"\n🎉 Extraction terminée ! Textes enregistrés dans {output_folder}\n"
final_msg += f"📄 {success_count} fichiers extraits avec succès.\n"
final_msg += f"⚠️ {error_count} erreurs rencontrées.\n"

print(final_msg.strip())
with open(log_file, "a", encoding="utf-8") as log:
    log.write(final_msg)