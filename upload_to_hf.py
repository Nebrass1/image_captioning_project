import os
from huggingface_hub import HfApi

HF_USERNAME = "Nebrass1" 

# Définition de l'ID du dépôt
REPO_ID = f"{HF_USERNAME}/image_captioning_resnet_lstm" 

# Initialisation de l'API
api = HfApi()

# Tente de créer le dépôt (ne le crée que s'il n'existe pas)
print(f"Tentative de création/vérification du dépôt: {REPO_ID}")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model")
    print(f"Dépôt {REPO_ID} créé avec succès.")
except Exception as e:
    if "You already have a repo named" in str(e):
         print(f"Dépôt {REPO_ID} existe déjà. Continuer l'upload.")
    else:
        print(f"Erreur lors de la création du dépôt: {e}")

# --- Liste des Fichiers à Uploader ---

files_to_upload = [
    # Les poids de votre Décodeur
    ("./caption_decoder_final.weights.h5", "caption_decoder_final.weights.h5", "Ajout des poids du décodeur (Entraînement 100%)"),
    # Votre Tokenizer
    ("./tokenizer.pkl", "tokenizer.pkl", "Ajout du tokenizer (vocabulaire non filtré)"),
    # L'architecture du modèle pour référence
    ("./model.py", "model.py", "Ajout de l'architecture Keras (ResNet+LSTM)")
]

# --- Boucle d'Upload ---
for local_path, path_in_repo, commit_msg in files_to_upload:
    if os.path.exists(local_path):
        print(f"Uploading {local_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            commit_message=commit_msg
        )
        print(f"✅ {path_in_repo} uploadé.")
    else:
        print(f"⚠️ Fichier manquant: {local_path}. Saut de l'upload.")

print("\n--- Uploads terminés sur Hugging Face ! ---")