import numpy as np
import os
import random
from tensorflow.keras.callbacks import ModelCheckpoint 
import pickle 
import heapq 

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input 

# Importer vos modules
from data_loader import load_data
from model import create_image_encoder, create_caption_decoder # Assurez-vous que EMBEDDING_DIM est défini dans model.py

# --- 1. Paramètres Globaux ---
CAPTION_FILE = './data/captions.txt'
IMAGE_DIR = './data/Images/' 
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
TEST_SPLIT = 0.1

# Configuration pour le test des poids entraînés sur 100% des images
SAMPLE_RATIO = 1.0 

# --- 2. Fonctions Utilitaires d'Inférence (Beam Search CORRIGÉE) ---

def word_for_id(integer, tokenizer):
    """Convertit un jeton entier en mot."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def preprocess_image(image_path):
    """Charge et prétraite l'image pour ResNet50 (utilise preprocess_input)."""
    img = load_img(image_path, target_size=TARGET_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) 
    img = preprocess_input(img) 
    return img
    
def length_normalized_score(log_prob, sequence_ids, alpha=0.9):
    """Calcule le score normalisé par la longueur pour la Beam Search. Alpha=0.9 pour favoriser les longues phrases."""
    length = len(sequence_ids) - 1 
    if length <= 0:
        return -float('inf')
    return log_prob / (length ** alpha)

def beam_search_caption(model, image_features, tokenizer, max_length, beam_size=3):
    """
    Génère une description en utilisant la Beam Search, avec correction log(0) et normalisation de longueur.
    """
    
    start_token = tokenizer.texts_to_sequences(['<start>'])[0][0]
    initial_sequence = [[0.0, [start_token]]] 
    
    final_captions = []

    for _ in range(max_length):
        all_candidates = []
        
        for log_prob, sequence_ids in initial_sequence:
            
            last_word_id = sequence_ids[-1]
            
            if last_word_id == tokenizer.word_index.get('<end>'):
                final_captions.append((log_prob, sequence_ids))
                continue
            
            current_sequence = pad_sequences([sequence_ids], maxlen=max_length, padding='post')[0]
            current_sequence = np.array([current_sequence])

            prediction = model.predict([image_features, current_sequence], verbose=0)[0]
            
            # CORRECTION: Éviter log(0) en remplaçant les probabilités nulles
            prediction[prediction < 1e-12] = 1e-12 
            log_probabilities = np.log(prediction)
            
            top_k_indices = np.argsort(log_probabilities)[-beam_size:]
            
            for next_word_id in top_k_indices:
                if next_word_id == 0:
                    continue
                    
                next_log_prob = log_probabilities[next_word_id]
                
                new_log_prob = log_prob + next_log_prob
                new_sequence_ids = sequence_ids + [next_word_id]
                
                all_candidates.append((new_log_prob, new_sequence_ids))

        if not all_candidates:
            break
            
        initial_sequence = heapq.nlargest(beam_size, all_candidates, key=lambda x: x[0])
        
        for log_prob, seq_ids in initial_sequence:
            if seq_ids[-1] == tokenizer.word_index.get('<end>'):
                 final_captions.append((log_prob, seq_ids))

    # S'assurer qu'au moins une séquence est présente
    if not final_captions:
        if initial_sequence:
            best_log_prob, best_sequence_ids = max(
                initial_sequence, 
                key=lambda x: length_normalized_score(x[0], x[1])
            )
        else:
            return "Génération échouée."
    else:
        best_log_prob, best_sequence_ids = max(
            final_captions, 
            key=lambda x: length_normalized_score(x[0], x[1])
        )


    # Conversion des IDs en mots
    final_caption = ' '.join([word_for_id(i, tokenizer) for i in best_sequence_ids])
    
    # CORRECTION: Nettoyage final pour éliminer les multiples jetons de fin
    final_caption = final_caption.replace('<start>', '').strip()
    final_caption = final_caption.replace(' end', ' ').strip() 
    
    return final_caption


# --- 3. Fonction Générateur de Données d'Entraînement (Inchangée) ---

def data_generator(data_keys, captions_mapping, tokenizer, image_features_map, max_length, vocab_size, batch_size):
    X_img, X_seq, Y_out = [], [], []

    while True:
        random.shuffle(data_keys) 
        
        for image_id in data_keys:
            if image_id in image_features_map:
                image_features = image_features_map[image_id]
                captions = captions_mapping[image_id]
                
                for caption in captions:
                    sequence = tokenizer.texts_to_sequences([caption])[0]
                    
                    for i in range(1, len(sequence)):
                        in_sequence = sequence[:i]
                        out_word = sequence[i]
                        
                        in_sequence = pad_sequences([in_sequence], maxlen=max_length, padding='post')[0]
                        out_word = to_categorical([out_word], num_classes=vocab_size)[0]
                        
                        X_img.append(image_features)
                        X_seq.append(in_sequence)
                        Y_out.append(out_word)

                        if len(X_img) >= batch_size:
                            yield (np.array(X_img), np.array(X_seq)), np.array(Y_out)
                            X_img, X_seq, Y_out = [], [], []


# --- 4. Fonction Principale ---

def main():
    print("--- 🧠 Démarrage du Projet Image Captioning ---")
    
    # 4.1. Chargement des Légendes et Tokenizer
    # NOTE: load_data UTILISE LA VERSION FILTRÉE (si vous l'avez mis à jour dans data_loader.py)
    captions_mapping, tokenizer, max_length = load_data(CAPTION_FILE)
    vocab_size = len(tokenizer.word_index) + 1 
    print(f"Vocabulaire total: {vocab_size}, Longueur max. de séquence: {max_length}")
    
    # 4.2. Séparation des Données (Train/Test)
    image_ids = list(captions_mapping.keys())
    random.shuffle(image_ids)
    
    sample_size = int(len(image_ids) * SAMPLE_RATIO)
    image_ids = image_ids[:sample_size]
    print(f"ATTENTION: Jeu de données: {SAMPLE_RATIO * 100:.0f}% ({len(image_ids)} images).")

    test_count = int(len(image_ids) * TEST_SPLIT)
    train_ids = image_ids[:-test_count]
    test_ids = image_ids[-test_count:]
    print(f"Images d'entraînement: {len(train_ids)}, Images de test: {len(test_ids)}")
    
    # 4.3. Pré-Encoder les Images (Codeur)
    image_encoder = create_image_encoder(input_shape=(*TARGET_SIZE, 3))
    image_features_map = {}
    
    print("\n--- 🖼️ Pré-encodage des images... ---")
    for i, image_id in enumerate(image_ids):
        # ... (Le code de pré-encodage est omis ici pour la concision, mais il doit être complet dans votre fichier)
        # Assurez-vous que le bloc de pré-encodage est bien présent et fonctionnel.
        try:
            filename = os.path.join(IMAGE_DIR, image_id + '.jpg') 
            
            if (i + 1) % 100 == 0:
               print(f"    -> Encodage en cours: Image {i+1}/{len(image_ids)}: {image_id}.jpg")
               
            if not os.path.exists(filename):
                continue 

            image = preprocess_image(filename)
            features = image_encoder.predict(image, verbose=0)
            image_features_map[image_id] = features[0]
            
        except Exception:
            continue
            
    print(f"Images encodées: {len(image_features_map)}")

    # 4.4. Préparation et Entraînement du Décodeur

    caption_decoder = create_caption_decoder(vocab_size, max_length)

    try:
        caption_decoder.load_weights('./caption_decoder_final.weights.h5')
        print("INFO: Poids finaux chargés. Entraînement ignoré.")
    except Exception:
        print("INFO: Poids non trouvés. Démarrage de l'apprentissage (ce qui prendra du temps).")


    caption_decoder.compile(loss='categorical_crossentropy', optimizer='adam')
    
    filepath = './weights-epoch-{epoch:02d}.h5'
    checkpoint = ModelCheckpoint(
        filepath, 
        monitor='loss',
        verbose=1,
        save_best_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]


    total_samples = 0
    for image_id in train_ids:
        if image_id in image_features_map:
            for caption in captions_mapping[image_id]:
                sequence = tokenizer.texts_to_sequences([caption])[0]
                total_samples += len(sequence) - 1
                
    steps_per_epoch = total_samples // BATCH_SIZE 
    
    print(f"\n--- 🚀 Début de l'entraînement du Décodeur ---")
    print(f"Total des échantillons: {total_samples}, Étapes par époque: {steps_per_epoch}")
    
    # 🚨 BLOC D'ENTRAÎNEMENT COMMENTÉ
    print("⚠️ Entraînement ignoré. Passage au test avec les poids chargés.")
    # caption_decoder.fit(
    #     data_generator(train_ids, captions_mapping, tokenizer, image_features_map, max_length, vocab_size, BATCH_SIZE),
    #     epochs=EPOCHS, 
    #     steps_per_epoch=steps_per_epoch,
    #     callbacks=callbacks_list, 
    #     verbose=1
    # )

    # Sauvegarde du Tokenizer
    print("--- 💾 Sauvegarde du Tokenizer ---")
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    print("Tokenizer sauvegardé dans tokenizer.pkl.")


    # 4.5. Inférence sur les Données de Test (avec Beam Search CORRIGÉE)

    print("\n--- ✅ Test et Inférence ---")
    
    inference_decoder = create_caption_decoder(vocab_size, max_length)
    
    try:
        inference_decoder.load_weights('./caption_decoder_final.weights.h5')
        print("INFO: Poids finaux chargés (caption_decoder_final.weights.h5).")
    except Exception:
         print("Avertissement: Échec du chargement des poids finaux. Le modèle pourrait être vierge.")
             
    test_samples = random.sample(test_ids, min(5, len(test_ids)))
    
    for image_id in test_samples:
        image_path = os.path.join(IMAGE_DIR, image_id + '.jpg')
        
        if image_id in image_features_map:
            image_features = np.expand_dims(image_features_map[image_id], axis=0)
            
            generated_caption = beam_search_caption(inference_decoder, image_features, tokenizer, max_length, beam_size=3)
            
            print(f"\nImage ID: {image_id}")
            print(f"Description Générée (Beam Search K=3): {generated_caption}")
            
            real_captions = [c.replace('<start> ', '').replace(' <end>', '').strip() 
                             for c in captions_mapping[image_id][:2]]
            print(f"Légendes Réelles (Exemple):")
            for cap in real_captions:
                 print(f" - {cap}")
        
    print("\n--- Projet terminé ---")


if __name__ == '__main__':
    main()