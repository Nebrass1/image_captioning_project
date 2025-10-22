import numpy as np
import pickle
import os
import heapq
from PIL import Image

# Keras/TF et dépendances nécessaires pour le modèle
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout, Add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer

import gradio as gr

# --- 1. FONCTIONS DE MODÈLE (Extrait de model.py) ---
EMBEDDING_DIM = 256 # Doit correspondre à model.py

def create_image_encoder(input_shape=(224, 224, 3)):
    # Similaire à votre code model.py, mais n'est utilisé que pour le chargement ici
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    image_features = base_model.output
    encoder_output = Dense(EMBEDDING_DIM, activation='relu')(image_features) 
    return Model(inputs=base_model.input, outputs=encoder_output)

def create_caption_decoder(vocab_size, max_length):
    # Similaire à votre code model.py
    input_image = Input(shape=(EMBEDDING_DIM,))
    image_features = Dropout(0.5)(input_image)
    image_features = Dense(EMBEDDING_DIM, activation='relu')(image_features)

    input_sequence = Input(shape=(max_length,))
    word_embeddings = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(input_sequence)
    word_embeddings = Dropout(0.5)(word_embeddings)
    lstm_output = LSTM(EMBEDDING_DIM)(word_embeddings)

    decoder_input = Add()([image_features, lstm_output])
    decoder_input = Dense(EMBEDDING_DIM, activation='relu')(decoder_input)

    outputs = Dense(vocab_size, activation='softmax')(decoder_input)
    
    return Model(inputs=[input_image, input_sequence], outputs=outputs)


# --- 2. FONCTIONS D'INFÉRENCE (Extrait de main.py) ---

TARGET_SIZE = (224, 224)

def preprocess_image(image):
    """Prétraite l'image PIL pour ResNet50."""
    img = image.resize(TARGET_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) 
    img = preprocess_input(img) 
    return img

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def length_normalized_score(log_prob, sequence_ids, alpha=0.9):
    length = len(sequence_ids) - 1 
    if length <= 0:
        return -float('inf')
    return log_prob / (length ** alpha)

def beam_search_caption(model, image_features, tokenizer, max_length, beam_size=3):
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

        if not all_candidates: break
            
        initial_sequence = heapq.nlargest(beam_size, all_candidates, key=lambda x: x[0])
        for log_prob, seq_ids in initial_sequence:
            if seq_ids[-1] == tokenizer.word_index.get('<end>'):
                 final_captions.append((log_prob, seq_ids))

    if not final_captions:
        if initial_sequence:
            best_log_prob, best_sequence_ids = max(initial_sequence, key=lambda x: length_normalized_score(x[0], x[1]))
        else:
            return "Génération échouée."
    else:
        best_log_prob, best_sequence_ids = max(final_captions, key=lambda x: length_normalized_score(x[0], x[1]))

    final_caption = ' '.join([word_for_id(i, tokenizer) for i in best_sequence_ids])
    final_caption = final_caption.replace('<start>', '').strip()
    final_caption = final_caption.replace(' end', ' ').strip() 
    return final_caption

# --- 3. CHARGEMENT GLOBAL DES MODÈLES ---

# Définition des chemins (Hugging Face les trouvera à la racine du Space)
WEIGHTS_PATH = "./caption_decoder_final.weights.h5"
TOKENIZER_PATH = "./tokenizer.pkl"

# Initialisation des variables globales
tokenizer = None
image_encoder = None
caption_decoder = None
MAX_LENGTH = 35 # Longueur max de séquence

try:
    # 1. Charger le Tokenizer
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    
    # 2. Créer l'Encoder (ResNet50)
    image_encoder = create_image_encoder()
    
    # 3. Créer le Décodeur (LSTM) et charger les poids
    caption_decoder = create_caption_decoder(VOCAB_SIZE, MAX_LENGTH)
    caption_decoder.load_weights(WEIGHTS_PATH)
    
    print("Modèles chargés et prêts pour l'inférence.")

except Exception as e:
    print(f"Erreur de chargement des modèles : {e}")
    # Si le chargement échoue, l'application ne fonctionnera pas

# --- 4. FONCTION PRINCIPALE D'INFÉRENCE POUR GRADIO ---

def generate_caption(image: Image.Image) -> str:
    """
    Fonction enveloppe pour Gradio.
    Prend une image PIL et retourne la description.
    """
    if image_encoder is None or caption_decoder is None:
        return "Erreur : Les modèles n'ont pas pu être chargés."
        
    # Prétraitement et encodage de l'image
    processed_image = preprocess_image(image)
    image_features = image_encoder.predict(processed_image, verbose=0)
    
    # Génération de la description avec Beam Search
    caption = beam_search_caption(
        caption_decoder, 
        image_features, 
        tokenizer, 
        MAX_LENGTH, 
        beam_size=3
    )
    return caption

# --- 5. INTERFACE GRADIO ---

if __name__ == '__main__':
    if image_encoder and caption_decoder:
        interface = gr.Interface(
            fn=generate_caption, 
            inputs=gr.Image(type="pil", label="Télécharger une Image"), 
            outputs=gr.Textbox(label="Description Générée (Beam Search K=3)"),
            title="Image Captioning (ResNet-LSTM) - Démo",
            description="Ce modèle utilise ResNet50 pour encoder l'image et un LSTM pour générer la description textuelle (Beam Search K=3). Il a été entraîné sur Flickr30k. Les légendes peuvent présenter un biais (Problème de l'entraînement non filtré)."
        )
        interface.launch()
    else:
        print("L'interface Gradio n'a pas été lancée en raison d'une erreur de chargement du modèle.")