from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout, Add

EMBEDDING_DIM = 256 # Taille de l'embedding pour l'image et le mot

def create_image_encoder(input_shape=(224, 224, 3)):
    """Crée le modèle Codeur (CNN) en utilisant ResNet50."""
    # 1. Charger ResNet50 sans la couche finale (classification)
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          pooling='avg', # Utilise l'Average Pooling pour obtenir un seul vecteur
                          input_shape=input_shape)
    
    # 2. Geler les poids du modèle (Transfer Learning)
    for layer in base_model.layers:
        layer.trainable = False
        
    # 3. Couche Dense pour obtenir l'embedding d'image de taille EMBEDDING_DIM
    image_features = base_model.output
    encoder_output = Dense(EMBEDDING_DIM, activation='relu')(image_features) 
    
    return Model(inputs=base_model.input, outputs=encoder_output)


def create_caption_decoder(vocab_size, max_length):
    """Crée le modèle Décodeur (LSTM) pour générer la description."""
    
    # Entrée 1: L'embedding d'image (sortie du Codeur)
    input_image = Input(shape=(EMBEDDING_DIM,))
    image_features = Dropout(0.5)(input_image)
    image_features = Dense(EMBEDDING_DIM, activation='relu')(image_features)

    # Entrée 2: La séquence de mots (le mot actuel pour prédire le suivant)
    input_sequence = Input(shape=(max_length,))
    
    # Couche d'Embedding des mots
    word_embeddings = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(input_sequence)
    word_embeddings = Dropout(0.5)(word_embeddings)
    
    # Couche LSTM pour traiter la séquence
    lstm_output = LSTM(EMBEDDING_DIM)(word_embeddings)

    # Fusion des caractéristiques de l'image et du LSTM
    # L'image est ajoutée à la sortie du LSTM
    decoder_input = Add()([image_features, lstm_output])
    decoder_input = Dense(EMBEDDING_DIM, activation='relu')(decoder_input)

    # Sortie: prédiction du mot suivant (classification sur tout le vocabulaire)
    outputs = Dense(vocab_size, activation='softmax')(decoder_input)
    
    # Modèle complet
    # Il a besoin à la fois de l'embedding d'image et de la séquence de mots.
    model = Model(inputs=[input_image, input_sequence], outputs=outputs)
    
    return model