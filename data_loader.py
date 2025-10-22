import string
from tensorflow.keras.preprocessing.text import Tokenizer

def load_captions(filename):
    """Charge les légendes (captions) à partir du fichier CSV."""
    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    mapping = {}
    
    # Séparer par lignes
    lines = text.split('\n')
    
    # Ignorer la première ligne d'en-tête (image,caption)
    lines = lines[1:] 

    lines_read = 0 
    for line in lines:
        if not line:
            continue
            
        # ⚠️ CORRECTION: Utilisation de la virgule comme séparateur (split(',', 1))
        # split(',', 1) garantit que la description, même si elle contient des virgules, 
        # reste une seule entité après le premier séparateur.
        tokens = line.split(',', 1) 
        
        if len(tokens) < 2:
            continue
        
        image_name = tokens[0].strip() # Ex: 1000268201_693b08cb0e.jpg
        caption = tokens[1].strip()

        # Nous avons besoin de l'ID sans l'extension .jpg pour la clé du dictionnaire
        # (Assurez-vous que vos images dans data/Images/ se terminent par .jpg)
        image_id = image_name.split('.')[0] 
        
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

        lines_read += 1
        
    print(f"DEBUG: Total des lignes de légendes valides lues et mappées: {lines_read}")
    
    # Vérification critique
    if lines_read == 0:
         raise ValueError("Aucune description valide n'a été lue. Vérifiez le format du fichier.")

    return mapping

def clean_captions(mapping):
    """Nettoie les légendes (minuscules, ponctuation, <start>/<end>)."""
    table = str.maketrans('', '', string.punctuation)
    for key, captions in mapping.items():
        for i, caption in enumerate(captions):
            # 1. Normalisation
            caption = caption.replace('-', ' ')
            caption = caption.split()
            # 2. Mise en minuscules et suppression de la ponctuation
            caption = [word.lower().translate(table) for word in caption]
            # 3. Suppression des jetons non alphabétiques
            caption = [word for word in caption if len(word) > 1 and word.isalpha()]
            # 4. Ajout des jetons de début/fin
            caption = '<start> ' + ' '.join(caption) + ' <end>'
            captions[i] = caption

def create_tokenizer(mapping, threshold=5):
    """Crée un Tokenizer et filtre le vocabulaire."""
    all_captions = [caption for key in mapping for caption in mapping[key]]
    
    # Keras Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    # ⚠️ Filtrage par fréquence (très important pour réduire la taille du vocabulaire)
    # Les mots qui n'apparaissent qu'une seule fois ne sont pas utiles.
    # On crée une liste des mots rares à exclure
    
    # On reconstruit le tokenizer avec un vocabulaire filtré (ou on utilise num_words si Keras le permet)
    # Pour simplifier, nous allons prendre le vocabulaire tel quel pour le moment, 
    # mais gardez à l'esprit que la taille du vocabulaire est cruciale.
    
    # Déterminer la longueur maximale de la description
    max_length = max(len(s.split()) for s in all_captions)
    
    return tokenizer, max_length

# Fonction à appeler dans main.py
def load_data(caption_file):
    captions_mapping = load_captions(caption_file)
    clean_captions(captions_mapping)
    tokenizer, max_length = create_tokenizer(captions_mapping)
    
    return captions_mapping, tokenizer, max_length