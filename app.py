import streamlit as st
import cv2
from PIL import Image
import numpy as np

prediction_test = """
{
  "waste_categories": [
    5
  ],
  "confidence_score": [
    0.425735205411911
  ],
  "bounding_boxes": [
    [
      256.298095703125,
      351.49322509765625,
      320.6392822265625,
      435.74066162109375
    ]
  ]
}
"""





def get_predictions(frame):

    return prediction_test


def draw_boxes(frame, predictions):
    """
    Annotates the frame with bounding boxes and prediction information.

    Parameters:
    - frame: The original image (numpy array).
    - predictions: JSON string containing 'bounding_boxes', 'waste_categories', and 'confidence_score'.

    Returns:
    - Annotated frame with bounding boxes and labels.
    """
    import json

    # Parse the predictions JSON
    prediction_data = json.loads(predictions)

    # Extraer las coordenadas, categorías y confianza
    bounding_boxes = prediction_data["bounding_boxes"]
    waste_categories = prediction_data["waste_categories"]
    confidence_scores = prediction_data["confidence_score"]

    # Convertir el frame de RGB a BGR para OpenCV
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Dibujar las bounding boxes en la imagen
    for i, box in enumerate(bounding_boxes):
        # Coordenadas del bounding box
        x_min, y_min, x_max, y_max = map(int, box)

        # Categoría y confianza
        category = waste_categories[i]
        confidence = confidence_scores[i]

        # Dibujar el rectángulo
        cv2.rectangle(opencv_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Texto para la categoría y confianza
        label = f"{category}: {confidence:.2f}"
        cv2.putText(opencv_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convertir la imagen anotada de vuelta a RGB
    annotated_frame = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    return annotated_frame



# Titre de l'application
st.title("Trashformers - Webcam")

# Bouton pour activer/désactiver la caméra
# Utilisation d'une case à cocher pour démarrer ou arrêter la caméra
run = st.sidebar.checkbox("Activer Webcam", value=False, key="webcam_toggle")

# Conteneur pour afficher le flux vidéo
video_feed = st.empty()

# Démarrer la capture vidéo si la webcam est activée
if run:
    # Initialisation de la caméra
    cap = cv2.VideoCapture(0)

    # Vérifie si la caméra s'ouvre correctement
    if not cap.isOpened():
        st.error("Impossible d'accéder à la caméra.")
    else:
        # Boucle pour afficher les frames en temps réel
        while run:
            # Lire une image depuis la webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Impossible de lire le flux de la caméra.")
                break

            # Convertir l'image en RGB (OpenCV utilise BGR par défaut)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            predictions = get_predictions(frame)
            annoted_frame = draw_boxes(frame, predictions)

            # Afficher l'image dans Streamlit
            video_feed.image(frame, channels="RGB")

            # Vérifier dynamiquement si la case à cocher est toujours activée
            run = st.session_state.webcam_toggle

        # Libérer la caméra lorsque le flux est arrêté
        cap.release()
else:
    # Message d'information si la webcam n'est pas activée
    st.info("Activez la caméra depuis la barre latérale pour démarrer le flux.")


# Initialiser les compteurs dans l'état de la session
if "counters" not in st.session_state:
    st.session_state["counters"] = {
        "Plastiques recyclables": 0,
        "Plastiques souples": 0,
        "Plastiques difficiles à recycler": 0,
        "Papier et carton": 0,
        "Métaux": 0,
        "Verre": 0,
        "Non recyclable": 0,
        "Non identifié": 0,
    }

# Associer les catégories à des icônes personnalisées
icon_urls = {
    "Plastiques recyclables": "https://img.freepik.com/premium-vector/plastic-recycling-icon-waste-sorting-trash-container-isolated-white-background_176411-3501.jpg?w=740",
    "Plastiques souples": "https://cdn.vectorstock.com/i/1000x1000/70/89/rubbish-container-for-plastic-waste-icon-recycle-vector-15577089.webp",
    "Plastiques difficiles à recycler": "https://img.freepik.com/premium-psd/green-recycle-basket-icon-plastic-3d-isolated-transparent-background_220739-150766.jpg?w=1380",
    "Papier et carton": "https://c7.alamy.com/comp/2RRKB0P/garbage-sorting-set-yellow-bin-with-recycling-symbol-for-paper-waste-vector-illustration-for-zero-waste-environment-protection-concept-2RRKB0P.jpg",
    "Métaux": "https://img.freepik.com/premium-vector/red-can-with-sorted-metal-garbage-vector-icon-recycling-garbage-separation_850451-1407.jpg?w=1380",
    "Verre": "https://c7.alamy.com/comp/2RR5AH4/waste-management-glass-recycle-bins-icon-as-eps-10-file-2RR5AH4.jpg",
    "Non recyclable": "https://www.shutterstock.com/shutterstock/photos/1830410063/display_1500/stock-vector-do-not-recycle-only-trash-sign-or-non-recyclable-waste-symbol-1830410063.jpg",
    "Non identifié": "https://www.shutterstock.com/shutterstock/photos/289322993/display_1500/stock-photo-recycle-garbage-questions-and-reusable-waste-management-solutions-or-confusion-concept-as-old-paper-289322993.jpg",
}

# Disposition en grille pour les compteurs
columns = st.columns(4)  # Organiser en 4 colonnes pour une meilleure lisibilité
for i, (category, icon) in enumerate(icon_urls.items()):
    with columns[i % 4]:  # Répartir les catégories entre les colonnes
        # Afficher l'icône agrandie et le compteur
        st.image(icon, width=100)  # Taille ajustée à 100 pixels de large
        st.markdown(f"**{category}**")
        st.markdown(f"Compteur : {st.session_state['counters'][category]}")
