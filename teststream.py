# Importaciones necesarias
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import base64
import os

import cv2

print("Detectando cámaras disponibles...")
for index in range(5):  # Probar hasta 5 dispositivos
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Cámara detectada en el índice {index}")
        cap.release()
    else:
        print(f"No se detectó cámara en el índice {index}")



# Enumerar dispositivos de cámara conectados
def list_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(f"Camera {index}")
        cap.release()
        index += 1
    return cameras

# Obtener lista de cámaras disponibles
available_cameras = list_cameras()

# Mostrar un selector en la barra lateral para elegir la cámara
camera_index = st.sidebar.selectbox(
    "p",
    range(len(available_cameras)),
    format_func=lambda x: available_cameras[x]
)




def get_predictions(frame):
    _, encoded_image = cv2.imencode('.jpg', frame)
    url = "http://0.0.0.0:8080/predict"
    data = {"file":encoded_image.tobytes()}
    response = requests.post(url=url,data=data)
    response.raise_for_status()
    predictions = response.json()

    # Ejemplo de respuesta simulada

    # predictions = """
    # {
    #   "waste_categories": [
    #     5
    #   ],
    #   "confidence_score": [
    #     0.425735205411911
    #   ],
    #   "bounding_boxes": [
    #     [
    #       256.298095703125,
    #       351.49322509765625,
    #       320.6392822265625,
    #       435.74066162109375
    #     ]
    #   ]
    # }
    # """

    # # Convertir la respuesta simulada a un objeto Python (opcional)
    # predictions = json.loads(predictions)
    return predictions  # Devuelve como objeto Python (dict)







# Leer la imagen y codificarla en base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Configurar la imagen de fondo
set_bg('trashformer.jpg')

# Leer y codificar la imagen
with open("trashformer.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Definir estilos para el fondo, los títulos, y el GIF
background_style = f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap'); /* Importar Montserrat */

    /* Fondo de la aplicación */
    .stApp {{
        background: url(data:image/png;base64,{encoded_string}) no-repeat center center;
        background-size: cover;
        background-attachment: scroll; /* El fondo se desliza */
    }}

    /* Configuración del título a la derecha */
    h1 {{
        font-size: 4em; /* Tamaño de la fuente */
        color: orange; /* Color de la letra */
        text-shadow: -4px -4px 0 blue, 4px -4px 0 yellow, -4px 4px 0 blue, 4px 3px 0 yellow; /* Grosor del borde */
        font-family: 'Montserrat', sans-serif;
        text-align: right; /* Alinea el texto a la derecha */
        position: fixed; /* Fijar posición */
        top: 20px; /* Espaciado desde la parte superior */
        right: 20px; /* Espaciado desde la derecha */
        z-index: 1000; /* Asegurar que el título esté sobre otros elementos */
        background: rgba(0, 0, 0, 0); /* Fondo con opacidad para resaltar */
        padding: 60px; /* Espaciado interno */
    }}

    /* Fondo del sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.7); /* Fondo negro con opacidad */
        color: yellow; /* Color del texto */
        font-family: Arial, sans-serif;
    }}

    .gif-container {{
    display: flex;
    justify-content: flex-end; /* Alinea a la derecha */
    margin-top: 20px; /* Ajusta la posición vertical del GIF */
}}
    </style>
"""

# Contenido principal de la página
st.markdown(background_style, unsafe_allow_html=True)

# Título fijo a la derecha
st.title("TRASH-FORMERS")

# GIF
gif_path = "single-use-plastic-reusable-bag.gif"  # Cambia por la ruta de tu GIF
with open(gif_path, "rb") as gif_file:
    gif_base64 = base64.b64encode(gif_file.read()).decode()

gif_html = f"""
<div class="gif-container">
    <img src="data:image/gif;base64,{gif_base64}" alt="Trashformers GIF"
    style="width:160px; height:auto; border: 5px solid blue;"/>
</div>
"""

st.markdown(gif_html, unsafe_allow_html=True)




# url de API
# RECUPERAR LA RESPUESTA
# SI LA RESPUESTA ES OK LA DEVOLVEMOS






def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), results_pred=True):
    """ Affiche une seule de ces bounding boxes sur une image
        Affiche le label de l'objet détecté
    """
    image = image.copy()
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)

    if results_pred:
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    else:
        p1, p2 = (int(box[0]),int(box[1])), (int(box[0])+int(box[2]),int(box[1])+int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
      tf = max(lw - 1, 1)  # font thickness
      w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
      outside = p1[1] - h >= 3
      p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
      image = cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
      image = cv2.putText(image,
                  label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                  0,
                  lw / 3,
                  txt_color,
                  thickness=tf,
                  lineType=cv2.LINE_AA)
      return image

def plot_bboxes(image, results, labels=[], colors=[], score=True, conf=None,results_pred=True):
    """
        Fonction qui plot toutes les bboxes sur une même image
        Elle s'adapte aussi si on utilise la fonction sur notre propre machine ou sur google colab
    """
    if results_pred==True:
      boxes = results[0]
    else:
      # A modifier
      boxes = results

    image = image.copy()

  #Define COCO Labels
    if labels == []:
      labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
    #Define colors
    if colors == []:
      #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
      colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]

    #plot each boxes
    for box in boxes:
      #add score in label if score=True
      if score :
        label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
      else :
        label = labels[int(box[-1])]
      #filter every box under conf threshold if conf threshold setted
      if conf :
        if box[-2] > conf:
          color = colors[int(box[-1])]
          image = box_label(image, box, label, color)
      else:
        color = colors[int(box[-1])]
        image = box_label(image, box, label, color, results_pred=results_pred)

    # Different cas si jamais on est sur google Colab ou pas pour l'entrainement
    # try:
    #   import google.colab
    #   IN_COLAB = True
    # except:
    #   IN_COLAB = False

    # if IN_COLAB:
    #   cv2_imshow(image) #if used in Colab
    # else :
    #   plt.imshow(image) #if used in Python
    plt.imshow(image)

    cv2.imwrite('icti.jpg',image)
    return image

def draw_boxes(frame, predictions):
    """
    Annotates the frame with bounding boxes and prediction information using box_label and plot_bboxes.

    Parameters:
    - frame: La imagen original en formato numpy array.
    - predictions: JSON string con 'bounding_boxes', 'waste_categories' y 'confidence_score'.

    Returns:
    - Frame anotado con bounding boxes.
    """
    import json

    # Parsear las predicciones del JSON
    prediction_data = predictions
    bounding_boxes = prediction_data["bounding_boxes"]
    categories = prediction_data["waste_categories"]
    confidence_scores = prediction_data["confidence_score"]

    # Crear datos estructurados para las funciones de anotación
    results = []
    for i in range(len(bounding_boxes)):
        box = bounding_boxes[i]
        label = f"Catégorie {categories[i]} ({confidence_scores[i]:.2f})"

        results.append([*box, confidence_scores[i], categories[i]])

    # Pasar los datos a la función plot_bboxes
    annotated_frame = plot_bboxes(frame, [results], score=True, results_pred=True)

    print(predictions)


    return annotated_frame






import streamlit as st
import cv2



# Bouton pour activer/désactiver la caméra
# Utilisation d'une case à cocher pour démarrer ou arrêter la caméra
run = st.sidebar.checkbox("Turn Web Cam On", value=False, key="webcam_toggle")


st.markdown(
    """
    <style>
    /* Ajustar la altura y el espaciado del contenido de la barra lateral */
    section[data-testid="stSidebar"] {
        padding: 10px; /* Espaciado interno de la barra lateral */
        height: 100vh; /* Altura completa de la barra lateral */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Distribuir espacio entre los elementos */
    }

    .sidebar-title {
        margin-bottom: 1    0px; /* Espaciado debajo del título */
    }

    .icon-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end; /* Alinear al borde inferior */
        align-items: center;
        height: 180px; /* Ajustar la altura de cada contenedor */
        margin-bottom: 0px; /* Espacio entre filas */
    }

    .icon-container img {
        margin-bottom: 5px; /* Espacio entre imagen y texto */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

            # Simuler les prédictions
            predictions = {
                "bounding_boxes": [[50, 50, 150, 150], [200, 100, 300, 250]],
                "waste_categories": [1, 2],
                "confidence_score": [0.85, 0.75],
            }

            # Annoter les frames
            annoted_frame = frame  # Si draw_boxes est défini, l'utiliser ici

            # Afficher l'image dans Streamlit
            video_feed.image(annoted_frame, channels="RGB")

            # Vérifier dynamiquement si la case à cocher est toujours activée
            run = st.session_state.webcam_toggle

        # Libérer la caméra lorsque le flux est arrêté
        cap.release()
else:
    # Message d'information si la webcam n'est pas activée
    st.info("Give me some trash, baby")

import streamlit as st
import base64
import os
import streamlit as st
import os
import base64

# Inicializar los contadores en el estado de la sesión
if "counters" not in st.session_state:
    st.session_state["counters"] = {
        "PLASTIC": 0,
        "GLASS": 0,
        "CARDBOARD": 0,
        "METAL": 0,
        "PAPER": 0,
        "ORGANIC": 0,
    }

# Asociar las categorías a íconos personalizados
icon_urls = {
    "PLASTIC": "https://img.freepik.com/premium-vector/plastic-recycling-icon-waste-sorting-trash-container-isolated-white-background_176411-3501.jpg?w=740",
    "GLASS": "https://c8.alamy.com/comp/2HN461D/garbage-bin-with-glass-waste-recycling-garbage-vector-illustration-2HN461D.jpg",
    "CARDBOARD": "https://c8.alamy.com/comp/2WNNPGD/fulled-opened-lid-container-for-storing-recycling-and-sorting-used-household-paper-waste-blue-transportable-trash-bin-for-scrap-paper-books-and-car-2WNNPGD.jpg",
    "METAL": "https://c8.alamy.com/comp/2HNAYHA/garbage-bin-with-metal-waste-recycling-garbage-vector-illustration-2HNAYHA.jpg",
    "PAPER": "https://c8.alamy.com/comp/2JAMW7K/paper-recycling-trash-can-blue-container-for-waste-sorting-2JAMW7K.jpg",
    "ORGANIC": "https://img.freepik.com/premium-vector/organic-waste-garbage-container-compost-recycling-bin_176411-5963.jpg",
}

# Estilos CSS para la barra lateral y el GIF
st.markdown(
    """
    <style>
    .icon-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        align-items: center;
        height: 170px;
        margin-bottom: 10px;
    }
    .icon-container img {
        width: 70px;
        height: auto;
        margin-bottom: 0px;
        border: 5px solid yellow;
    }
    .sidebar-gif {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 0px;
        padding: 0px;
    }
    .sidebar-gif img {
        width: 150px;
        height: auto;
        border: 3px solid yellow;
        border-radius: 5px;
    }
    .sidebar-gif p {
        text-align: center;
        font-size: 30px;
        color: yellow;
        margin-bottom: 0px; /* Espacio entre el texto y el GIF */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colocar los íconos en la barra lateral en filas
with st.sidebar:
    rows = [[], []]  # Crear 2 filas vacías
    items = list(icon_urls.items())

    # Dividir los íconos en filas
    for i, (category, icon) in enumerate(items):
        if i < 3:
            rows[0].append((category, icon))  # Primera fila
        else:
            rows[1].append((category, icon))  # Segunda fila

    # Mostrar las filas en el sidebar
    for row in rows:
        cols = st.columns(len(row))  # Crear columnas dinámicamente según el número de imágenes
        for col, (category, icon) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="icon-container">
                        <p><strong>{category}: {st.session_state['counters'][category]}</strong></p>
                        <img src="{icon}">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Ruta del GIF
    gif_path = "robot-attack-saturday-night-live.gif"  # Cambia por la ruta correcta del GIF

    # Verificar si el archivo existe
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as gif_file:
            gif_base64 = base64.b64encode(gif_file.read()).decode()

        # Mostrar el texto y el GIF al final del sidebar
        st.markdown(
            f"""
            <div class="sidebar-gif">
                <p>RECYCLE, YOU MUST</p> <!-- Texto sobre el GIF -->
                <img src="data:image/gif;base64,{gif_base64}" alt="Sidebar GIF">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("El archivo GIF no se encontró. Verifica la ruta.")
