import cv2

# Backends posibles
backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_DSHOW, cv2.CAP_V4L2]

for backend in backends:
    print(f"Probando backend {backend}")
    for index in range(5):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"Cámara detectada en el índice {index} con backend {backend}")
            cap.release()
        else:
            print(f"No se detectó cámara en el índice {index} con backend {backend}")
