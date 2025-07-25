# LenguajeCol - Señalador IA con diseño moderno, voz, animaciones y ayuda

import cv2
import numpy as np
import mediapipe as mp

import time

# Inicializar MediaPipe y motor de voz
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Colores y fuente
COLOR_FONDO = (240, 230, 210)
COLOR_CAJA = (0, 180, 255)
COLOR_TEXTO = (20, 20, 20)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Funciones de lógica
def detectar_palabra_especial(dedos_arriba):
    if dedos_arriba == [1, 1, 0, 0, 1]: return "Hola"
    elif dedos_arriba == [0, 1, 0, 1, 0]: return "Gracias"
    elif dedos_arriba == [1, 0, 1, 0, 1]: return "Te quiero"
    return ""

def detectar_letra(mano):
    dedos_arriba = []
    puntos = mano.landmark

    if puntos[4].x < puntos[3].x: dedos_arriba.append(1)
    else: dedos_arriba.append(0)

    for id in [8, 12, 16, 20]:
        if puntos[id].y < puntos[id - 2].y:
            dedos_arriba.append(1)
        else:
            dedos_arriba.append(0)

    palabra = detectar_palabra_especial(dedos_arriba)
    if palabra: return palabra

    letras = {
        (0, 0, 0, 0, 0): "A",
        (0, 1, 1, 0, 0): "U",
        (0, 1, 0, 0, 0): "D",
        (0, 1, 1, 1, 1): "B",
        (1, 1, 0, 0, 0): "L",
        (0, 1, 1, 0, 1): "W",
        (0, 1, 0, 1, 1): "Y",
        (1, 0, 0, 0, 0): "E",
        (1, 1, 1, 1, 1): "F",
        (1, 0, 0, 0, 1): "C",
        (1, 0, 1, 1, 0): "M",
        (1, 0, 1, 0, 0): "N",
        (1, 1, 0, 1, 0): "K",
        (0, 0, 1, 1, 1): "I",
        (0, 0, 0, 1, 1): "J"
    }

    return letras.get(tuple(dedos_arriba), "")

# Configurar cámara y ventana
cv2.namedWindow("LenguajeCol", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LenguajeCol", 1280, 720)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ultima_palabra = ""
tiempo_ultima_palabra = time.time()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
) as manos:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fondo = np.full((720, 1280, 3), COLOR_FONDO, dtype=np.uint8)

        # Área de detección de mano
        x1, y1, x2, y2 = 340, 100, 940, 600
        roi_mano = frame[y1:y2, x1:x2]
        frame_rgb = cv2.cvtColor(roi_mano, cv2.COLOR_BGR2RGB)
        resultado = manos.process(frame_rgb)

        letra_detectada = ""

        if resultado.multi_hand_landmarks:
            for mano in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    roi_mano, mano, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_CAJA, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=COLOR_TEXTO, thickness=2)
                )
                letra = detectar_letra(mano)
                if letra:
                    letra_detectada = letra

        # Mostrar la mano detectada en pantalla principal
        fondo[60:660, 140:1140] = cv2.resize(roi_mano, (1000, 600))
        cv2.rectangle(fondo, (140, 60), (1140, 660), COLOR_CAJA, 4)

        # Mostrar letra detectada en el centro inferior
        if letra_detectada:
            text_size = cv2.getTextSize(letra_detectada, FUENTE, 2.0, 3)[0]
            text_x = int((1280 - text_size[0]) / 2)
            cv2.putText(fondo, letra_detectada, (text_x, 710), FUENTE, 2.0, COLOR_CAJA, 3, cv2.LINE_AA)

            if letra_detectada != ultima_palabra or time.time() - tiempo_ultima_palabra > 3:
                tiempo_ultima_palabra = time.time()

        # Mostrar pantalla
        cv2.imshow("LenguajeCol", fondo)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
