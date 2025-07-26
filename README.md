# ✋ LenguajeCol — Reconocimiento de Lengua de Señas Colombiana 🇨🇴

Aplicación de visión artificial en Python que detecta letras del alfabeto colombiano y palabras clave mediante gestos de mano usando MediaPipe y OpenCV. Diseñado con una interfaz moderna e intuitiva.

## 🎯 Características

- Detección de letras con la mano usando cámara web
- Reconocimiento de palabras clave como "Gracias", "Hola", "Te quiero"
- Diseño moderno, minimalista, con animaciones suaves
- Todo en una sola ventana interactiva
- Preparado para ampliarse con más funciones como voz o exportación

## 📸 Interfaz
![Preview](screenshots/demo.png)


## 🎥 Demostración

[🎬 Ver demostración en video](

https://github.com/user-attachments/assets/99b2684d-c4c5-4834-8357-b1f8da988cfe
)

## 🧠 Reconocimiento de Letras

| Letra | Gesto de dedos arriba |
|-------|------------------------|
| A     | 0 0 0 0 0              |
| B     | 0 1 1 1 1              |
| C     | Forma semicircular     |
| ...   | ...                    |

## 🗣️ Palabras especiales

- `[1, 1, 0, 0, 1]` → Hola
- `[0, 1, 0, 1, 0]` → Gracias
- `[1, 0, 1, 0, 1]` → Te quiero

✍️ Autores
Proyecto desarrollado por estudiantes del SENA:

👨‍💻 Octavio Gutiérrez – @tav1o__
👨‍💻 Jolian [Apellido] – (Puedes agregar su usuario de GitHub si lo desea
