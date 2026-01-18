import cv2
import numpy as np

# Cargar imagen en color
NOMBRE_IMAGEN = "people.jpg"
imagen_color = cv2.imread(NOMBRE_IMAGEN)


if imagen_color is None:
    print("Erro: não foi possível carregar a imagem. Verifique o nome e a pasta.")
    raise SystemExit

#  Mostrar información de la imagen
altura, largura, canais = imagen_color.shape
tipo_dados = imagen_color.dtype
print(f"Dimensões da imagem: {largura} x {altura}")
print(f"Número de canais: {canais}")
print(f"Tipo de dados: {tipo_dados}")

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

#Aplicar Canny para detección de bordes

imagen_bordes = cv2.Canny(imagen_gris, 100, 200)

# Detectar contornos externos y dibujarlos

contornos, jerarquia = cv2.findContours(imagen_bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagen_contornos = imagen_color.copy()

cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
print(f"Número de contornos encontrados: {len(contornos)}")

if len(contornos) > 0:
    
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)

    cv2.drawContours(imagen_contornos, [contorno_mayor], -1, (0, 0, 255), 3)
    cv2.rectangle(imagen_contornos, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(imagen_contornos, "Maior contorno", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
else:
    print("Aviso: nenhum contorno encontrado.")

# Mostrar imágenes procesadas en ventanas
cv2.imshow("Imagem Original", imagen_color)
cv2.imshow("Escala de Cinza", imagen_gris)
cv2.imshow("Deteccao de Bordas (Canny)", imagen_bordes)
cv2.imshow("Contornos Detectados", imagen_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detección de rostros con Haar Cascade
# Cargar clasificador frontal por defecto incluido en OpenCV
clasificador_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detectar rostros sobre la imagen en gris
rostros = clasificador_rostro.detectMultiScale(
    imagen_gris,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Rostos detectados: {len(rostros)}")

# Dibujar rectángulos de rostros
imagen_rostros = imagen_color.copy()
for (x, y, w, h) in rostros:
    cv2.rectangle(imagen_rostros, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Mostrar detección de rostros
cv2.imshow("Rostos Detectados", imagen_rostros)
cv2.waitKey(0)
cv2.destroyAllWindows()
