from ultralytics import YOLO
import cv2
import time
# Cargar el modelo YOLO
model = YOLO('Yolo-weights/yolov8n.pt')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
Ptime = 0
accepted_classes = ['person', 'car', 'truck', 'bus', 'motorbike', 'bicycle']
while True:
    success, img = cap.read()
    if not success:
        print("Error al leer desde la cámara.")
        break

    # Realizar inferencia
    results = model.predict(source=img, save=False, show=False, conf=0.5)
    print(results)
    # Dibujar las detecciones en la imagen
    for result in results[0].boxes:
        box = result.xyxy[0]  # Coordenadas de la caja [x1, y1, x2, y2]
        conf = result.conf[0]  # Confianza de la detección
        cls = result.cls[0]  # Clase detectada
        label = f'{model.names[int(cls)]} {conf:.2f}'
        # Si la clase detectada está en la lista de clases aceptadas
        
        # Dibujar la caja
        x1, y1, x2, y2 = map(int, box)  # Convertir coordenadas a enteros
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Caja azul
        # Dibujar la etiqueta
        cv2.putText(img, label, (max(0, x1), max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Show fps
    Ctime = time.time()
    fps = 1 / (Ctime - Ptime)
    Ptime = Ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Mostrar la imagen con las detecciones
    cv2.imshow("Image", img)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
