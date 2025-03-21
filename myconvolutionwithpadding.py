import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image_path, filtro):
    
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen")
        return None

    # Obtener dimensiones
    img_height, img_width = image.shape
    filtro_height, filtro_width = filtro.shape


    # Crear imagen con padding (borde relleno de ceros)
    padded_image = np.pad(image, pad_width=10, mode='constant', constant_values=0)

    # Inicializar imagen de salida
    output = np.zeros((img_height, img_width), dtype=np.float32)

    # Aplicar convolución
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + filtro_height, j:j + filtro_width]
            output[i, j] = np.sum(region * filtro)

    output = np.clip(output, 0, 255).astype(np.uint8)

    return image, padded_image, output  # Devuelve la imagen original y la filtrada

# Ejemplo de uso
if __name__ == "__main__":
    image_path = "Turquia.jpg"  # Cambia esto a la ruta correcta de tu imagen

    # Filtro Sobel en X (detección de bordes)
    filtro = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Aplicar convolución
    original, padded, resultado = convolution(image_path, filtro)

    if resultado is not None:
        # Mostrar las imágenes con Matplotlib
        plt.figure(figsize=(10, 5))

        # Imagen original
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Imagen Original")
        plt.axis("off")

        # Imagen con padding aplicado
        plt.subplot(1, 3, 2)
        plt.imshow(padded, cmap='gray')
        plt.title("Imagen con Padding")
        plt.axis("off")

        # Imagen con convolución y padding
        plt.subplot(1, 3, 3)
        plt.imshow(resultado, cmap='gray')
        plt.title("Imagen con Convolución (Con Padding)")
        plt.axis("off")

        # Mostrar la figura
        plt.show()