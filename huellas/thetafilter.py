# Importar los paquetes necesarios
import cv2
import numpy as np


# Definición de la clase
class thetaFilter:

    def __init__(self, Image_gr):  ## Se define el constructor de la clase
        self.image = Image_gr  ## Se define el self de la imagen donde se recib en blanco y negro
        self.theta = 0  ## Se inicializa theta
        self.delta_theta = 0  ## Se inicializa delta theta

    def set_theta(self, theta, delta_theta):  ## Se define el metodo para definir el angulo y el delta del angulo
        self.theta = theta
        self.delta_theta = delta_theta

    def filtering(self):  ## Se define el metodo para el filtrado de la imagne por angulo
        image_gray_fft = np.fft.fft2(self.image)  ## se define la fft de la señal en grises
        image_gray = self.image
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # pre-computations
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])  ## Se define el numero de columnas y filas
        enum_rows = np.linspace(0, num_rows - 1, num_rows)               ## Se generan los vectores y el meshgrid para la matriz
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns

        # band pass filter mask
        band_pass_mask_1 = np.zeros_like(image_gray)  ## Se generan matrices de ceros
        band_pass_mask_2 = np.zeros_like(image_gray)
        band_pass_mask_3 = np.zeros_like(image_gray)
        band_pass_mask_4 = np.zeros_like(image_gray)

        thetamayor = self.theta + self.delta_theta  ## Se definen los rangos de theta segun lo que ingreso
        thetamenor = self.theta - self.delta_theta

        arg1 = col_iter - half_size
        arg2 = row_iter - half_size

        idx_low = (180 / np.pi) * (np.arctan2(arg2, arg1 * -1)) < thetamayor - 90   ## Se configuran los rangos para la mascara de acuerdo con las posiciones
        idx_high = (180 / np.pi) * (np.arctan2(arg2, arg1 * -1)) > thetamenor - 90

        idx_low_2 = (180 / np.pi) * (np.arctan2(arg2, arg1 * 1)) > -(thetamayor - 90) ## Se genera loa segundos valores para la mascara como espejo de la anterior
        idx_high_2 = (180 / np.pi) * (np.arctan2(arg2, arg1 * 1)) < -(thetamenor - 90)

        band_pass_mask_1[idx_low]    = 1   ## se pone los valores de las mascaras en 1
        band_pass_mask_2[idx_high]   = 1

        band_pass_mask_3[idx_low_2]  = 1
        band_pass_mask_4[idx_high_2] = 1

        mask = cv2.bitwise_and(band_pass_mask_1, band_pass_mask_2)  ## Se crea la primera mascara de acuerdo de a los primeros rangod
        mask2 = cv2.bitwise_and(band_pass_mask_3, band_pass_mask_4) ## Se crea la primera mascara de acuerdo de a los segundos rangod

        mask_total = cv2.bitwise_or(mask, mask2)       ## Se hace un or entre las mascaras para generar la mascara total

        fft_filtered = image_gray_fft_shift * mask_total  ## Se genera la imagen filtrada en frecuencia
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))  ## Se hace la fft inversa
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        cv2.imshow("Imagen Filtrada con angulo de {}".format(self.theta), image_filtered)  ## Se imprime en pantalla la imagen filtrada
        cv2.imshow("Imagen en grises", image_gray)             ## Se imprime la imagen en grises
        cv2.imshow("Mascara con angulo de {}".format(self.theta), 255 * mask_total)  ## Se muestra la mascara
        cv2.waitKey(0)

        return image_filtered  ## Se retorna la imagen filtrada
