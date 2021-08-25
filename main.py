import cv2
import os
import sys
from thetafilter import thetaFilter as tf


def banco_filtros(Imagen):
    image_gray = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)

    filtrado_1 = tf(image_gray)
    filtrado_1.set_theta(0, 5)
    imagen_1 = filtrado_1.filtering()

    filtrado_2 = tf(image_gray)
    filtrado_2.set_theta(45, 5)
    imagen_2 = filtrado_2.filtering()

    filtrado_3 = tf(image_gray)
    filtrado_3.set_theta(90,5)
    imagen_3 = filtrado_3.filtering()

    filtrado_4 = tf(image_gray)
    filtrado_4.set_theta(135, 5)
    imagen_4 = filtrado_4.filtering()

    new_imagen = (imagen_1 + imagen_2 + imagen_3 + imagen_4) / 4
    cv2.imshow("Imagen Promediada", new_imagen)
    cv2.waitKey(0)


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    banco_filtros(image)
