import cv2  ## Se llaman las librerias necesarias
import os
import sys
from thetafilter import thetaFilter as tf


def banco_filtros(Imagen):  ## Se crea una funcion para el banco de filtros pedido, se recibe como parametro una imagen
    image_gray = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)   ## A la imagen de entrada se le aplica conversiona grises

    filtrado_1 = tf(image_gray)  ## Se define la clase y se le ingresa la imagen
    filtrado_1.set_theta(0, 5)   ## Se usa el metodo set_theta para ingresar el angulo y el delta
    imagen_1 = filtrado_1.filtering()  ## Se usa el metodo filtering con los parametros

    filtrado_2 = tf(image_gray) ## Se define la clase y se le ingresa la imagen
    filtrado_2.set_theta(45, 5) ## Se usa el metodo set_theta para ingresar el angulo y el delta
    imagen_2 = filtrado_2.filtering()  ## Se usa el metodo filtering con los parametros

    filtrado_3 = tf(image_gray) ## Se define la clase y se le ingresa la imagen
    filtrado_3.set_theta(90, 5) ## Se usa el metodo set_theta para ingresar el angulo y el delta
    imagen_3 = filtrado_3.filtering()  ## Se usa el metodo filtering con los parametros

    filtrado_4 = tf(image_gray)  ## Se define la clase y se le ingresa la imagen
    filtrado_4.set_theta(135, 5) ## Se usa el metodo set_theta para ingresar el angulo y el delta
    imagen_4 = filtrado_4.filtering()  ## Se usa el metodo filtering con los parametros

    new_imagen = (imagen_1 + imagen_2 + imagen_3 + imagen_4) / 4  ## Se define la nueva imagen
    cv2.imshow("Imagen Promediada", new_imagen)  ## Se muestra la nueva imagen
    cv2.waitKey(0)


if __name__ == '__main__':   ## Se define el main
    path = sys.argv[1]  ## Se define el path de la imagen
    image_name = sys.argv[2]  ## Se define el nombre de la imagen
    path_file = os.path.join(path, image_name) ## Se define la ruta d ela imagen
    image = cv2.imread(path_file)  ## Se lee la imagen
    banco_filtros(image)   ## Se ingresa la imagen a la funcion de bancos de filtros
