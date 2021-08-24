import cv2  ## Se importan las librerias necesarias
import os
import sys
from thetafilter import thetaFilter as tf ## Se importa la clase proveniente de otro archivo.py


def banco_filtros(Imagen):   ## Se crea la funcion de banco de filtros, se pide como parametro una imagen de entrada
    image_gray = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)  ## se transforma la imagen a grises

    filtrado_1 = tf(image_gray)  ## Se define un objeto con la clase thetha filter y se ingresa la imagen en grises
    filtrado_1.set_theta(0, 5)   ## Se definen los parametros theta y delta theta
    imagen_1 = filtrado_1.filtering()  ## Se usa el metodo filtering y se retorna la imagen filtrada

    filtrado_2 = tf(image_gray)  ## Se define un objeto con la clase thetha filter y se ingresa la imagen en grises
    filtrado_2.set_theta(45, 5) ## Se definen los parametros theta y delta theta
    imagen_2 = filtrado_2.filtering() ## Se usa el metodo filtering y se retorna la imagen filtrada

    filtrado_3 = tf(image_gray) ## Se define un objeto con la clase thetha filter y se ingresa la imagen en grises
    filtrado_3.set_theta(90, 5) ## Se definen los parametros theta y delta theta
    imagen_3 = filtrado_3.filtering()  ## Se usa el metodo filtering y se retorna la imagen filtrada

    filtrado_4 = tf(image_gray)  ## Se define un objeto con la clase thetha filter y se ingresa la imagen en grises
    filtrado_4.set_theta(135, 5) ## Se definen los parametros theta y delta theta
    imagen_4 = filtrado_4.filtering() ## Se usa el metodo filtering y se retorna la imagen filtrada

    new_imagen = (imagen_1 + imagen_2 + imagen_3 + imagen_4) / 4  ## Se crea la nueva imagen a partir de las imagenes filtradas
    cv2.imshow("Imagen Promediada", new_imagen)  ## Se muestra la iamgen promediada
    cv2.waitKey(0)


if __name__ == '__main__':  ## se define el main del codigo
    path = sys.argv[1]    ## Se define el path de la imagen
    image_name = sys.argv[2]  ##Se define el nombre de la imagen
    path_file = os.path.join(path, image_name) ## Se genera la ruta de la imagen para leerla
    image = cv2.imread(path_file)  ## Se define la imagen
    banco_filtros(image)  ## Se usa la funcion bamco_filtros ingresando la imagen deseada
