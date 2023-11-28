#========================================================================================================================

'''
    Modelo de aprendizaje de máquina para el tamizaje de casos de PTB utilizando CXR

    Grupo2:
        * María Paula Cabezas Charry
        * Juan David Carvajal Cucuñame
        * Ángel Gabriel Pasaje Erazo
'''

#========================================================================================================================

# Bibliotecas y Módulos

from flask import Flask, render_template, request, redirect, send_file
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

#========================================================================================================================

app = Flask(__name__)

# Constantes de configuración
app.config['CARPETA_DE_CARGA'] = 'static/images'
app.config['EXTENSIONES_DE_ARCHIVO_PERMITIDAS'] = {'png', 'jpg', 'jpeg'}
app.config['ETIQUETAS_DE_CLASES'] = ["Normal", "Tuberculosis"] 

# Cargar el modelo entrenado al ejecutar la app
model = tf.keras.models.load_model('model.h5')

#========================================================================================================================

# Función reutilizada para comprobar el formato de la imagen
def archivo_permitido(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['EXTENSIONES_DE_ARCHIVO_PERMITIDAS']

# Función para aplicar el filtro de mejoramiento de contraste
def aplicar_filtro_CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Convertir a espacio de color LAB
    l, a, b = cv2.split(lab) # Dividir para aplicar filtro
    l = clahe.apply(l) # aplicar filtro a l -> luminosidad de negro a blanco
    lab = cv2.merge((l, a, b)) # combinar nuevamente
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # Convertir de nuevo la imagen a espacio de color RGB

#=====================================================================================================================================

# Rutas de la aplicación

@app.route('/', methods=['GET', 'POST'])
def index():

    # Comprobar método POST HTTP
    if request.method == 'POST':

        # Comprobar que la imagen haya sido cargada en la GUI y no esté vacío su nombre
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']

        # Comprobar que no esté vacío el nombre de la imagen
        if file.filename == '':
            return redirect(request.url)
        
        # Comprobar que la imagen tenga un tipo de formato permitido
        if file and archivo_permitido(file.filename):

            filename = secure_filename(file.filename) # Asegurar formato del nombre del archivo subido por usuario
            ruta_de_imagen = os.path.join(app.config['CARPETA_DE_CARGA'], filename) # Crear la ruta para guardar la imagen en servidor
            file.save(ruta_de_imagen) # Guardar la imagen en la ruta definida

            image = cv2.imread(ruta_de_imagen) # Leer la imagen almacenada en servidor
            imagen_procesada = aplicar_filtro_CLAHE(image) # Aplicar el filtro de mejora de contraste a la imagen

            # Guardar en servidor la imagen procesada en la ruta especificada con un nombre y formato específicos
            nombre_imagen_original = filename.split('.')[0] # Quitar el '.png' de la imagen
            cv2.imwrite(os.path.join(app.config['CARPETA_DE_CARGA'], f'{nombre_imagen_original}_procesada.jpg'), imagen_procesada) 

            imagen_procesada = cv2.resize(imagen_procesada, (128, 128))  # Redimensionar a 128x128 pixeles -> formato entrada modelo
            imagen_procesada = imagen_procesada / 255.0  # Normaliza la imagen -> Rango de 0 a 1

            # Predecir la clase de la imagen procesada
            prediccion = model.predict(np.array([imagen_procesada])) # Pasar la imagen procesada como arreglo a la entrada del modelo
            #print(prediccion)
            indice_clase = np.argmax(prediccion[0]) # Encontrar el valor máximo del arreglo para pasarlo como índice

            # Obtener la etiqueta de clase basada en el índice
            clase_predicha = app.config['ETIQUETAS_DE_CLASES'][indice_clase]

            return render_template('index.html', 
                                   bandera_imagen_procesada=True, 
                                   imagen_original=filename, 
                                   imagen_procesada=f'{nombre_imagen_original}_procesada.jpg', 
                                   clase_predicha=clase_predicha
                                  )
    return render_template('index.html', bandera_imagen_procesada=False) # Para que no se cargue el contenedor de datos si no hay 'POST'

#========================================================================================================================

# Ejecutar la app
if __name__ == '__main__':
    app.run()

#========================================================================================================================