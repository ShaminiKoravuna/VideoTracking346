'''
DVS346 Tracking using CAMShift for Pynq using DVSAbsLib library and Time Sections library.

'''
import sys
print("Python executable being used:", sys.executable)

import numpy as np
import cv2
import timesectionslib as ts




import dvsabslib as dal
import timesectionslib as ts
#import matplotlib
#matplotlib.use('Agg')


from time import time
import random
import math
from numpy.core.fromnumeric import mean, resize
import matplotlib



# Inicialización de la librería DVSAbsLib
dvslib = dal.DVSAbsLib(max_packet_interval=20000, hdmi_resolution=(640, 480), pynq=True, noise_filter=True)
dvslib.init()
dvslib.device_info()



# Inicialización de la librería para medir tiempos por secciones y totales
timeslib = ts.TimeSections(['get image', 'umbral', 'apertura', 'tracking', 'cam shift', 'show image'])
timeslib.init()



# CONSTANTES
DVS_RES_W = 346
DVS_RES_H = 260
N_TRAZO = 30

puntos = list() # Lista de centroides para dibujar la trayectoria




# Localización inicial de la primera ventana para el tracking
x, y, w, h = int((2/3)*DVS_RES_W), int((2/3)*DVS_RES_H), int(DVS_RES_W/3), int(DVS_RES_H/3)  # Valor central de la imagen
track_window = (x, y, w, h)  # Creación de la ventana

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


print('\nRUNNING\n')


while True:
    try:


        timeslib.start_total()



        ''' CAPTACIÓN DE LA IMAGEN DEL DVS '''

        timeslib.start_time('get image')

        [img, events] = dvslib.get_image('hist')
        timeslib.events_number(events)
        
        timeslib.end_time('get image')






        ''' PRE-PROCESAMIENTO DE LA IMAGEN '''


        timeslib.start_time('umbral')

        # Aplicamos un umbral
        mascara_gris = np.full((DVS_RES_H, DVS_RES_W), 127, np.uint8)
        umbral = cv2.threshold(img - mascara_gris, 0, 255, cv2.THRESH_BINARY)[1]

        timeslib.end_time('umbral')




        timeslib.start_time('apertura')
    
        # Apertura para eliminar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        apertura = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)

        # Dilatamos la apertura para tapar agujeros
        dilatada = cv2.dilate(apertura, None, iterations=2)

        timeslib.end_time('apertura')




        # Imagen final a mostrar
        res = img.copy()

        # Copiamos la imagen dilatada para realizar el tracking
        frame = dilatada.copy()









        ''' ALGORITMO DE TRACKING (CAM SHIFT) '''
    
        timeslib.start_time('tracking')
            
        # Extrae la ROI del frame
        roi = frame[y:y+h, x:x+w]
        # Calcula su histograma
        roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
        # Normaliza el histograma
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Back Projection is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model
        dst = cv2.calcBackProject([frame], [0], roi_hist, [0, 180], 1)




        timeslib.start_time('cam shift')

        # Aplica el algoritmo CAMshift al frame, con la ventana dada y los criterios de parada dados
        ret, track_window = cv2.CamShift(frame, track_window, term_crit)

        timeslib.end_time('cam shift')




        # Dibuja el bounding box encima del objeto trackeado
        pts = cv2.boxPoints(ret)
        centroide = (int((pts[0][0] + pts[2][0]) / 2), int((pts[0][1] + pts[2][1]) / 2))  # Calcula las coordenadas del centroide del bounding box como la media de la diagonal AC
        pts = np.int0(pts)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

        # Si hay un objeto detectado
        if not (centroide == (0, 0)):
            res = cv2.polylines(res, [pts], True, (0, 255, 0), 1)  # Dibuja el bounding box del objeto
            res = cv2.circle(res, centroide, 1, (0, 0, 255), -1)  # Dibuja el centroide del objeto


        # Permite dibujar la trayectoria de manera independiente
        #trayectoria = np.full((DVS_RES_H, DVS_RES_W), 127, np.uint8)
        #trayectoria = cv2.cvtColor(trayectoria, cv2.COLOR_GRAY2BGR)

        # Cuando el objeto de va de la escena, se deja de registrar pero no se borra la trayctoria anterior
        if not (centroide == (0, 0)):
            puntos.append(centroide)
            if len(puntos) > N_TRAZO:
                puntos.pop(0)
        else:
            if len(puntos) > 0:
                puntos.pop(0)

        # Dibuja la trayectoria del movimiento del objeto
        for i in range(0, len(puntos) - 1):
            cv2.line(res, puntos[i], puntos[i + 1], (0, 255, 255), 1)


        timeslib.end_time('tracking')










        ''' MUESTRA LA IMAGEN '''

        timeslib.start_time('show image')

        # Muestra la imagen final por la interfaz HDMI
        dvslib.hdmi_write(res)
        
        timeslib.end_time('show image')





        
        timeslib.end_total()






    # Cuando se interrumpe la ejecución del código, se cierra la conexión con el DVS
    except KeyboardInterrupt:
        dvslib.hdmi_close()
        dvslib.device.shutdown()
        print('\nSTOPPED\n')
        timeslib.show_stats('us')
        print('\nGenerating graphs...\n')
        timeslib.graphics(['get image', 'umbral', 'apertura', 'tracking', 'cam shift', 'show image'], title='Tiempos Pynq Z2')
        print('\nReady! FINISHED\n')
        break
