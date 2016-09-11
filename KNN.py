# -*- coding: utf-8 -*-
'''
Metodos de Aprendizaje Automatico
Facultad de Ingenieria (UdelaR), 2016
Practico 3, Ejercicio 7

@author: 
    Erguiz, Daniel        4.554.025-3
    Mechulam, Nicolas     4.933.997-7
    Salvia, Damian        4.452.120-0
    
@summary: 
   Modela el algoritmo K Nearest Neighbors.

@attention: 
    ejemplos  : [{a1:v1,...,aN:vN}], donde a:Atributo y v:Valor  
'''

from math import sqrt
from collections import defaultdict
from itertools import groupby

class KNN:
    
    # Comunes
    examples = [ # Proposito de test. Tomado del libro.
            {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'},
            {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Fuerte" ,"JugarTenis":'-'},
            {"Cielo":"Nubes" , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'+'},
            {"Cielo":"Lluvia", "Temperatura":"Suave", "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'+'},
            {"Cielo":"Lluvia", "Temperatura":"Baja" , "Humedad":"Normal", "Viento":"Debil"  ,"JugarTenis":'+'},
            {"Cielo":"Lluvia", "Temperatura":"Baja" , "Humedad":"Normal", "Viento":"Fuerte" ,"JugarTenis":'-'},
            {"Cielo":"Nubes" , "Temperatura":"Baja" , "Humedad":"Normal", "Viento":"Fuerte" ,"JugarTenis":'+'},
            {"Cielo":"Sol"   , "Temperatura":"Suave", "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'},
            {"Cielo":"Sol"   , "Temperatura":"Baja" , "Humedad":"Normal", "Viento":"Debil"  ,"JugarTenis":'+'},
            {"Cielo":"Lluvia", "Temperatura":"Suave", "Humedad":"Normal", "Viento":"Debil"  ,"JugarTenis":'+'},
            {"Cielo":"Sol"   , "Temperatura":"Suave", "Humedad":"Normal", "Viento":"Fuerte" ,"JugarTenis":'+'},
            {"Cielo":"Nubes" , "Temperatura":"Suave", "Humedad":"Alta"  , "Viento":"Fuerte" ,"JugarTenis":'+'},
            {"Cielo":"Nubes" , "Temperatura":"Alta" , "Humedad":"Normal", "Viento":"Debil"  ,"JugarTenis":'+'},
            {"Cielo":"Lluvia", "Temperatura":"Suave", "Humedad":"Alta"  , "Viento":"Fuerte" ,"JugarTenis":'-'}
        ]
    values  = defaultdict(set)
    tA      = None
    
    # Exlcusivos
    K = 1
	
    
    def __init__(self,tA,ejemplos=None,parm1=None):
        # Si dan ejemplos nuevos, instanciarlos
        if ejemplos: self.examples = ejemplos
        
        # Si se especifico k se setea como  cantidad de vecinos a considerar
        if parm1: self.K = parm1
        
        # Guardar el atributo objetivo
        self.tA = tA
        
        # Extraer los valores de atributos para cada dato
        for example in self.examples:
            for atributo,valor in example.items():
                self.values[atributo].add(valor)
	
    	
    def distance(self,example1,example2):

        def one_hot_encoding(att,val):
            # Para una pareja (att,val) crea un vector según los valores de los ejemplos
            # en donde vector_i = True si valores_ejemplo[att] = valor sino False. 
            # Se agrega una dimension mas para contemplar el caso de que el valor nunca haya sido visto
            # que adopta el valor True si _no_ esta en los ejemplos, sino False
            encode = [True if v==val else False for v in self.values[att]]
            encode.append(not (val in self.values[att]))
            return encode
        
        def euclidean(v1,v2): # @TODO Al final no lo use
            # Calcula distancia euclidea enrte ds vectores
            return sqrt(sum((y-x)**2 for x,y in zip(v1,v2)))
        
        # Calcular el one_hot_enconding para cada ejemplo
        e1 = map(lambda (att,val):one_hot_encoding(att,val), example1.items())
        e2 = map(lambda (att,val):one_hot_encoding(att,val), example2.items())
        
        # Para cada codificación (e_i) calcular la distancia entre ellos para obtener la distancia total
        return sqrt(sum((x-y)**2 for v1,v2 in zip(e1,e2) for x,y in zip(v1,v2)))
	
    
    def classify(self,new_exaple):
        # Devuelve la categoria para el nuevo ejemplo        
        # Obtener los k vecinos mas cercanos
    	ordenados = sorted(self.examples, key=lambda example : self.distance(new_exaple,example))
    	kvecinos = ordenados[:self.K]
    	#esta parte es una opcion diseño, no se si sera lo correcto
    	# aunque es discreto el target_attribute para aproximar mejor la solucion aplico como si fuera una solucion real
    	# devuelvo el promedio de los k vecinos 
#         salida = 0
#         for vecino in k_cercanos:
#             salida += vecino[self.target_attribute]
#         
#         return int(salida/self.k)
        
        # Agrupar por clases segun el atributo objetivo
        agrupados = groupby(kvecinos, key=lambda s : s[self.tA])
        
        # Devolver el valor cuya clase sea mayoritaria en los k vecinos
        return max(agrupados, key=lambda x:len(list(x[1])))[0]
    
    
    def predicts(self,example):
        # 1 si predice ok, 0 sino
        valor = example[self.tA]
        res = self.classify(example)
        return 1 if valor == res else 0
    
    
    def __str__(self):
        return ""
    

# TODO - Duda: Se deberia incluir el target att en el calculo? De momento se considera, pero para mi no esta bien
# Test
# tA = "JugarTenis"
# inst = KNN(tA,parm1=3)
# 
# print inst.distance(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}, 
#     {"Cielo":"Sol"   , "Temperatura":"Baja" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
# 
# print inst.classify(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
# 
# print inst.predicts(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )