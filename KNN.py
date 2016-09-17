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
    K     = 1
    mu    = defaultdict(lambda:0)
    sigma = defaultdict(lambda:0)


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

        # Obtener mu y sigma para poder normalizar
        for att,vals in self.values.items():
            if all(type(val) is int for val in vals): # Solo cuando los valores sean enteros
                size = len(vals)
                self.mu[att]    =      1.0/size * sum(vals)
                self.sigma[att] = sqrt(1.0/size * sum((val - self.mu[att])**2 for val in vals))


    def distance(self,example1,example2):

        def normalize(att,val):
            # Normaliza el valor para el atributo en la muestra
            if self.sigma[att] == 0 : print att,val,self.mu[att]
            return {att : 1.0/self.sigma[att] * (val - self.mu[att])}

        def one_hot_encoding(att,val):
            # Codifica un vector cuyos valores pueden ser enumerados
            # en un vector "aplanado" de dimension mayor con valores booleanos
            # p.e. <...,A:Vi,...> -> <...,AisV1:False,...,AisVi:True,...,AisVn:False,...>
            return {"%s_is_%s"%(att,pval) : (1 if pval==val else 0) for pval in self.values[att]}

        def euclidean(v1,v2):
            # Calcula distancia euclidea enrte dos vectores
            return sqrt(sum((y-x)**2 for x,y in zip(v1,v2)))

        # Obtener vectores codigicados segun los valores de los atributos
        e1,e2 = {},{}
        for att in example1:
            # Si el valor es entero lo normaliza segun sigma y mu
            if type(example1[att]) is int:
                e1.update(normalize(att, example1[att]))
                e2.update(normalize(att, example2[att]))
            # Si el valor es string (categorial) calcula el one-hot-encodig aplanado
            if type(example1[att]) is str:
                e1.update(one_hot_encoding(att,example1[att]))
                e2.update(one_hot_encoding(att,example2[att]))

        # Para cada codificación calcular la distancia
        return euclidean(e1.values(),e2.values())


    def classify(self,new_exaple):
        # Devuelve la categoria para el nuevo ejemplo
    	ordenados = sorted(self.examples, #
                           key = lambda e: self.distance(new_exaple,{i:e[i] for i in e if i != self.tA}))

        # Obtener los k vecinos mas cercanos
    	kvecinos = ordenados[:self.K]

        # Agrupar por clases segun el atributo objetivo
        agrupados = groupby(kvecinos, key=lambda s: s[self.tA])

        # Devolver el valor cuya clase sea mayoritaria en los k vecinos
        # TODO - Si tengo mas de un maximo se queda con el primero que tenga
        return max(agrupados, key=lambda x:len(list(x[1])))[0]


    def predicts(self,example):
        # 1 si predice ok, 0 sino
        valor = example.pop(self.tA) # Quitar el atributo objetivo
        res = self.classify(example)
        example.update({self.tA:valor}) # Volver a colocar el atributo objetivo
        return 0 if valor == res else 1


# TODO - Duda: Se deberia incluir el target att en el calculo? De momento se considera, pero para mi no esta bien
# Test
# tA = "JugarTenis"
# inst = KNN(tA,parm1=3)
#
# print inst.distance(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'},
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
#
# print inst.classify(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
#
# print inst.predicts(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
