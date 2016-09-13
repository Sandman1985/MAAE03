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
   Modela el algoritmo Naive Bayes.

@attention: 
    ejemplos  : [{a1:v1,...,aN:vN}], donde a:Atributo y v:Valor  
'''


from collections import defaultdict
from operator import mul
from bcolz.ctable import ctable


class NB:
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
    values   = defaultdict(set)
    tA       = None
    
    # Exlcusivos
    m   = 0
    Pv  = defaultdict(lambda:0) # Vector P(v_j)
    Pav = defaultdict(lambda:defaultdict(lambda:[])) # Matriz P(a_i|v_j)
    
    
    def __init__(self,tA,ejemplos=None,parm1=None):
        # Si dan ejemplos nuevos, instanciarlos
        if ejemplos: self.examples = ejemplos
        
        # Si se especifico m se setea como "equivalent sample size"
        if parm1: self.m = parm1
        
        # Guardar el atributo objetivo
        self.tA = tA
		
        # Extraer los valores de atributos para cada dato
        ctA, cCond = defaultdict(lambda:0), defaultdict(lambda:0)
        for example in self.examples:
            tAval = example[self.tA]
            ctA[tAval] += 1
            for atributo,valor in example.items():
                self.values[atributo].add(valor)
                cCond[(atributo,valor,tAval)] += 1
		
#         print "ctA",ctA
#         print "cCond",cCond
        
        # Calcular probabilidades
        total = len(self.examples)        
        for tAval in self.values[self.tA]: # Probabilidad de salida
            self.Pv[tAval] = 1.0 * ctA[tAval] / total
            
        for atributo in self.values.keys(): # Probabilidades condicionadas
            if atributo != self.tA:
                p = 1.0 / len(self.values[atributo])
                for valor in self.values[atributo]:
                    for tAval in self.values[self.tA]:
                        # aplica estimacion de probabilidades, suponiendo una predisposicion equitativa entre los valores de un atributo
                        if (ctA[tAval] + self.m) == 0: raise Exception("%s has no value %s" % (self.tA,str(tAval)))
                        self.Pav[tAval][valor] = 1.0 * (cCond[(atributo,valor,tAval)] + (self.m * p)) / (ctA[tAval] + self.m)    
        
#         print "Pv",len(self.Pv),self.Pv
#         print "Pav",len(self.Pav),self.Pav    
    
    
    def classify(self,new_example):
        # Devuelve la categoria para el nuevo ejemplo
        def prod(factors): # Hace productoria de factores
            return reduce(mul, factors, 1)
        return max(self.values[self.tA], key=lambda val: self.Pv[val] * prod(self.Pav[val].values()))
    
    
    def predicts(self,new_example):
        # 1 si predice ok, 0 sino
        valor = new_example[self.tA]
        res = self.classify(new_example)
        return 0 if valor == res else 1
    

# Test
# tA = "JugarTenis"
# inst = NB(tA,parm1=0)
# 
# print inst.classify(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
# 
# print inst.predicts(
#     {"Cielo":"Sol"   , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'-'}
# )
		