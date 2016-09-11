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
   Modela el algoritmo ID3 para Decision Tree.

@attention: 
    ejemplos  : [{a1:v1,...,aN:vN}], donde a:Atributo y v:Valor  
'''

from math import log
from collections import defaultdict
from itertools import groupby
import sys  
reload(sys)  
sys.setdefaultencoding('utf-8')

FORK = u'\u251c'
LAST = u'\u2514'
VERTICAL = u'\u2502'
HORIZONTAL = u'\u2500'
VACIO = u' '

class DT:
    
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
    values     = defaultdict(set)
    tA         = None
    
    # Exlcusivos
    max_depht  = float('inf')
	
    
    def __init__(self,tA,ejemplos=None,max_prof=None):
        # Si dan ejemplos nuevos, instanciarlos
        if ejemplos: self.examples = ejemplos
        
        # Dar limite de profundidad para poda
        if max_prof: self.max_depht = max_prof
        
        # Guardar el atributo objetivo
        self.tA = tA
        
        # Extraer los valores de atributos para cada dato
        for example in self.examples:
            for atributo, valor in example.items():
                self.values[atributo].add(valor)
        
        Atts = [A for A in self.examples[0].keys() if A != tA]
        self.tree = self.ID3(Atts,self.examples)
     
                		    
    def ID3(self,attributes,S, depth=0):
        # Computa el algoritmo ID3
        
        def mas_comun(tA,S=self.examples):
            # Determina cual es el valor mas comun para el atributo objetivo
            # para un determinado subconjunto de ejemplos
            polaridades = map(lambda s: s[tA],S) 
            return max(polaridades, key=polaridades.count)
        
        def subS(v,S,A):
            # Calcula subconjunto de S donde el atributo A tiene valor v
            return [s for s in S if s[A] == v]
        
        # Si todos los ejemplos tiene la misma polaridad, returna nodo con esa polaridad
        if len(self.values[self.tA])==1:
            return Nodo(self.values[self.tA][0])
        
        # Si no hay mas atributos o alcanza un max, retorna el mas comun
        if not attributes or (self.max_depht and depth > self.max_depht):
            return Nodo(mas_comun(self.tA,S))
        
        # En otro caso
        def mejor_clasifica(As,tA):            
            def information_gain(A,tA,S):
                def entropy(S,tA):                    
                    def subclases(S):
                        ordenado = sorted(S, key=lambda s : s[tA])
                        agrupado = groupby(ordenado, key=lambda s : s[tA])
                        # Obtiene las sublclases de S segun su polaridad
                        return [[y for y in list(x[1])] for x in agrupado]
                    # Calcula la entropia de S a partir de las subclases
                    return sum(-(1.0*len(subclass)/len(S))*log((1.0*len(subclass)/len(S)),2) for subclass in subclases(S))
                # Calcula Information Gain a partir de la entropia y las sublcases
                return entropy(S,tA) - sum(entropy(subS(v,S,A),tA) * len(subS(v,S,A))/len(S) for v in self.values[A])
            # Determina cual es el mejor atributo que clasifica a los ejemplos segun el Information Gain
            return max(As, key=lambda A : information_gain(A,tA,self.examples))
               
        A = mejor_clasifica(attributes,self.tA)
        raiz = Nodo(A,mas_comun(self.tA,S))
        for value in self.values[A]: # Buscar hijos
            ejemplos_v = subS(value, S, A)
            if not ejemplos_v:
                hijo = Nodo(mas_comun(self.tA))
            else:
                attributes.remove(A)
                hijo = self.ID3(attributes, S=ejemplos_v,depth=depth+1)
                attributes.append(A)
            raiz.add_hijo(hijo,value)                
        
        return raiz
    
    
    def classify(self,new_example):
        return self.tree.classify(new_example,self.tA)
    
        
    def predicts(self,example):
        # Obtener el resultado real
        valor = example[self.tA]
        res = self.classify(example)
        return 1 if valor == res else 0

class Nodo:
# Mantiene la estructura del arbol de decision
    
    def __init__(self,dato,mas_comun=None):
        self.dato       = dato
        self.mas_comun  = mas_comun
        self.hijos      = {}
    
    
    def add_hijo(self,subarbol,etiqueta):
        self.hijos.update({etiqueta:subarbol})
    
        
    def classify(self,test,target):
        if self.hijos:
            valor = test[self.dato]
            if self.hijos.has_key(valor):
                return self.hijos[valor].classify(test,target)
            else: # Tengo un valor nuevo atributo (nunca visto) de los ejemplos de entrenamiento
                return self.mas_comun
        else:
            return self.dato
    
    
    def __str__(self):
        def explore(self, prefijo='',nodos=0):        
            hijos = self.hijos.items()
            next_prefijo = u''.join([prefijo,VERTICAL,u'    '])
            ret = u''.join([u'\n',prefijo,self.dato])
            max_prof = 0
            for etiqueta,subarbol in hijos[:-1]:
                ret += u''.join([u'\n',prefijo,FORK,HORIZONTAL,HORIZONTAL,u' (-',etiqueta,u'-)'])
                ret += u''.join([u'\n',prefijo,VERTICAL,u'    ',VERTICAL])
                tree, nodos, prof = explore(subarbol,next_prefijo,nodos)
                if max_prof < prof : max_prof = prof 
                ret += u''.join([tree])
            if hijos:
                etiqueta, subarbol = hijos[-1]
                last_prefijo = u''.join([prefijo,VACIO,u'    '])
                ret += u''.join([u'\n',prefijo,LAST,HORIZONTAL,HORIZONTAL,u' (-',etiqueta,u'-)'])
                ret += u''.join([u'\n',prefijo,VACIO,u'    ',VERTICAL])
                tree, nodos, prof = explore(subarbol,last_prefijo,nodos)
                if max_prof < prof : max_prof = prof
                ret += u''.join([tree])
            else:
                ret += u''.join([u'\n',prefijo])
            return ret,nodos+1,max_prof+1
        
        tree, nodes, depth = explore(self)
        nodes = u"NODOS:\n%i" % nodes
        depth = u"PROFUNDIDAD:\n%i" % depth
        tree  = u"ARBOL:%s" % tree       
        return u'\n'.join([nodes, depth, tree])


# Test
tA = "JugarTenis"
inst = DT(tA)

print inst.classify(
    {"Cielo":"Nubes" , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'+'}
)

print inst.predicts(
    {"Cielo":"Nubes" , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'+'}
)
