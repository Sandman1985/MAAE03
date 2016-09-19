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
	
    
    def __init__(self,tA,ejemplos=None,parm1=None):
        # Si dan ejemplos nuevos, instanciarlos
        if ejemplos: self.examples = ejemplos
        
        # Dar limite de profundidad para poda
        if parm1: self.max_depht = parm1
        
        # Guardar el atributo objetivo
        self.tA = tA
        
        # Extraer los valores de atributos para cada dato
        for example in self.examples:
            for atributo,valor in example.items():
                self.values[atributo].add(valor)
        
        Atts = [A for A in self.examples[0].keys() if A != tA]
        self.tree = self.decision_tree(Atts)
     
                		    
    def decision_tree(self,attributes):
        # Computa el algoritmo ID3
        
        def mas_comun(S=self.examples):
            # Determina cual es el valor mas comun para el atributo objetivo
            # para un determinado subconjunto de ejemplos
            polaridades = map(lambda s: s[self.tA],S) 
            return max(polaridades, key=polaridades.count)
        
        def subS(v,S,A):
            # Calcula subconjunto de S donde el atributo A tiene valor v
            return [s for s in S if s[A] == v]
        
#         def subclasses(S):
#             ordenado = sorted(S, key=lambda s : s[self.tA])
#             agrupado = groupby(ordenado, key=lambda s : s[self.tA])
#             # Obtiene las sublclases de S segun su polaridad
#             return [[y for y in list(x[1])] for x in agrupado]
        
        def subclasses(S):
            count = defaultdict(lambda:0)
            for s in S: # Cuenta la cantidad por clase
                val = s[self.tA]
                count[val] += 1
            return count
        
        def entropy(S):   
            # Calcula la entropia de S a partir de las subclases
#             return sum(-(1.0*len(subclass)/len(S))*log((1.0*len(subclass)/len(S)),2) for subclass in subclasses(S))
            acc,lenS = 0,len(S)
            for lenSubS in subclasses(S).values():
                acc += -(1.0*lenSubS/lenS) * log((1.0*lenSubS/lenS),2)
            return acc
        
        def information_gain(A,S):
            # Calcula Information Gain a partir de la entropia y las sublcases
            acc = 0
            for v in self.values[A]:
                vSA = subS(v,S,A)
                acc += entropy(vSA) * len(vSA)/len(S)
            return entropy(S) - acc
        
        def mejor_clasifica(As):            
            # Determina cual es el mejor atributo que clasifica a los ejemplos segun el Information Gain
            return max(As, key=lambda A : information_gain(A,self.examples))
        
        def ID3(atts,S,depth):
            # Si todos los ejemplos tiene la misma polaridad, returna nodo con esa polaridad
            if len(self.values[self.tA])==1:
                return Nodo(list(self.values[self.tA])[0])
            
            # Si no hay mas atributos o alcanza un max, retorna el mas comun
            if not atts or depth > self.max_depht:
                return Nodo(mas_comun(S))
            
            # En otro caso       
            A = mejor_clasifica(atts)
            raiz = Nodo(A,mas_comun(S))
            for value in self.values[A]: # Buscar hijos
                ejemplos_v = subS(value,S,A)
                if not ejemplos_v:
                    hijo = Nodo(mas_comun())
                else:
                    atts.remove(A)
                    hijo = ID3(atts,S=ejemplos_v,depth=depth+1)
                    atts.append(A)
                raiz.add_hijo(hijo,value)                
            
            return raiz
        
        return ID3(attributes,self.examples,0)
    
    def classify(self,new_example):
        # Devuelve la categoria para el nuevo ejemplo
        return self.tree.classify(new_example,self.tA)
    
        
    def predicts(self,example):
        # 1 si predice ok, 0 sino
        valor = example[self.tA]
        res = self.classify(example)
        return 0 if valor == res else 1
    
    
    def __str__(self):
        return self.tree.__str__()



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
# tA = "JugarTenis"
# inst = DT(tA,parm1=5)
# 
# print inst.classify(
#     {"Cielo":"Nubes" , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'+'}
# )
# 
# print inst.predicts(
#     {"Cielo":"Nubes" , "Temperatura":"Alta" , "Humedad":"Alta"  , "Viento":"Debil"  ,"JugarTenis":'+'}
# )
