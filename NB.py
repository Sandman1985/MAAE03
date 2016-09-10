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


class NB:
    # Dataset - Proposito de test. Tomado del libro.
    examples = [
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
    values    = defaultdict(set)
    examples  = None
    eq_ss	  = 0
    target_attribute = None
    # diccionario donde key es una tupla (atributo_i,a_i,salida_i) y value es P(a_i|salida_i) 
    prob_condicionadas = defaultdict(set)
    # diccionario donde key  es salida_i  y value es P(salida_i)
    prob_salida = defaultdict(set)
    
    def __init__(self,ejemplos=None,m =None,t_attribute=None):
        '''
        Instancia la clase, indicando opcionalmente un dataset de ejemplos  
        '''
        # Si dan ejemplos nuevos, instanciarlos
        if ejemplos: self.examples = ejemplos
        
        # Si se especifico m se setea como  equivalent sample size
        if m: self.eq_ss = m
        
        if t_attribute : target_attribute = t_attribute
		
        # Extraer los valores de atributos para cada dato
        for example in self.examples:
            for atributo, valor in example.items():
                self.values[atributo].add(valor)
		
		self.calcular_probabilidades()
	
	
	def calcular_probabilidades(): 
	# Calcula las probabilidades necesarias para aplicar NB
	# P(salidad_i) -> probabilidad de cada valor del target_attribute
	# P(a_i|salida_i) -> probabilidad de que ocurra a_i condicionada a salida_i
		
		# cantidad de ejemplos en el conjunto de datos de entrad
		cant_examples = len(self.examples)
			
		#cantidad de ejemplos para cada valor posible del target_attribute
		#es un diccionario {salida: cant}
		cant_examples_por_salida = defaultdict(set)			
			
		# diccionario donde key es una tupla (atributo_i, a_i,salida_i) y value es la cantidad de examples donde 
		# el valor del atributo i vale a_i y el valor del target_attribute vale salida_i
		cant_condicionadas =defaultdict(set)
			
		# inicializacion de cant_condicionadas con todos los values en 0			
		for atributo in self.values.keys():
			if atributo != self.target_attribute :
				for valor in self.values[atributo]:
					for  salida in self.values[self.target_attribute]:
						cant_condicionadas[(atributo,valor,salida)] = 0
			
		for example in self.examples:
			salida = example[self.target_attribute]			
			cant_examples_por_salida[salida]+= 1
			for atributo, valor in example.items(): 
				cant_condicionadas[(atributo,valor,salida)] = += 1
			
		#Calcula P(salidad_i)
		for salida in self.values[self.target_attribute]:
			prob_salida[salida] = cant_examples_por_salida[salida] / cant_examples
			
		# Calcula P(a_i|salida_i)
		for atributo in self.values.keys():
			if atributo != self.target_attribute :
				p = 1/ self.values[atributo].size
				for valor in self.values[atributo]:
					for  salida in self.values[self.target_attribute]:
						# aplica estimacion de probabilidades, suponiendo una predisposicion equitativa entre los valores de un atributo
						prob_condicionadas[(atributo,valor,salida)] = (cant_condicionadas[(atributo,valor,salida)] + (self.eq_ss * p ))/ (cant_examples_por_salida[salida] + self.eq_ss)
	
	def clasificar(elemento):
	# retorna el valor estimado para el target_attribute
		target_value = None
		prob_target_value = 0
		# busca el valor del target_attribute que maximice  P(salidad_i)II P(a_i|salida_i)
		for  salida in self.values[self.target_attribute]:
			prob_actual = prob_salida[salida]
			for atributo, valor in example.items():
				if atributo != self.target_attribute:
					prob_actual = prob_actual * prob_condicionadas[(atributo ,valor,salida)]
			if prob_actual > prob_target_value:
				prob_target_value = prob_actual
				target_value = salida
		return (target_value,prob_target_value)




		