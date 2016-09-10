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


class K-NN:

	examples  = None
	k = 1
	target_attribute = None

	
    def __init__(self,ejemplos=None,k =None,t_attribute=None):
        '''
        Instancia la clase, indicando opcionalmente un dataset de ejemplos  
        '''
        # Si dan ejemplos nuevos, instanciarlos
        if ejemplos: self.examples = ejemplos
        
        # Si se especifico k se setea como  cantidad de vecinos a considerar
        if k: self.k= k
        
		if t_attribute : target_attribute = t_attribute
		
	def calcular_distancia(example1 , example2):
	# calcula distancia euclidea
   
       # Debe ser recontra choto, pero no encuentro una forma linda de hacerlo
	
	return
	
	def clasificar(elemento):
		ordenado = sorted(self.examples, key=lambda example : calcualar_distancia(elemento,example))
		k_cercanos = ordenados[:self.k]
		#esta parte es una opcion diseño, no se si sera lo correcto
		# aunque es discreto el target_attribute para aproximar mejor la solucion aplico como si fuera una solucion real
		# devuelvo el promedio de los k vecinos 
		salida = 0
		for vecino in k_cercanos:
			salida = vecino[target_attribute] + salida
		
		return int(salida/self.k)

