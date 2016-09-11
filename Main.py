# -*- coding: utf-8 -*-
'''
Metodos de Aprendizaje Automatico
Facultad de Ingenieria (UdelaR), 2016
Practico 2, Ejercicio 6

@author: 
    Erguiz, Daniel        4.554.025-3
    Mechulam, Nicolas     4.933.997-7
    Salvia, Damian        4.452.120-0
    
@summary: 
       Obtiene los datos del dataset y los procesa
       a fin de obtener el arbol de desicion
'''

from DT  import DT
from NB  import NB
from KNN import KNN

from csv    import DictReader
from random import shuffle
from copy   import deepcopy
from os     import remove
from os.path import exists

def data(csv_file,boolean,without):
    # Obtiene los ejemplos a partir de un archivo *.csv
    with open(csv_file, mode='r') as f:
        D = [row for row in DictReader(f,delimiter=";")]
        # Procesamiento segun las flags
        if boolean: # Transformar a booleano - Update no retorna valores, por eso no se asigna a nada
            [d.update({'absences':('Aceptable' if int(d['absences']) < 10 else 'Muchas')}) for d in D]
            [d.update({'G1':('Malo' if int(d['G1']) < 12 else 'Bueno')}) for d in D]
            [d.update({'G2':('Malo' if int(d['G2']) < 12 else 'Bueno')}) for d in D]
            [d.update({'G3':('Malo' if int(d['G3']) < 12 else 'Bueno')}) for d in D]
            [d.update({'age':('Menor' if int(d['age']) < 18 else 'Adulto')}) for d in D]
        if without: # Remover G1 y G2
            D = [{k:v for k,v in d.items() if k!='G1' and k!='G2' } for d in D]
        return D 

        
def delta(train_set,test_set,tA,parm):    
    # Calcula el error del algoritmo
    hypotesis = parm["inst"](tA,ejemplos=train_set,parm1=parm["parm1"][0])
    return 1.0*sum(hypotesis.predicts(test) for test in test_set)/len(test_set)

    
def cross_validation(S,tA,parm):
    # Realiza KFold cross-validation de tamanio K sobre el conjunto S
    # a fin de obtener el delta estimado y la varianza
    
    def KFold(S,K): # K-Fold Cross Validation
        for k in xrange(K):
            train = [x for i,x in enumerate(S) if i % K != k]
            test  = [x for i,x in enumerate(S) if i % K == k]
            yield train, test
            
    def LOO(S): # Leave-One-Out Cross Validation
        for k in xrange(len(S)):
            train = [x for i,x in enumerate(S) if k != i]
            test  = [x for i,x in enumerate(S) if k == i]
            yield train, test
        
    # Entrenar
    CVset = list(KFold(S,parm["KFold"]) if parm["KFold"] else LOO(S))
    deltas, cont, total = [], 1, len(CVset)
    for train, test in CVset:
        print "\r[CROSS-VALIDATION] Progress: %i/%i" % (cont,total),
        current_delta = delta(train,test,tA,parm)
        deltas.append(current_delta)
        cont += 1
    
    estimated = sum(deltas)/total
    variance = 1.0*sum([(d - estimated )**2 for d in deltas])/total
    return estimated, variance

    
def process(D,tA,parm,slice=0.2):
    # Procesa un conjunto de datos D a partir de un atributo objetivo tA
    # particionando la muestra ara entrenamiento y verificacion
    # a fin de evaluar la calidad de la solucion
    shuffle(D)  
    
    test_size = int(round(len(D)*slice))
    test_sample, train_sample = D[:test_size], D[test_size:]
    print "[DATOS] Total =",len(D),", Test =",len(test_sample),", Train =",len(train_sample)
    
    delta_estimated, variance = cross_validation(train_sample,tA,parm)
    delta_real                = delta(train_sample,test_sample,tA,parm)
    
    return delta_estimated, variance, delta_real


#########################    PRINCIPAL    ##############################
tA = "G3"   

datasets = { 
    "MAT":"Dataset/student-mat.csv",
    "POR":"Dataset/student-por.csv"
}
# cases = {"TEST":"Dataset/student-test.csv"}

parms = {
#     "DT":{
#         "inst":DT,
#         "parm1":[5,10,None], # max depth
#         "parm1name":"DEPTH",
#         "KFold":10
#     },
    "NB":{
        "inst":NB,
        "parm1":[0,0.5,1], # equivalent sample data
        "parm1name":"M",
        "KFold":10
    },
#     "KNN":{
#         "inst":KNN,
#         "parm1":[1,2,3], # k neighbors
#         "parm1name":"K",
#         "KFold":None # Usa LOO CV
#     }
}

boolean_set = ["ORIG","BOOL"]
without_set = ["CON","SIN"]

remove('Summarize.txt') if exists('Summarize.txt') else None
for name,path in datasets.items():
    for boolean in boolean_set:
        for without in without_set:
            for case,parm in parms.items():
                _parm = []
                while parm['parm1']: # Prueba cada caso
                    
                    # Descirbir caso
                    tcase = "%s %s %s %s G1&G2 with %s" % (
                        name, 
                        case, 
                        boolean, 
                        without, 
                        "%s%s" % (parm["parm1name"],str(parm['parm1'][0]))
                    )
                    print "\n",tcase
                    
                    # Cargar el dataset
                    dataset = data(path,boolean=="BOOL",without=="SIN")
                    
                    try: # Procesar el caso, si ocurre un error se imprime
                        d_est, var, d_real = process(dataset,tA,parm)
                        res = "DELTA_ESTIMADO: %f\nVARIANZA: %5.3f\nDELTA_REAL: %f" % (d_est,var,d_real)
                    except Exception as error:
                        res = str(error)
                    print "\n",res
                    
                    # Imprimir resultado
                    with open('Summarize.txt','a+') as f:
                        f.write('\n'.join(["\n\n*** %s ***" % tcase , res]))
                           
                    _parm.append(parm['parm1'].pop(0))
                    
                parm['parm1'] = _parm        
              
