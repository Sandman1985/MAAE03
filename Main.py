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
from time   import time, strftime

def data(csv_file,boolea):
    # Obtiene los ejemplos a partir de un archivo *.csv
    with open(csv_file, mode='r') as f:
        # Obtener datos crudos
        D = [row for row in DictReader(f,delimiter=";")]
        
        # Quitar G1 y G2 del dataset 
        D = [{k:v for k,v in d.items() if k!='G1' and k!='G2'} for d in D]
        
        # Obtener aquellos datos que sean numericos
        D = [{k:(int(v) if v.isdigit() else v) for k,v in d.items()} for d in D]

        # Preprocesamiento segun flags
        if boolean: # Transformar a booleano - Update no retorna valores, por eso no se asigna a nada
            [d.update({'absences':('Aceptable' if int(d['absences']) < 10 else 'Muchas')}) for d in D]
            [d.update({'G3':('Malo' if int(d['G3']) < 12 else 'Bueno')}) for d in D]
            [d.update({'age':('Menor' if int(d['age']) < 18 else 'Adulto')}) for d in D]
        
        # Normalizar valors en [0,1]
#         [d.update({'school':(0 if d['school']=="GP" else 1)}) for d in D]
#         [d.update({'sex'   :(0 if d['sex']=="F"  else 1)}) for d in D]
#         if boolean:
#             [d.update({'age':(0 if d['age']=="Menor" else 1)}) for d in D]
#             [d.update({'absences':(0 if d['absences']=="Aceptable" else 1)}) for d in D]
#             [d.update({'G3':(0 if d['age']=="Malo" else 1)}) for d in D]
#         else:
#             [d.update({'age':(1.0 * (int(d['age']) - 15) / 8)}) for d in D]
#             [d.update({'absences':(1.0 * (int(d['absences'])) / 94)}) for d in D]
#             [d.update({'G3':(1.0 * (int(d['G3'])) / 21)}) for d in D]
#         [d.update({'address'   :(0 if d['address']=="U" else 1)}) for d in D]
#         [d.update({'famsize'   :(0 if d['famsize']=="LE3" else 1)}) for d in D]
#         [d.update({'Pstatus'   :(0 if d['Pstatus']=="T" else 1)}) for d in D]
#         [d.update({'Medu'      :(1.0 * (int(d['Medu'])) / 5)}) for d in D]
#         [d.update({'Fedu'      :(1.0 * (int(d['Fedu'])) / 5)}) for d in D]
#         [d.update({'Mjob'      :(0 if d['Mjob']=="teacher" else 0.25 if d['Mjob']=="health" else 0.5 if d['Mjob']=="services" else 0.75 if d['Mjob']=="at_home" else 1.0)}) for d in D]
#         [d.update({'Fjob'      :(0 if d['Fjob']=="teacher" else 0.25 if d['Fjob']=="health" else 0.5 if d['Fjob']=="services" else 0.75 if d['Fjob']=="at_home" else 1.0)}) for d in D]
#         [d.update({'reason'    :(0 if d['reason']=="home" else 0.33 if d['reason']=="reputation" else 0.66 if d['reason']=="course" else 1.0)}) for d in D]
#         [d.update({'guardian'  :(0 if d['guardian']=="mother" else 0.5 if d['guardian']=="father" else 1.0)}) for d in D]
#         [d.update({'traveltime':(1.0 * (int(d['traveltime']) - 1) / 3)}) for d in D]
#         [d.update({'studytime' :(1.0 * (int(d['studytime']) - 1) / 3)}) for d in D]
#         [d.update({'failures'  :(1.0 * (int(d['failures'])) / 4)}) for d in D]
#         [d.update({'schoolsup' :(0 if d['schoolsup']=="yes" else 1)}) for d in D]
#         [d.update({'famsup'    :(0 if d['famsup']=="yes" else 1)}) for d in D]
#         [d.update({'paid'      :(0 if d['paid']=="yes" else 1)}) for d in D]
#         [d.update({'activities':(0 if d['activities']=="yes" else 1)}) for d in D]
#         [d.update({'nursery'   :(0 if d['nursery']=="yes" else 1)}) for d in D]
#         [d.update({'higher'    :(0 if d['higher']=="yes" else 1)}) for d in D]
#         [d.update({'internet'  :(0 if d['internet']=="yes" else 1)}) for d in D]
#         [d.update({'romantic'  :(0 if d['romantic']=="yes" else 1)}) for d in D]
#         [d.update({'famrel'    :(1.0 * (int(d['famrel']) - 1) / 4)}) for d in D]
#         [d.update({'freetime'  :(1.0 * (int(d['freetime']) - 1) / 4)}) for d in D]
#         [d.update({'goout'     :(1.0 * (int(d['goout']) - 1) / 4)}) for d in D]
#         [d.update({'Dalc'      :(1.0 * (int(d['Dalc']) - 1) / 4)}) for d in D]
#         [d.update({'Walc'      :(1.0 * (int(d['Walc']) - 1) / 4)}) for d in D]
#         [d.update({'health'    :(1.0 * (int(d['health']) - 1) / 4)}) for d in D]
        
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
        print "\r[CROSS-VALIDATION] Progress: %i/%i" % (cont,total)
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
    
    delta_estimated, variance = cross_validation(train_sample,tA,parm)
    delta_real                = delta(train_sample,test_sample,tA,parm)
    
    return round(delta_estimated,3), round(variance,3), round(delta_real,3), len(D), len(test_sample), len(train_sample) 


#########################    PRINCIPAL    ##############################
tA = "G3"   

# Filepath a los datasets
datasets = { 
    "MAT":"Dataset/student-mat.csv",
    "POR":"Dataset/student-por.csv"
}

# Configuracion parametrica
parms = {
     "DT":{
         "inst":DT,
         "parm1":[None,5], # max depth
         "parm1name":"DEPTH",
         "KFold":None
     },
     "NB":{
         "inst":NB,
         "parm1":[0,0.5,1], # equivalent sample data
         "parm1name":"M",
         "KFold":None
     },
    "KNN":{
        "inst":KNN,
        "parm1":[1,3], # k neighbors
        "parm1name":"K",
        "KFold":None # Usa LOO CV
    }
}

boolean_set = ["ORIG","BOOL"]

# Auxiliar
def format_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return '%02d:%02d:%02d' % (h, m, s)

# Comienzo principal
line = '%-18s %14s %11s %10s %10s %8s %8s %8s\n' % ("CASO","DELTA_ESTIMADO","DELTA_REAL","VARIANZA","TIEMPO","#DATOS","#TRAIN","#TEST")
with open('Summarize.txt','w') as f:
    f.write(line)
                
for name,path in datasets.items():
    for boolean in boolean_set:
        for case,parm in parms.items():
            _parm = []
            while parm['parm1']: # Prueba cada caso
                
                # Declarar caso caso
                tcase = "%s %s %s %s" % (name, case, boolean, "%s=%s" % (parm["parm1name"],str(parm['parm1'][0])))
                print "\n",tcase
                
                # Cargar el dataset
                dataset = data(path,boolean=="BOOL")
                
                t_start = time()
#                 try: # Procesar el caso, si ocurre un error se imprime (ocurre en NB cuando denominador 0 y m=0)
                dest, var, dreal, total, ctrain, ctest = process(dataset,tA,parm)
                res = "DELTA_ESTIMADO: %4.3f\nVARIANZA: %4.3f\nDELTA_REAL: %4.3f" % (dest,var,dreal)
#                 except Exception as error:
#                     res = str(error)
                t_elapsed = time() - t_start
                
                # Imprimir resultado
                print "\n",res
                line = '%-18s %14.3f %11.3f %10.3f %10s %8i %8i %8i\n' % (tcase,dest,dreal,var,format_time(t_elapsed),total,ctrain,ctest)
                with open('Summarize.txt','a+') as f:
                    f.write(line)
                       
                _parm.append(parm['parm1'].pop(0))
                
            parm['parm1'] = _parm        
              
