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
from time   import time

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

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
        "parm1":[5], # max depth
        "parm1name":"DEPTH",
        "KFold":10
    },
    "NB":{
        "inst":NB,
        "parm1":[0,0.5,1], # equivalent sample data
        "parm1name":"M",
        "KFold":10
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
    # Da formato HH:MM:SS a partir de segundos
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return '%02d:%02d:%04.2f' % (h, m, s)

def print_chart(x,y_dict,y_label,name):
    # Genera una grafica de barras
    plt.clf()
    plt.gca().set_xlim(-1,len(x))
    x_pos = xrange(len(x))
    i,colors=0,['#00007A','#0000F7']
    for label,y in y_dict.items():
        plt.bar(
            x_pos,
            y,
            align = 'center', 
            alpha = 0.4,
            label = label,
            color = 'Blue'
        )
        i+=1
    plt.xticks(x_pos, x, rotation=40, ha='right')
    plt.ylabel(y_label)
    plt.title('Result %s' % name)
    plt.savefig("%s.png" % name.replace(" ","_"))

# Comienzo principal
line = '%-20s %14s %11s %10s %10s %12s %8s %8s\n' % ("CASO","DELTA_ESTIMADO","DELTA_REAL","VARIANZA","TIEMPO","#DATOS","#TRAIN","#TEST")
with open('Summarize.txt','w') as f:
    f.write(line)
    
chart = {'case':[],'dreal':[],'dest':[],'var':[],'time':[]}            
for case,parm in parms.items():
    for boolean in boolean_set:
        for name,path in datasets.items():
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
                line = '%-20s %14.3f %11.3f %10.3f %10s %12i %8i %8i\n' % (tcase,dest,dreal,var,format_time(t_elapsed),total,ctrain,ctest)
                with open('Summarize.txt','a+') as f:
                    f.write(line)
                
                # Agregar datos para graficar
                chart['case'].append(tcase)
                chart['dreal'].append(dreal)
                chart['dest'].append(dest)
                chart['time'].append(t_elapsed)
                     
                _parm.append(parm['parm1'].pop(0))
                
            parm['parm1'] = _parm 
            
    # Graficar resultados    
    print_chart(chart['case'], {'Time':chart['time']} ,'SEGS','%s by time' % case)
    print_chart(chart['case'], {'Real':chart['dreal'],'Estimated':chart['dest']},'RATE','%s by error' % case)
    chart = {'case':[],'dreal':[],'dest':[],'var':[],'time':[]}
