import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def estadisticos_continuos(dataframe):
    '''
    Devuelve los estadísticos de media, moda y desviación estándar asociados a 
    un atributo de tipo contínuo (tipos int64 o float64). Organiza la información
    en un DataFrame como tabla.
    
    Elementos:
        
    - dataframe: Dataframe a analizar
        
    '''
    columna, media, mediana, desviacion = [], [], [], []
    for col in dataframe:
        if (dataframe[col].dtypes == 'int64') or (dataframe[col].dtypes == 'float64'):
            columna.append(col)
            media.append(dataframe[col].mean())
            mediana.append(dataframe[col].median())
            desviacion.append(dataframe[col].std())
    return pd.DataFrame({'Media': media,
                  'Mediana': mediana,
                  'Desviación Estándar': desviacion}, index = columna)

def grafestad_discretos(dataframe, right_=3.5, hspace_=1.1):
    '''
    Devuelve gráficos de barras de las frecuencias de atributos de
    tipo discreto (tipo object).
    
    Elementos:
        
    - dataframe: Dataframe a analizar
        
    '''
    variable = []
    for col in dataframe:
        if dataframe[col].dtypes == 'object':
            variable.append(col)
    for i in range (len(variable)):
        plt.subplot(3,len(variable) ,i+1)
        sns.countplot(dataframe[variable[i]])
        plt.title(f'Frecuencias de {variable[i]}')
        plt.xlabel('')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=90)
    return plt.subplots_adjust(right=right_, hspace=hspace_)

def grafestad_continuos(dataframe):
    '''
    Devuelve gráficos de barras de las frecuencias de atributos de
    tipo continuo (tipos int64 y float64). Adicionalmente, genera
    dos líneas verticales que indican la media (rojo) y moda (azul)
    del atributo analizado.
    
    Elementos:
        
    - dataframe: Dataframe a analizar
        
    '''
    variable = []
    for col in dataframe:
        if (dataframe[col].dtypes == 'int64') or (dataframe[col].dtypes == 'float64'):
            variable.append(col)
    for i in range (len(variable)):
        plt.subplot(2,4,i+1)
        plt.hist(dataframe[variable[i]].dropna())
        plt.title(f'Frecuencias de {variable[i]}')
        plt.axvline(dataframe[variable[i]].mean(), color= 'tomato')
        plt.axvline(dataframe[variable[i]].median(), color= 'blue')
        plt.xlabel('')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=30)
    return plt.subplots_adjust(right=3.5, hspace=.5)

def conversor_perdidos(dataframe, valor):
    '''
    Convierte un valor determinado de dato perdido, convirtiéndolo en NaN
    
    Elementos:
    - dataframe: Dataframe a analizar.
    - valor: Valor que representa dato perdido y que se desea convertir.
    '''
    return dataframe.replace(valor, np.nan)

def cant_nan(dataframe):
    '''
    Devuelve la cantidad de NaN presentes en el dataframe, según columna.
    
    Elementos: 
    - dataframe: Dataframe a analizar.    
    '''
    nombre, att = [], []
    for col in dataframe:
        nombre.append(col)
        att.append(dataframe[col].isna().sum())
    return pd.DataFrame({'Variable': nombre,
                         'Cantidad de NaN': att})
    
def cant_perdidos_nombres(dataframe):
    '''
    Inspecciona en la prueba 2 la cantidad de valores perdidos llamados
    'sem validade', 'nulidade' y 'zero'.
    
    Elementos: 
    - dataframe: Dataframe a analizar.
    '''
    name, count_sem, count_nulidade, count_zero_ = [], [], [], []
    for col in dataframe:
        name.append(col)
        count_sem.append(sum(dataframe[col] == 'sem validade'))
        count_nulidade.append(sum(dataframe[col] == 'nulidade'))
        count_zero_.append(sum(dataframe[col] == 'zero'))
    return pd.DataFrame({'Columna': name,
                         'Cantidad sem validade': count_sem,
                         'Cantidad nulidade': count_nulidade,
                         'Cantidad zero': count_zero_})

def conversor_num(dataframe, col):
    '''
    Transforma los datos de tipo string compuestos por integers modificados con
    caracteres especiales, (por ejemplo, "17"). Devuelve el valor numérico del
    valor en forma de un integer (int64).
        
    Elementos
    - dataframe: Dataframe a analizar.
    - col: Columna a analizar.
    '''
    valor = []
    for i in range(len(dataframe[col])):
        if type(dataframe[col][i]) == str:
            letra_letra = [char for char in dataframe[col][i]]
            if len(letra_letra) == 4:
                valor.append(int(letra_letra[1]+letra_letra[2]))
            elif len(letra_letra) == 3:
                valor.append(int(letra_letra[1]))
        else:
            valor.append(dataframe[col][i])
    return valor

def conversor_int(dataframe, col):
    '''
    Transforma los datos de tipo string de una columna en valores de tipo
    integer (int64).
        
    Elementos
    - dataframe: Dataframe a analizar.
    - col: Columna a analizar.
    '''
    valor = []
    for i in range(len(dataframe[col])):
        if type(dataframe[col][i]) == str:
            valor.append(int(dataframe[col][i]))
        else:
            valor.append(dataframe[col][i])
    return valor

def binarizador(dataframe, col, valor, codigo, operador= 'igual'):
    '''
    Transforma los valores de una variable a formato binario.
    
    Elementos:
    - dataframe: Dataframe a analizar.
    - col: Columna a analizar.
    - valor: Valor de la variable a representar con mayor o menor frecuencia.
    - codigo: Representa el valor binario que asume la variable. En caso de
    asumir un determinado valor (0 o 1), la función automáticamente asigna
    el valor contrario a los otros valores.
    - operador: Representa la operación aritmética a evaluar en la función.
    Por defecto: 'igual'. Opciones: 'igual', 'mayorig' (Mayor o igual).
    '''
    if operador == 'igual':
        if codigo == 0:
            bin = np.where(dataframe[col] == valor, codigo, 1)
        elif codigo == 1:
            bin = np.where(dataframe[col] == valor, codigo, 0)
            
    elif operador == 'mayorig':
        if codigo == 0:
            bin = np.where(dataframe[col] >= valor, codigo, 1)
        elif codigo == 1:
            bin = np.where(dataframe[col] >= valor, codigo, 0)
    
    
    
    
    return bin

def regresores(dataframe, vector):
    '''
    Devuelve una lista ordenada con los coeficientes de correlación de Pearson
    asociado a cada atributo (regresores) para un vector objetivo.
    
    Elementos
    - dataframe: Dataframe a analizar.
    - vector: Vector objetivo.
    '''
    columns = dataframe.columns
    att_name, pearson_r, abs_pearson_r = [], [], []
    for col in columns:
        if col != vector:
            att_name.append(col)
            pearson_r.append(dataframe[col].corr(dataframe[vector]))
            abs_pearson_r.append(abs(dataframe[col].corr(dataframe[vector])))
    features = pd.DataFrame({'Atributo': att_name,
                             'Pearson': pearson_r,
                             'Abs Pearson': abs_pearson_r})
    features = features.set_index('Atributo')
    return features.sort_values(by=['Abs Pearson'], ascending= False)

def report_scores(predicho, dato):
    '''
    Devuelve los valores de MSE y R2 de un modelo lineal.
    '''
    print(f'MSE: {mean_squared_error(dato, predicho)}');
    print(f'Pearson: {r2_score(dato, predicho)}');

def estimador_binario(varlogit, variabilizador):
    '''
    Devuelve una lista ordenada con las probabilidades de cada regresor.
        
    Elementos
    - varlogit: Atributo analizador.
    - variabilizador: Objeto que contiene una lista con los nombres de los atributos implicados en la regresión del vector objetivo
    '''
    name, invlogit_0, invlogit_1 = [], [], []
    for i in range(len(variabilizador)):
        name.append(variabilizador[i])
        estimador_0 = varlogit.params['Intercept'] + (varlogit.params[variabilizador[i]] * 0)
        estimador_1 = varlogit.params['Intercept'] + (varlogit.params[variabilizador[i]] * 1)
        invlogit_0.append(1/(1 + np.exp(-estimador_0)))
        invlogit_1.append(1/(1 + np.exp(-estimador_1)))
    return pd.DataFrame({'Atributo': name,
                         'Estimación 0': invlogit_0,
                         'Estimación 1': invlogit_1})

def notas(dataframe, col):
    '''
    Devuelve gráficos de barra para la caracterización de vectores objetivos.
    
    Elementos:
    - dataframe: Dataframe a analizar.
    - col: Columna a analizar.
    '''
    plt.subplot(1,3,1)
    sns.countplot(dataframe[col[0]])
    plt.title('Cantidad de personas vs. Puntaje del primer semestre')
    plt.ylabel('Cantidad de personas')
    plt.xlabel('Puntaje')
    plt.legend()
    plt.subplot(1,3,2)
    sns.countplot(dataframe[col[1]])
    plt.title('Cantidad de personas vs. Puntaje del segundo semestre')
    plt.ylabel('Cantidad de personas')
    plt.xlabel('Puntaje')
    plt.legend()
    plt.subplot(1,3,3)
    sns.countplot(dataframe[col[2]])
    plt.title('Cantidad de personas vs. Puntaje anual')
    plt.ylabel('Cantidad de personas')
    plt.xlabel('Puntaje')
    plt.legend()
    return plt.subplots_adjust(right=3)