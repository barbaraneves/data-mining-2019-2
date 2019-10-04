import numpy as np

def mae(y_real, y_predito):
    n = len(y_real)
    absolute = np.abs(y_real - y_predito)
    somatorio = np.sum(absolute)
    return somatorio * (1/n)

def mse(y_real, y_predito):
    n = len(y_real)
    quadrado = (y_real - y_predito)**2
    somatorio = np.sum(quadrado)
    return somatorio * (1/n)

def rmse(y_real, y_predito):
    calculo_mse = mse(y_real, y_predito)
    raiz_mse = np.sqrt(calculo_mse)
    return raiz_mse

def msle(y_real, y_predito):
    n = len(y_real)
    logaritmo = np.log(y_real + 1) - np.log(y_predito + 1)
    quadrado = logaritmo ** 2
    somatorio = np.sum(quadrado)
    return somatorio * (1/n)

def rmsle(y_real, y_predito):
    calculo_msle = msle(y_real, y_predito)
    raiz_msle = np.sqrt(calculo_msle)
    return raiz_msle

def ss_res(y_real, y_predito):
    quadrado = (y_real - y_predito)**2
    somatorio = np.sum(quadrado)
    return somatorio

def ss_tot(y_real):
    mean_y = np.mean(y_real)
    quadrado = (y_real - mean_y)**2
    somatorio = np.sum(quadrado)
    return somatorio

def r2(y_real, y_predito):
    calculo_ssres = ss_res(y_real, y_predito)
    calculo_sstot = ss_tot(y_real)
    divisao = calculo_ssres / calculo_sstot
    return 1 - divisao