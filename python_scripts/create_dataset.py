#!/usr/bin/env python3
def create_dataset(lista_canais, save_path):
    '''
    Cria um objeto do tipo dicionário contendo os canais de interesse do 
    dataset, informados pelo usuário na forma de lista. 
    O dict contém as seguintes keys: 
    'data': array de shape (n,7680), onde n = len(lista_canais).
    'target': array contendo as classes correspondentes a cada frequência
    de estimulação.
    'fs': frequência de amostragem utilizada nos testes (256 Hz).
    
    Parâmetros de entrada: 
    lista_canais: lista de canais de interesse, disponíveis nos arquivos
    originais do dataset. 
    save_path: caminho onde o objeto será salvo.
    
    Retorna: 
    Objeto de tipo dict salvo no local especificado por save_path.
    '''
    import numpy as np 
    from scipy.io import loadmat, savemat

    # carregando os canais de interesse dos arquivos .mat e gerando o array
    n_canais = len(lista_canais)
    path = '/home/renato/Dropbox/Mestrado/final/kolodziej_dataset/all_data/S{}_{}Hz.mat'
    data = [np.transpose(loadmat(path.format(i,j))['X'])[lista_canais] 
                for j in range(5,9) for i in range(1,6)]
    data = np.vstack(data)

    # criando o array de targets 
    target = [np.full(n_canais*5, i, dtype='int64') for i in range(5,9)]
    target = np.hstack((target[0], target[1], target[2], target[3]))
    # salvando o dataset
    dataset_dict = {'data': data,
                   'target': target,
                   'fs': 256.}
    return savemat(save_path, dataset_dict)

    
