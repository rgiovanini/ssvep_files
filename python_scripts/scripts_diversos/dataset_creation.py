# carregando os vetores e colocando ja na forma de duas dimensões 

# ARRUMA ISSO PELO AMOR DE DEUS!!!!!!!!!! VERGONHOSO!

# carregando 5hz
data_temp = np.array([loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/all_data/S{}_5Hz.mat'.format(i)) 
                      for i in range(1,6)])
data5 = np.array(np.vstack(([np.transpose(data_temp[i]['X']) for i in range(5)])))
data5[31] == np.transpose(data_temp[1]['X'])[15]

# criando o label 
label5 = np.full(data5.shape[0],5,dtype='int64')

#carregando 6 hz
data_temp = np.array([loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/all_data/S{}_6Hz.mat'.format(i))
                     for i in range(1,6)])
data6 = np.array(np.vstack(([np.transpose(data_temp[i]['X']) for i in range(5)])))
data_total = np.vstack((data5, data6))
label6 = np.full(data6.shape[0],6,dtype='int64')
label = np.hstack((np.transpose(label5), np.transpose(label6)))

#carregando 7 hz
data_temp = np.array([loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/all_data/S{}_7Hz.mat'.format(i))
                     for i in range(1,6)])
data7 = np.array(np.vstack(([np.transpose(data_temp[i]['X']) for i in range(5)])))
data_total = np.vstack((data_total, data7))
label7 = np.full(data7.shape[0],7,dtype='int64')
label = np.hstack((np.transpose(label), np.transpose(label7)))

#carregando 8 hz
data_temp = np.array([loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/all_data/S{}_8Hz.mat'.format(i))
                     for i in range(1,6)])
data8 = np.array(np.vstack(([np.transpose(data_temp[i]['X']) for i in range(5)])))
data_total = np.vstack((data_total, data8))
label8 = np.full(data8.shape[0],8,dtype='int64')
label = np.hstack((np.transpose(label), np.transpose(label8)))

# criando o dict e salvando todo o dataset como um .mat

dataset_kolodziej = {'data': data_total,
                     'label': label,
                     'fs': 256.,
                     'path': '/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/'}

from scipy.io import savemat
savemat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/dataset_kolodziej', dataset_kolodziej)


