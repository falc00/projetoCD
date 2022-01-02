import os
datasets = ['dataset_1/profiling','dataset_2/profiling']
dir = datasets[0]
for dataset in datasets:    
    for f in os.listdir(dataset):
        if f != '.DS_Store':
            for file in os.listdir(dataset + '/' + f):
                os.remove(dataset + '/' + f + '/' + file)
        #os.remove(dataset + '/' + f + '/.DS_Store')
for f in os.listdir('data'):
    os.remove('data/'+ f)