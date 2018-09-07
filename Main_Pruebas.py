import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from TensorflowEchoStateNetwork.esn_ip_cell import ESNIPCell
import os
import pandas as pd
from functions import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.cluster import DBSCAN
#from sklearn import metrics

# Algunos Hyperparametros

#units = 10
washout_size = 0
scale = 1
#connectivity = 0.3
N = 3  # Number of labels / classes
time_steps=100
max_length = 60000
initial_crop = 2000
#maxSamples = int(((max_length - initial_crop)/2)/time_steps) #Se divide para 2 xq se toman mitad y mitad para train y para test
maxSamples = 250
increment=10
channels = 1 # Numero de canales o sensores o mediciones de la serie de tiempo y que se encuentra en el .csv (# de columnas)
maxTrainingSamples = maxSamples # numero maximo de muestras (porciones de senal) que se usaran para la optimizacion de IP


# Parametros IP

mean=0.
#std=0.6
learning_rate=0.001

tag = np.reshape([np.zeros(maxSamples), np.ones(maxSamples), np.full(maxSamples,2)], -1)  # Tag each point with a corresponding label. HACER ESTO DINAMICO!!!! QUE CREZCA EN FUNCION DE LOS DATOS


# Parametros para el analisis de silueta

range_n_clusters = [3, 4, 5, 6, 7]

# 0,40,96,235,249,255,57,76,77,132,187,205,26,41,94,133,154,166,

# Carga de datos, Adecuacion de los datos de train (Optimizacion IP) y prueba (para el reservorio)
#rules=[96, 77, 83, 133, 124, 86]
rules=[23,33,51,83,123,178,
        124,54,147,110,137,193,18,22,86,105,122,150] # numero de reglas a utilizar, 6 de cada clase
filename='rule_'
filepath='/home/jean/Dropbox/JeanCarlo_GIDTEC y Varios_Respaldos_Personal/TFM/Datos CA/'
savePath='/home/jean/Dropbox/JeanCarlo_GIDTEC y Varios_Respaldos_Personal/TFM/Experiments/'


cluster = []
testData = []
trainIPdata=[]
for rule in rules:
    df = pd.read_csv(filepath + filename + str(rule) + '_plots_single.csv', header=None, skiprows=20, nrows=max_length)
    values = df.values  # Sacamos del dataframe
    signals = np.float32(values[initial_crop:max_length, 1:channels+1]) # Siempre se le suma +1 a los canales para tomar los datos.  initial_crop es la cantidad de ticks iniciales que se cortan
    trainSamples=get_more_overlaped_samples(signals,time_steps,increment,maxTrainingSamples) # Devuelve la muestra de (maxSamples,time_steps,channels)
    trainIPdata.extend(trainSamples.reshape(maxSamples*time_steps,channels)) # Aplanamos en una sola serie larga la muestra actual (maxSamples*time_steps,channels), es como si pusieramos las porciones anteriores una a continuacion de otra (esto para todas las clases para IP)
    testSamples = split_in_subsamples(signals,time_steps,maxSamples)
    testData.append(testSamples)
    cluster.append(np.transpose(np.float32(signals)))


testData = np.array(testData).reshape([len(rules)*maxSamples,time_steps,channels])/257 # Devuelve arreglo de shape (maxSamples*#reglas,time_steps,channels) y "Normalizado" a 257. Si por ejemplo maxSamples=40 y tenemos 3 reglas, la primera dimension (del batch size) seria 120. Se puede poner -1 tambien en la primera dimension
trainIPdata = np.array(trainIPdata)/257 # Normalizamos los datos que se van a usar en la optimizacion IP

print('Datos_cargados')

#plt.plot(trainIPdata)
#plt.ylabel('some numbers')
#plt.show()

# Grid Search para el tamano del Reservorio



#unidades=np.arange(2,102,2)  # De 2 en 2 hasta llegar a 100


unidades = [6]
desviaciones = [1.0]
conexiones = [1.0]
for units in unidades:

    for std in desviaciones:
        for connectivity in conexiones:
            evolutionSR_NoIP = []
            evolutionSR_WithIP = []
            noIP_2clusters_count = []
            noIP_3clusters_count = []
            noIP_4clusters_count = []
            noIP_5clusters_count = []
            noIP_6clusters_count = []
            noIP_7clusters_count = []
            noIP_8clusters_count = []
            withIP_2clusters_count = []
            withIP_3clusters_count = []
            withIP_4clusters_count = []
            withIP_5clusters_count = []
            withIP_6clusters_count = []
            withIP_7clusters_count = []
            withIP_8clusters_count = []
            Silouethe_scores_means_IP=[]
            Silouethe_scores_means_noIP=[]
            noIP_samples_4clusters_count = []
            noIP_samples_6clusters_count = []
            withIP_samples_4clusters_count = []
            withIP_samples_6clusters_count = []



            for z in range(10): # Numero de repeticiones
                print('Estamos con '+ str(units)+' unidades')
                print('Ronda: ' + str(z))
                experimentName = str(units) + '_' + str(std) + '_'+str(connectivity)
                tf.reset_default_graph() # Para limpiar todo el grafo en cada iteracion


                with tf.Session() as S:


                    # Ejecucion sin optimizar IP
                    print("Building graph...")
                    # Instancia clase ESN
                    data_t = tf.constant(testData, dtype=tf.float32)  # testData son nuestros datos a utilizar. Debe ir aqui!!, no antes
                    esn = ESNIPCell(units, scale, connectivity, mean=mean, std=std, learning_rate=learning_rate, input_size=trainIPdata.shape[1])
                    outputs, final_state = tf.nn.dynamic_rnn(esn, data_t, dtype=tf.float32)
                    washed = tf.squeeze(tf.slice(outputs, [0, washout_size, 0], [-1, -1, -1]))

                    S.run(tf.global_variables_initializer())
                    print("Computing embeddings...")

                    res = S.run(washed)

                    var= [v for v in tf.global_variables() if v.name == 'rnn/ESNIPCell/GainVector:0'] # Ver el contenido de las variables
                    print(S.run(var))

                    # Histogramas para ver la estructura de los datos (las distribuciones)

                    if units>=10:
                        aux=10 # Lo dejamos clavado en 10 para unicamente ver las distribuciones de las 10 primeras neuronas. Esto para resevorios con mas de 10 unidades
                    if units<10:
                        aux=units


                    f, a = plt.subplots(aux, 1,figsize=(12, 22))
                    plt.subplots_adjust(bottom=.08, top=.92, left=.1, right=.95, hspace=0.8, wspace=0.35)
                    a = a.ravel()
                    for idx, ax in enumerate(a):
                        ax.hist(res[:,-1,idx],200,range=(-1, 1))
                        ax.set_title('Neurona del Reservorio: '+str(idx))
                        #ax.set_xlabel('Intervalo [-1,1]')
                        #ax.set_ylabel('Y')
                    #plt.tight_layout()
                    #f.suptitle('Distribuciones sin Plasticidad Intrínseca', fontsize=30)  # Creo q es mejor poner en el titulo de la imagen en el latex
                    #plt.show()
                    if not os.path.exists(savePath+experimentName):
                        os.makedirs(savePath+experimentName)
                    f.savefig(savePath+experimentName+'/fig_No_IP'+'_round_'+str(z)+'.pdf')
                    plt.close('all')

                    noIP_Silhouette_BestScores = []
                    noIP_bestScores_clusters = []
                    noIP_pairs = []
                    noIP_3clusters_scores = []
                    noIP_Silhouette_score=[]
                    noIP_samples_count = []
                    noIP_4clusters_samples_count= []
                    noIP_6clusters_samples_count = []

                    # Graficos de pares de neuronas
                    aux_pares=0
                    for i in range(units-1):
                        for j in range(i+1,units):
                            print('Par de neuronas: '+ str(i)+'_'+str(j))
                            aux_pares += 1

                            #plt.figure()
                            #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                            # define the data
                            x = res[:,-1, i]
                            y = res[:,-1, j]

                            points = np.transpose(np.vstack((x, y)))
                            # Obtencion del numero estimado de clusters

                            scores=[]
                            for n_clusters in range_n_clusters:
                                # Initialize the clusterer with n_clusters value and a random generator
                                # seed of 10 for reproducibility.
                                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                                cluster_labels = clusterer.fit_predict(points)

                                # The silhouette_score gives the average value for all the samples.
                                # This gives a perspective into the density and separation of the formed
                                # clusters
                                silhouette_avg = silhouette_score(points, cluster_labels)
                                scores.append([n_clusters, silhouette_avg])



                            max_indx=np.argmax(scores,axis=0)#Obtenemos los indices del mayor silhouette score
                            bestScore=scores[max_indx[1]] # obtenemos el par [n_clusters,best_score]
                            n_clusters=bestScore[0] # El nuevo n_clusters  # Aqui vendria la condicion if n_clusters == 3, plotearse
                            noIP_Silhouette_BestScores.append(bestScore[1])
                            noIP_bestScores_clusters.append(bestScore[0])
                            noIP_pairs.append('Par_' + str(i) + '_' + str(j))

                            if n_clusters == 3: #or n_clusters == 4:

                                noIP_Silhouette_score.append(bestScore[1])
                                noIP_3clusters_scores.append(['Par_' + str(i) + '_' + str(j),bestScore,n_clusters])

                                # Create a subplot with 1 row and 2 columns
                                fig, (ax1, ax2) = plt.subplots(1, 2)
                                fig.set_size_inches(18, 7)

                                # The 1st subplot is the silhouette plot
                                # The silhouette coefficient can range from -1, 1 but in this example all
                                # lie within [-0.1, 1]
                                ax1.set_xlim([-0.1, 1])
                                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                                # plots of individual clusters, to demarcate them clearly.
                                ax1.set_ylim([0, len(points) + (n_clusters + 1) * 10])

                                # Volvemos a hacer la clusterizacion
                                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                                cluster_labels = clusterer.fit_predict(points)
                                silhouette_avg = silhouette_score(points, cluster_labels)

                                # Contamos la cantidad de muestras por clase (o cluster)
                                unique_elements, counts_elements = np.unique(np.array(cluster_labels),
                                                                             return_counts=True)
                                noIP_samples_count.append([unique_elements, counts_elements,'Par_' + str(i) + '_' + str(j)])

                                if max(unique_elements) == 5:  #Esto para cuando clusteriza 6 grupos
                                    noIP_6clusters_samples_count.append(np.sort(counts_elements))

                                if max(unique_elements) == 3:  #Esto para cuando clusteriza 4 grupos
                                    noIP_4clusters_samples_count.append(np.sort(counts_elements))


                                # Compute the silhouette scores for each sample
                                sample_silhouette_values = silhouette_samples(points, cluster_labels)

                                y_lower = 10
                                for p in range(n_clusters):
                                    # Aggregate the silhouette scores for samples belonging to
                                    # cluster p, and sort them
                                    ith_cluster_silhouette_values = \
                                        sample_silhouette_values[cluster_labels == p]

                                    ith_cluster_silhouette_values.sort()

                                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                                    y_upper = y_lower + size_cluster_i

                                    color = cm.nipy_spectral(float(p) / n_clusters)
                                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                                      0, ith_cluster_silhouette_values,
                                                      facecolor=color, edgecolor=color, alpha=0.7)

                                    # Label the silhouette plots with their cluster numbers at the middle
                                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(p))

                                    # Compute the new y_lower for next plot
                                    y_lower = y_upper + 10  # 10 for the 0 samples

                                ax1.set_title("El diagrama de silueta para los diversos clusters.")
                                ax1.set_xlabel("Valores de coeficientes de silueta")
                                ax1.set_ylabel("Etiqueta de cluster")

                                # The vertical line for average silhouette score of all the values
                                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                                # 2nd Plot showing the actual clusters formed
                                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                                ax2.scatter(points[:, 0], points[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                            c=colors, edgecolor='k')

                                # Labeling the clusters
                                centers = clusterer.cluster_centers_
                                # Draw white circles at cluster centers
                                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                            c="white", alpha=1, s=200, edgecolor='k')

                                for p, c in enumerate(centers):
                                    ax2.scatter(c[0], c[1], marker='$%d$' % p, alpha=1,
                                                s=50, edgecolor='k')

                                ax2.set_title("Visualización de los datos agrupados.")
                                ax2.set_xlabel("Espacio de características para neurona "+str(i))
                                ax2.set_ylabel("Espacio de características para neurona "+str(j))

                                plt.suptitle(("Análisis de silueta mediante K Means sin Plasticidad Intrínseca "
                                              " con n_clusters = %d" % n_clusters,
                                              " y coeficiente = %.8f" % silhouette_avg),
                                             fontsize=14, fontweight='bold')
                                plt.savefig(
                                    savePath + experimentName + '/fig_neurons_No_IP_' + str(i) + '_' + str(j) + '_round_' + str(
                                        z) + '.pdf')

                                plt.close('all')
                    print('pares analizados: '+str(aux_pares))

                    #Ploteo Radio Spectral

                    sr = esn.getSpectralRadius()
                    Radius1 = S.run(sr)
                    print('El radio spectral es: ' + str(Radius1))


                    #Verificacion del Reservorio

                    wr=esn.getReservoirWeights()
                    reservoirState= S.run(wr)

                    # Optimizacion del reservorio mediante IP


                    for e in range(3): # Un maximo de 3 epocas
                        print('Optimization:',e)
                        indx = np.random.choice(trainIPdata.shape[0], len(trainIPdata), replace=False)  # Mezclar los puntos de datos
                        y_ip,gain,bias=esn.optimizeIPscan(tf.constant(trainIPdata[indx,:],dtype=tf.float32))
                        h_prev,GainRun,BiasRun = S.run([y_ip, gain, bias])
                        print(GainRun)


                        # Ejecucion con IP
                        print("Ejecucion con IP...")
                        # outputs, final_state = tf.nn.dynamic_rnn(esn, data_t, dtype=tf.float32)
                        # washed = tf.squeeze(tf.slice(outputs, [0, washout_size, 0], [-1, -1, -1]))

                        print("Computing embeddings...")

                        res = S.run(washed)

                        # Histogramas para ver la estructura de los datos (las distribuciones)

                        if units >= 10:
                            aux = 10  # Lo dejamos clavado en 10 para unicamente ver las distribuciones de las 10 primeras neuronas. Esto para resevorios con mas de 10 unidades
                        if units < 10:
                            aux = units

                        f, a = plt.subplots(aux, 1, figsize=(12, 22))  # Lo dejamos clavado en 10 para unicamente ver las distribuciones de las 10 primeras neuronas
                        plt.subplots_adjust(bottom=.08, top=.92, left=.1, right=.95, hspace=0.8, wspace=0.35)
                        a = a.ravel()
                        for idx, ax in enumerate(a):
                            ax.hist(res[:, -1, idx], 200, range=(-1, 1)) # Lo que hay en res son las activaciones del reservorio
                            ax.set_title('Neurona del Reservorio: ' + str(idx))
                            # ax.set_xlabel('Intervalo [-1,1]')
                            # ax.set_ylabel('Y')
                        # plt.tight_layout()
                        # f.suptitle('Distribuciones sin Plasticidad Intrínseca', fontsize=30)  # Creo q es mejor poner en el titulo de la imagen en el latex
                        #plt.show()
                        f.savefig(savePath+experimentName+'/fig_with_IP_optim_'+str(e)+'_round_'+str(z)+'.pdf')
                        plt.close('all')

                        # Graficos de pares de neuronas (Ahora con IP)

                    # Ploteo radio spectral despues de la Optimizacion

                    sr = esn.getSpectralRadius()
                    Radius2 = S.run(sr)
                    print('El radio spectral es despues de IP: ' + str(Radius2))

                    withIP_Silhouette_BestScores = []
                    withIP_bestScores_clusters = []
                    withIP_pairs = []
                    withIP_3cluster_scores = []
                    withIP_Silhouette_score=[]
                    withIP_samples_count = []
                    withIP_4clusters_samples_count = []
                    withIP_6clusters_samples_count = []

                    aux_pares = 0

                    # Clusters en los pares de neuronas
                    for i in range(units-1):
                        for j in range(i+1,units):
                            print('Par de neuronas: ' + str(i) + '_' + str(j))
                            aux_pares += 1
                            #plt.figure()
                            #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                            # define the data
                            x = res[:, -1, i]  # -1 indica el ultimo paso de tiempo
                            y = res[:, -1, j]

                            points = np.transpose(np.vstack((x, y)))
                            # Obtencion del numero estimado de clusters


                            scores = []
                            for n_clusters in range_n_clusters:
                                # Initialize the clusterer with n_clusters value and a random generator
                                # seed of 10 for reproducibility.
                                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                                cluster_labels = clusterer.fit_predict(points)

                                # The silhouette_score gives the average value for all the samples.
                                # This gives a perspective into the density and separation of the formed
                                # clusters
                                silhouette_avg = silhouette_score(points, cluster_labels)
                                scores.append([n_clusters, silhouette_avg])


                            max_indx = np.argmax(scores, axis=0)  # Obtenemos los indices del mayor silhouette score
                            bestScore = scores[max_indx[1]]  # obtenemos el par [n_clusters,best_score]
                            n_clusters = bestScore[0]  # El nuevo n_clusters  # Aqui vendria la condicion if n_clusters == 3, plotearse
                            withIP_Silhouette_BestScores.append(bestScore[1])
                            withIP_bestScores_clusters.append(bestScore[0])
                            withIP_pairs.append('Par_' + str(i) + '_' + str(j))


                            if n_clusters == 3: #or n_clusters == 4:

                                withIP_Silhouette_score.append(bestScore[1])
                                withIP_3cluster_scores.append(['Par_' + str(i) + '_' + str(j), bestScore,n_clusters])

                                # Create a subplot with 1 row and 2 columns
                                fig, (ax1, ax2) = plt.subplots(1, 2)
                                fig.set_size_inches(18, 7)

                                # The 1st subplot is the silhouette plot
                                # The silhouette coefficient can range from -1, 1 but in this example all
                                # lie within [-0.1, 1]
                                ax1.set_xlim([-0.1, 1])
                                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                                # plots of individual clusters, to demarcate them clearly.
                                ax1.set_ylim([0, len(points) + (n_clusters + 1) * 10])

                                # Volvemos a hacer la clusterizacion
                                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                                cluster_labels = clusterer.fit_predict(points)
                                silhouette_avg = silhouette_score(points, cluster_labels)

                                # Compute the silhouette scores for each sample
                                sample_silhouette_values = silhouette_samples(points, cluster_labels)

                                # Contamos la cantidad de muestras por clase (o cluster)
                                unique_elements, counts_elements = np.unique(np.array(cluster_labels),
                                                                             return_counts=True)
                                withIP_samples_count.append([unique_elements, counts_elements,'Par_' + str(i) + '_' + str(j)])

                                if max(unique_elements) == 5:  # Esto para cuando clusteriza 6 grupos
                                    withIP_6clusters_samples_count.append(np.sort(counts_elements))

                                if max(unique_elements) == 3:  # Esto para cuando clusteriza 4 grupos
                                    withIP_4clusters_samples_count.append(np.sort(counts_elements))

                                y_lower = 10
                                for p in range(n_clusters):
                                    # Aggregate the silhouette scores for samples belonging to
                                    # cluster p, and sort them
                                    ith_cluster_silhouette_values = \
                                        sample_silhouette_values[cluster_labels == p]

                                    ith_cluster_silhouette_values.sort()

                                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                                    y_upper = y_lower + size_cluster_i

                                    color = cm.nipy_spectral(float(p) / n_clusters)
                                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                                      0, ith_cluster_silhouette_values,
                                                      facecolor=color, edgecolor=color, alpha=0.7)

                                    # Label the silhouette plots with their cluster numbers at the middle
                                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(p))

                                    # Compute the new y_lower for next plot
                                    y_lower = y_upper + 10  # 10 for the 0 samples

                                ax1.set_title("El diagrama de silueta para los diversos clusters.")
                                ax1.set_xlabel("Valores de coeficientes de silueta")
                                ax1.set_ylabel("Etiqueta de cluster")

                                # The vertical line for average silhouette score of all the values
                                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                                # 2nd Plot showing the actual clusters formed
                                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                                ax2.scatter(points[:, 0], points[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                            c=colors, edgecolor='k')

                                # Labeling the clusters
                                centers = clusterer.cluster_centers_
                                # Draw white circles at cluster centers
                                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                            c="white", alpha=1, s=200, edgecolor='k')

                                for p, c in enumerate(centers):
                                    ax2.scatter(c[0], c[1], marker='$%d$' % p, alpha=1,
                                                s=50, edgecolor='k')

                                ax2.set_title("Visualización de los datos agrupados.")
                                ax2.set_xlabel("Espacio de características para neurona "+str(i))
                                ax2.set_ylabel("Espacio de características para neurona "+str(j))

                                plt.suptitle(("Análisis de silueta mediante K Means con Plasticidad Intrínseca "
                                              " con n_clusters = %d" % n_clusters,
                                              " y coeficiente = %.8f" % silhouette_avg),
                                             fontsize=14, fontweight='bold')
                                plt.savefig(savePath + experimentName + '/fig_neurons_With_IP_' + str(i)+'_' + str(j) +'_round_'+str(z)+ '.pdf')
                                #plt.show()
                                plt.close('all')
                    print('pares analizados: ' + str(aux_pares))


                    # Obtencion de medias de muestras por clase y almacenamiento para el caso sin IP y con IP

                    # Sin IP
                    if len(noIP_4clusters_samples_count) >= 1:
                        meanclass0 = np.mean(np.array(noIP_4clusters_samples_count)[:, 0])
                        meanclass1 = np.mean(np.array(noIP_4clusters_samples_count)[:, 1])
                        meanclass2 = np.mean(np.array(noIP_4clusters_samples_count)[:, 2])
                        meanclass3 = np.mean(np.array(noIP_4clusters_samples_count)[:, 3])

                        noIP_4clusters_meansamples_count = [meanclass0,meanclass1,meanclass2,meanclass3]
                    else:noIP_4clusters_meansamples_count =['Ningun par obtuvo 4 grupos']


                    if len(noIP_6clusters_samples_count) >= 1:
                        meanclass0 = np.mean(np.array(noIP_6clusters_samples_count)[:, 0])
                        meanclass1 = np.mean(np.array(noIP_6clusters_samples_count)[:, 1])
                        meanclass2 = np.mean(np.array(noIP_6clusters_samples_count)[:, 2])
                        meanclass3 = np.mean(np.array(noIP_6clusters_samples_count)[:, 3])
                        meanclass4 = np.mean(np.array(noIP_6clusters_samples_count)[:, 4])
                        meanclass5 = np.mean(np.array(noIP_6clusters_samples_count)[:, 5])

                        noIP_6clusters_meansamples_count = [meanclass0, meanclass1, meanclass2, meanclass3, meanclass4, meanclass5]
                    else:noIP_6clusters_meansamples_count =['Ningun par obtuvo 6 grupos']

                    # Con IP
                    if len(withIP_4clusters_samples_count) >= 1:
                        meanclass0 = np.mean(np.array(withIP_4clusters_samples_count)[:, 0])
                        meanclass1 = np.mean(np.array(withIP_4clusters_samples_count)[:, 1])
                        meanclass2 = np.mean(np.array(withIP_4clusters_samples_count)[:, 2])
                        meanclass3 = np.mean(np.array(withIP_4clusters_samples_count)[:, 3])

                        withIP_4clusters_meansamples_count = [meanclass0, meanclass1, meanclass2, meanclass3]
                    else:withIP_4clusters_meansamples_count = ['Ningun par obtuvo 4 grupos']

                    if len(withIP_6clusters_samples_count) >= 1:
                        meanclass0 = np.mean(np.array(withIP_6clusters_samples_count)[:, 0])
                        meanclass1 = np.mean(np.array(withIP_6clusters_samples_count)[:, 1])
                        meanclass2 = np.mean(np.array(withIP_6clusters_samples_count)[:, 2])
                        meanclass3 = np.mean(np.array(withIP_6clusters_samples_count)[:, 3])
                        meanclass4 = np.mean(np.array(withIP_6clusters_samples_count)[:, 4])
                        meanclass5 = np.mean(np.array(withIP_6clusters_samples_count)[:, 5])

                        withIP_6clusters_meansamples_count = [meanclass0, meanclass1, meanclass2, meanclass3, meanclass4,
                                                        meanclass5]
                    else:withIP_6clusters_meansamples_count = ['Ningun par obtuvo 6 grupos']


                    # Creamos un diccionario de los parametros para Imprimir en un .txt algunos de los parametros

                    params = {'washout_size':washout_size,'scale':scale,'connectivity':connectivity,'Number_of_labels':N
                              ,'time_steps':time_steps,'max_length':max_length,'initial_crop':initial_crop,'maxSamples':maxSamples,'maxTrainingSamples':maxTrainingSamples
                              ,'increment':increment,'channels':channels,'mean':mean,'std':std,'learning_rate':learning_rate
                              ,'Architecture':str(experimentName),'num_epochs':3,'batch_size':maxSamples*N,'Spectral_Radius_No_IP':Radius1
                              ,'Spectral_Radius_after_IP':Radius2,'Gain_Vector':str(GainRun),'Bias_Vector':str(BiasRun), 'Last_state':str(h_prev)
                              ,'Rules':str(rules),'reservoirState':str(reservoirState),'withIP_Silhouette_BestScores':str(withIP_Silhouette_BestScores),'withIP_bestScores_clusters':str(withIP_bestScores_clusters)
                              ,'withIP_pairs':str(withIP_pairs),'withIP_3cluster_scores':str(withIP_3cluster_scores),'noIP_Silhouette_BestScores':str(noIP_Silhouette_BestScores),'noIP_bestScores_clusters':str(noIP_bestScores_clusters)
                              ,'noIP_pairs':str(noIP_pairs),'noIP_3clusters_scores':str(noIP_3clusters_scores)
                              ,'noIP_2clusters_count':noIP_bestScores_clusters.count(2),'noIP_3clusters_count':noIP_bestScores_clusters.count(3),'noIP_4clusters_count':noIP_bestScores_clusters.count(4),'noIP_5clusters_count':noIP_bestScores_clusters.count(5),'noIP_6clusters_count':noIP_bestScores_clusters.count(6),'noIP_7clusters_count':noIP_bestScores_clusters.count(7),'noIP_8clusters_count':noIP_bestScores_clusters.count(8)
                              ,'withIP_2clusters_count':withIP_bestScores_clusters.count(2),'withIP_3clusters_count':withIP_bestScores_clusters.count(3),'withIP_4clusters_count':withIP_bestScores_clusters.count(4), 'withIP_5clusters_count':withIP_bestScores_clusters.count(5), 'withIP_6clusters_count':withIP_bestScores_clusters.count(6), 'withIP_7clusters_count':withIP_bestScores_clusters.count(7), 'withIP_8clusters_count':withIP_bestScores_clusters.count(8)
                              ,'mean_silhouette_score_withIP':str(np.mean(withIP_Silhouette_score)),'mean_silhouette_score_noIP':str(np.mean(noIP_Silhouette_score)),'silhouette_score_withIP':str(withIP_Silhouette_score),'silhouette_score_noIP':str(noIP_Silhouette_score),'withIP_samples_count_ofclusters':str(withIP_samples_count),'noIP_samples_count_ofclusters':str(noIP_samples_count)
                              ,'noIP_4clusters_meansamples_count':str(noIP_4clusters_meansamples_count),'noIP_6clusters_meansamples_count':str(noIP_6clusters_meansamples_count),'withIP_4clusters_meansamples_count':str(withIP_4clusters_meansamples_count),'withIP_6clusters_meansamples_count':str(withIP_6clusters_meansamples_count)}

                    writeDictinTXTfile(savePath, experimentName, params,z)

                    print('Datos_Parametros_Guardados')

                    # Almacenamiento de metricas para el resumen

                    evolutionSR_NoIP.append(Radius1)
                    evolutionSR_WithIP.append(Radius2)
                    noIP_2clusters_count.append(noIP_bestScores_clusters.count(2))
                    noIP_3clusters_count.append(noIP_bestScores_clusters.count(3))
                    noIP_4clusters_count.append(noIP_bestScores_clusters.count(4))
                    noIP_5clusters_count.append(noIP_bestScores_clusters.count(5))
                    noIP_6clusters_count.append(noIP_bestScores_clusters.count(6))
                    noIP_7clusters_count.append(noIP_bestScores_clusters.count(7))
                    noIP_8clusters_count.append(noIP_bestScores_clusters.count(8))
                    withIP_2clusters_count.append(withIP_bestScores_clusters.count(2))
                    withIP_3clusters_count.append(withIP_bestScores_clusters.count(3))
                    withIP_4clusters_count.append(withIP_bestScores_clusters.count(4))
                    withIP_5clusters_count.append(withIP_bestScores_clusters.count(5))
                    withIP_6clusters_count.append(withIP_bestScores_clusters.count(6))
                    withIP_7clusters_count.append(withIP_bestScores_clusters.count(7))
                    withIP_8clusters_count.append(withIP_bestScores_clusters.count(8))
                    Silouethe_scores_means_IP.append(str(np.mean(withIP_Silhouette_score)))
                    Silouethe_scores_means_noIP.append(str(np.mean(noIP_Silhouette_score)))
                    noIP_samples_4clusters_count.append(noIP_4clusters_meansamples_count)
                    noIP_samples_6clusters_count.append(noIP_6clusters_meansamples_count)
                    withIP_samples_4clusters_count.append(withIP_4clusters_meansamples_count)
                    withIP_samples_6clusters_count.append(withIP_6clusters_meansamples_count)


            #Conteo de muestras por grupo promedio de las 10 rondas

            # #No Ip
            # meanclass0 = np.mean(np.array(noIP_samples_4clusters_count)[:, 0])
            # meanclass1 = np.mean(np.array(noIP_samples_4clusters_count)[:, 1])
            # meanclass2 = np.mean(np.array(noIP_samples_4clusters_count)[:, 2])
            # meanclass3 = np.mean(np.array(noIP_samples_4clusters_count)[:, 3])
            #
            # noIP_4clusters_mean10rounds_samplescount = [meanclass0, meanclass1, meanclass2, meanclass3]
            #
            # meanclass0 = np.mean(np.array(noIP_samples_6clusters_count)[:, 0])
            # meanclass1 = np.mean(np.array(noIP_samples_6clusters_count)[:, 1])
            # meanclass2 = np.mean(np.array(noIP_samples_6clusters_count)[:, 2])
            # meanclass3 = np.mean(np.array(noIP_samples_6clusters_count)[:, 3])
            # meanclass4 = np.mean(np.array(noIP_samples_6clusters_count)[:, 4])
            # meanclass5 = np.mean(np.array(noIP_samples_6clusters_count)[:, 5])
            #
            # noIP_6clusters_mean10rounds_samplescount = [meanclass0, meanclass1, meanclass2, meanclass3, meanclass4, meanclass5]
            #
            # # Con IP
            # meanclass0 = np.mean(np.array(withIP_samples_4clusters_count)[:, 0])
            # meanclass1 = np.mean(np.array(withIP_samples_4clusters_count)[:, 1])
            # meanclass2 = np.mean(np.array(withIP_samples_4clusters_count)[:, 2])
            # meanclass3 = np.mean(np.array(withIP_samples_4clusters_count)[:, 3])
            #
            # withIP_4clusters_mean10rounds_samplescount = [meanclass0, meanclass1, meanclass2, meanclass3]
            #
            # meanclass0 = np.mean(np.array(withIP_samples_6clusters_count)[:, 0])
            # meanclass1 = np.mean(np.array(withIP_samples_6clusters_count)[:, 1])
            # meanclass2 = np.mean(np.array(withIP_samples_6clusters_count)[:, 2])
            # meanclass3 = np.mean(np.array(withIP_samples_6clusters_count)[:, 3])
            # meanclass4 = np.mean(np.array(withIP_samples_6clusters_count)[:, 4])
            # meanclass5 = np.mean(np.array(withIP_samples_6clusters_count)[:, 5])
            #
            # withIP_6clusters_mean10rounds_samplescount = [meanclass0, meanclass1, meanclass2, meanclass3, meanclass4,
            #                                       meanclass5]



            summary = {'washout_size':washout_size,'scale':scale,'connectivity':connectivity,'Number_of_labels':N
                            ,'time_steps':time_steps,'max_length':max_length,'initial_crop':initial_crop,'maxSamples':maxSamples,'maxTrainingSamples':maxTrainingSamples
                            ,'increment':increment,'channels':channels,'mean':mean,'std':std,'learning_rate':learning_rate
                            ,'Architecture':str(experimentName),'num_IP_epochs':3,'batch_size':maxSamples*N
                            ,'Rules':str(rules),'evolutionSR_NoIP':str(evolutionSR_NoIP),'evolutionSR_WithIP':str(evolutionSR_WithIP)
                            ,'noIP_2clusters_count':str(noIP_2clusters_count),'noIP_3clusters_count':str(noIP_3clusters_count),'noIP_4clusters_count':str(noIP_4clusters_count),'noIP_5clusters_count':str(noIP_5clusters_count),'noIP_6clusters_count':str(noIP_6clusters_count),'noIP_7clusters_count':str(noIP_7clusters_count),'noIP_8clusters_count':str(noIP_8clusters_count)
                            ,'withIP_2clusters_count':str(withIP_2clusters_count),'withIP_3clusters_count':str(withIP_3clusters_count),'withIP_4clusters_count':str(withIP_4clusters_count),'withIP_5clusters_count':str(withIP_5clusters_count),'withIP_6clusters_count':str(withIP_6clusters_count),'withIP_7clusters_count':str(withIP_7clusters_count),'withIP_8clusters_count':str(withIP_8clusters_count),'noIP_Silhouette_score_6clusters':str(noIP_Silhouette_score)
                            ,'withIP_Silhouette_score_6clusters':str(withIP_Silhouette_score),'mean_noIP_2clusters_count':str(np.mean(noIP_2clusters_count)),'mean_noIP_3clusters_count':str(np.mean(noIP_3clusters_count)),'mean_noIP_4clusters_count':str(np.mean(noIP_4clusters_count)),'mean_noIP_5clusters_count':str(np.mean(noIP_5clusters_count)),'mean_noIP_6clusters_count':str(np.mean(noIP_6clusters_count)),'mean_noIP_7clusters_count':str(np.mean(noIP_7clusters_count)),'mean_noIP_8clusters_count':str(np.mean(noIP_8clusters_count))
                            ,'mean_withIP_2clusters_count':str(np.mean(withIP_2clusters_count)),'mean_withIP_3clusters_count':str(np.mean(withIP_3clusters_count)),'mean_withIP_4clusters_count':str(np.mean(withIP_4clusters_count)),'mean_withIP_5clusters_count':str(np.mean(withIP_5clusters_count)),'mean_withIP_6clusters_count':str(np.mean(withIP_6clusters_count)),'mean_withIP_7clusters_count':str(np.mean(withIP_7clusters_count)),'mean_withIP_8clusters_count':str(np.mean(withIP_8clusters_count))
                            ,'Silouethe_scores_means_IP':str(Silouethe_scores_means_IP),'Silouethe_scores_means_noIP':str(Silouethe_scores_means_noIP),'noIP_4clusters_mean10rounds_samplescount':str(noIP_samples_4clusters_count),'noIP_6clusters_mean10rounds_samplescount':str(noIP_samples_6clusters_count),'withIP_4clusters_mean10rounds_samplescount':str(withIP_samples_4clusters_count),'withIP_6clusters_mean10rounds_samplescount':str(withIP_samples_6clusters_count)}

            writeDictinTXTfile(savePath, experimentName, summary, 'summary')

            print('Datos_Resumen_Guardados')








