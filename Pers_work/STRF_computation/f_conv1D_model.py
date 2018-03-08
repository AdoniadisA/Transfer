import numpy as np
import time
import matplotlib.pyplot as plt

def mk_and_fit_conv1Dmodel(X,Y,epochs,batch_size,time_window,kernel_init,lr,activation,loss,optim,validation_split,early_stop=False):
    #### KERAS 1D CONV MODEL
    #import os
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""

    #import tensorflow as tf
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Create your first MLP in Keras
    from keras.models import Sequential
    from keras.layers import Conv1D,Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.constraints import non_neg,min_max_norm
    import keras.initializers
    import keras.optimizers

    # create model
    model = Sequential()
    layer = Conv1D(input_shape=np.shape(X)[1:3],filters=1,kernel_size=time_window,strides=1,
                padding='causal',activation='relu',dilation_rate=1,use_bias=True,
                bias_initializer='random_uniform')

    layer.kernel_initializer = keras.initializers.RandomUniform(minval=kernel_init[0], maxval=kernel_init[1], seed=None)
    model.add(layer)
    #layer2 = Dense(1,use_bias=True, kernel_initializer='zeros', bias_initializer='random_uniform',kernel_constraint=min_max_norm(min_value=1, max_value=1, axis=0))
    #model.add(layer2)
    # Compile model
    sgd = getattr(keras.optimizers,optim)(lr=lr)
    model.compile(loss = loss, optimizer=sgd)

    # Fit the model
    early_cbk = []
    if early_stop:
        early_cbk = [keras.callbacks.EarlyStopping(patience = 20,verbose=1)]

    start_time = time.time()
    history = model.fit(X,Y,validation_split = validation_split,epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks =early_cbk)
    print('Elapsed fitting time : {}'.format(time.time() - start_time))

    return model,history

#Compare out and predicted
def prediction_score_and_plots(model,X,Y,W,Z,t_12,predicted,history,cellidx,time_window,onTest=True,fig = True):
    import random as rand
    if fig:
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
        #plt.plot(history_early.history['val_loss'],'--')
        plt.legend(('loss','loss with validation','validation_loss'))
        plt.title('loss over training')
        plt.figure

    score = []
    if onTest:
        for idx in range(Z.shape[0]):
            example = idx
            if fig :
                plt.figure()
                #plt.plot(Z[example,:,0])
                #plt.plot(Z[example,:,1])
                plt.plot(Z[example,:])
                plt.plot(predicted[example,:,0])
                plt.title("individual PSTH of {}th cell from sound {}".format(cellidx,example))
                plt.legend(('output1','output2','prediction','prediction_early_stop'))        
            #c1 = np.corrcoef(Z[example,:,0],predicted[example,:,0])[0,1]
            #c2 = np.corrcoef(Z[example,:,1],predicted[example,:,0])[0,1]
            #score.append( (c1**2/2 + c2**2/2)/t_12[example] )
            score.append( np.corrcoef(Z[example,:].squeeze(),predicted[example,:,0].squeeze())[0,1] )
            #print('Explained Score for sound {}: {}'.format(example,score[example]))
    else :
        for idx in range(Y.shape[0]):
            example = idx #rand.randint(0,Y.shape[0]-1)   
            score.append( np.corrcoef(Y[example,:].squeeze(),predicted[example,:,0].squeeze())[0,1] )
            if fig:
                plt.figure()
                plt.plot(Y[example,:,0])
                plt.plot(predicted[example,:,0])
                plt.title("individual PSTH of {}th cell from sound {}".format(cellidx,example))
                plt.legend(('output','prediction'))
    
    score = np.mean(score)
    #print('meanScore for cell {} : {}'.format(cellidx,score))
    #Plot STRF
    dirac_spec = np.concatenate((np.ones((X.shape[2],1)),np.zeros((X.shape[2],time_window-1))),axis=1)
    weights = model.get_weights()[0].squeeze().transpose()

    STRF = np.zeros(weights.shape)
    for idx in range(weights.shape[0]):
        conv = np.convolve(dirac_spec[idx],weights[idx],mode='full')
        STRF[idx][:] = conv[0:weights.shape[1]]

    if fig:
        plt.figure()
        plt.imshow(STRF)
    
    return score