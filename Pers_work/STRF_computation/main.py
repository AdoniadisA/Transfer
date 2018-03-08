import os, io, re, scipy.io, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nems.utilities as nu
import nems.db as nd
import nems.utilities.baphy
from f_pre_process import getInsOuts
from f_conv1D_model import mk_and_fit_conv1Dmodel,prediction_score_and_plots

parmfilepath='/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'

options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}
event_times, spike_dict, stim_dict, state_dict = nu.baphy.baphy_load_recording(parmfilepath,options)
cellidx = 21

for cell in range(1):
    time_window = 10

    X,Y,W,Z,t_12 = getInsOuts(cellidx,event_times,options['rasterfs'],stim_dict,spike_dict,False)
    #cellidx,event_times,fs_spectro,stim_dict,spike_dict,boolFigure=False

    model,history = mk_and_fit_conv1Dmodel(X,Y,1000,40,time_window,[0,0.001],0.005,'relu','poisson','SGD',0.2,False)
    predicted = model.predict(W)
    predicted_fit = model.predict(X)
    #Z.shape

    score = prediction_score_and_plots(model,X,Y,W,Z,t_12,predicted,history,cellidx,time_window,onTest=True,fig = False)
    score_fit = prediction_score_and_plots(model,X,Y,W,Z,t_12,predicted_fit,history,cellidx,time_window,onTest=False,fig = False)
    print('score test : {} '.format(score))
    print('score_fit : {} '.format(score_fit))
    
    
save = False
if save:
    # SAVE MODEL
    config = model.get_config()[0]
    print(config)
    save_name = config['class_name'] + '_'  + config['config']['activation'] + '_compiler-' + \
        loss + '-' + str(lr) + '_'+ '_ker-' + \
        str(config['config']['kernel_initializer']['config']['minval']) + '-' + \
        str(config['config']['kernel_initializer']['config']['maxval']) + '_' + \
        'epochs-' + str(epochs) + '_batch-' + str(batch_size) 

    model.save('STRF_computation/models_trained/' + save_name)