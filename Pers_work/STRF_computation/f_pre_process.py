import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def raster(event_name,cellid,event_times,spike_dict,rasterfs,PreEndTime=200,PostBegTime=500,split=False):
    # spike_dict : spikes of a hole recording
    # unitidx : unit's spikes we want to look at
    # stim_dict : stimuli, used for the names of the events
    # eventidx : idx of event we want to look at
    # rasterfs : spike frequency
    
    binlen=1.0/rasterfs
    h=np.array([])
    ff = (event_times['name']==event_name)
    ## pull out each epoch from the spike times, generate a raster of spike rate
    halfNb = int(event_times.loc[ff].shape[0]/2)        
    
    #m = np.empty([2,PostBegTime-PreEndTime])
    m = np.empty([2,550])
    
    for idx,(i,d) in enumerate(event_times.loc[ff].iterrows()):
        #edges=np.arange(d['start']+PreEndTime/rasterfs,d['start']+PostBegTime/rasterfs+binlen,binlen)
        edges=np.arange(d['start'],d['end']+binlen,binlen)
        th,e=np.histogram(spike_dict[cellid],edges)
        th=np.reshape(th,[1,-1])
        if h.size==0:
            # lazy hack: intialize the raster matrix without knowing how many bins it will require
            h=th
        else:
            # concatenate this repetition, making sure binned length matches
            if th.shape[1]<h.shape[1]:
                h=np.concatenate((h,np.zeros([1,h.shape[1]])),axis=0)
                h[-1,:]=np.nan
                h[-1,:th.shape[1]]=th
            else:
                h=np.concatenate((h,th[:,:h.shape[1]]),axis=0)
        if idx == halfNb-1 and split==True:
            m[0,:] = np.nanmean(h,axis=0)[0:m.shape[1]]
            h=np.array([])

    if split==True:
        m[1,:] = np.nanmean(h,axis=0)[0:m.shape[1]]
    else:
        m = np.nanmean(h,axis=0)[0:m.shape[1]]

    return h,m



def getTrainTestTimes(event_times,trainNb,testNb):
    # event_times : timings of events 
    # trainNb : number of stimuli presented for the trains
    # testNb : number of stimuli presented for the tests
    
    wavEvents = event_times[event_times['name'].str.contains('.wav')]
    occurences =  wavEvents['name'].value_counts(sort=True)

    Train_names = list(occurences[occurences==trainNb].index)
    Test_names = list(occurences[occurences==testNb].index)
    if Train_names == [] or Test_names == [] :
        raise ValueError('wrong trainNb or testNb')        
    
    Train_times = pd.DataFrame(columns={'name','start','end'})
    Train_times = Train_times[['name','start','end']] #Order the columns
    Test_times = Train_times.copy()

    #Get stimuli onset and offset times for trains
    trial_indexs = event_times['name'][event_times['name']=='TRIAL'].index
    idx1 = 0; idx2 = 0;
    for trial_idx in trial_indexs:
        name = event_times.iloc[trial_idx+1]['name']
        if name in Train_names :
            Train_times.at[idx1,'name'] = name
            Train_times.at[idx1,'start'] = event_times.iloc[trial_idx+3]['end']
            Train_times.at[idx1,'end'] = event_times.iloc[trial_idx+4]['start']
            idx1 +=1
        elif name in Test_names :
            Test_times.at[idx2,'name'] = name
            Test_times.at[idx2,'start'] = event_times.iloc[trial_idx+3]['end']
            Test_times.at[idx2,'end'] = event_times.iloc[trial_idx+4]['start']
            idx2 +=1
        else : 
            raise ValueError('Neither a Test nor a Train stimuli name')

    #Train_times = Train_times.sort_values('name')
    #Test_times = Test_times.sort_values('name')

    return Train_times,Test_times


def getInsOuts(cellidx,event_times,rasterfs,stim_dict,spike_dict,boolFigure=False):
    print('Fetching ins & outs...\n')
    # fix random seed for reproducibility
    np.random.seed(7)

    # Segregate Train and Tests
    Train_times,Test_times = getTrainTestTimes(event_times,3,24)

    # Compute on a specific cell --> TODO : for all
    cellid = list(spike_dict.keys())[cellidx] 

    #_________________Training input (X)_______________________#
    stim_shape = np.shape(stim_dict[list(stim_dict.keys())[0]])
    nbTrains = len(set(Train_times['name']))

    # sound_time : from end of prestimsilence and beg of poststimsilence
    PreStimidx = list(event_times['name']).index('PreStimSilence')
    Endidx = event_times.columns.get_loc('end')
    #PostBegTime = int(event_times.iloc[PreStimidx+1,Endidx-1]*rasterfs)
    #PreEndTime = int(event_times.iloc[PreStimidx,Endidx]*rasterfs)
    PostBegTime = 550
    PreEndTime = 0
    sound_time = PostBegTime - PreEndTime
    

    X = np.zeros( (nbTrains,sound_time,stim_shape[0]) )
    for idx,event_name in enumerate(set(Train_times['name'])):
        X[idx,:,:] = np.transpose(stim_dict[event_name][:,PreEndTime:PostBegTime]) 
    X = X/X.max()

    #_________________Training output (Y)_______________________#
    Y = np.zeros( (nbTrains,sound_time,1) )
    for idx,event_name in enumerate(set(Train_times['name'])):
        h,m = raster(event_name,cellid,event_times,spike_dict,rasterfs,PreEndTime,PostBegTime)
        Y[idx,:,0] = np.transpose(m[0:np.size(Y,1)])
    Y = Y/Y.max()

    #_________________TEST input (W)_______________________#
    nbTests = len(set(Test_times['name']))
    W = np.zeros( (nbTests,sound_time,stim_shape[0]) )
    for idx,event_name in enumerate(set(Test_times['name'])):
        W[idx,:,:] = np.transpose(stim_dict[event_name][:,PreEndTime:PostBegTime]) 
    W = W/W.max()

    #_________________Test output (Z)_______________________#
    Z = np.zeros( (nbTests,sound_time) )
    t_12 = np.zeros(3)
    for idx,event_name in enumerate(set(Test_times['name'])):
        h,m = raster(event_name,cellid,event_times,spike_dict,rasterfs,PreEndTime,PostBegTime,split=False)
        #Z[idx,:,:] = np.swapaxes(m,0,1)
        Z[idx,:] = m
        if boolFigure :
            plt.figure()
            #plt.plot(Z[idx,:,0])
            #plt.plot(Z[idx,:,1])
            plt.plot(Z[idx,:])
            plt.title('Split spike rates from sound{}'.format(idx))
        #t_12[idx] = np.corrcoef(Z[idx,:,0],Z[idx,:,1])[0,1]
        t_12[idx] = 1
        #print('Correlation between sound{}\'s split spike rates: {}'.format(idx,t_12[idx]))

    return X,Y,W,Z,t_12


