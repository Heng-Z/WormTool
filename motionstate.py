#%%
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from WormTool import timeseries 
class MotionState:
    def __init__(self,curvature,window=50):
        if curvature.shape[0] < curvature.shape[1]:
            curvature = curvature.T
        eigen_basis = np.load('../excursion/Eigenworm_basis/eigenworms_N2.npy')
        self.eigenworm = curvature@eigen_basis[:,:2]
        self.L = curvature.shape[0]
        self.window = window
        self.curvature = curvature
        

    def get_motion_state(self,std2mean=0.6):
        phase = np.arctan2(self.eigenworm[:,1],self.eigenworm[:,0])
        phase_unwrap = np.unwrap(phase)
        phase_unwrap_sm = np.convolve(phase_unwrap,np.ones(15)/15,mode='same')
        dphase = np.gradient(phase_unwrap_sm,0.01)
        window = self.window
        curvature = self.curvature
        state = np.zeros((self.L//window+1,))
        for i in range(state.shape[0]):
            mean_ps = np.mean(dphase[i*window:min((i+1)*window,self.L)]) # mean phase speed
            std_ps = np.std(dphase[i*window:min((i+1)*window,self.L)]) # std of phase speed
            if std_ps<std2mean*mean_ps and mean_ps>0:
                state[i] = 1
            else:
                state[i] = 0
        self.state = state
        return state

    def piecewise_state(self):
        if not hasattr(self,'state'):
            self.get_motion_state()
        # piecewise function of state
        state_pw = np.zeros((self.L,))
        for i in range(self.state.shape[0]):
            state_pw[i*self.window:min((i+1)*self.window,self.L)] = self.state[i]
        self.state_pw = state_pw
        return state_pw
    
    def extract_all_forward(self):
        # extract all forward motion
        forward = []
        for i in range(self.state.shape[0]):
            if self.state[i] == 1:
                forward.append(self.curvature[i*self.window:min((i+1)*self.window,self.L),:])
        # Concatenate all forward motion
        forward = np.concatenate(forward,axis=0)
        return forward
    
    def forward_indices(self):
        # return indices of forward motion
        indices = []
        for i in range(self.state.shape[0]):
            if self.state[i] == 1:
                indices.append(np.arange(i*self.window,min((i+1)*self.window,self.L)))
        return np.concatenate(indices,axis=0)
        
    def find_long_run(self,min_run=1000):
        # Search for long run of state 1
        # e.g.  0001111100110 has a long run of 5 
        # find the start and end indices of the long run
        # return a list of tuples (start,end)
        if not hasattr(self,'state'):
            self.get_motion_state()
        state = self.state
        long_run = []
        i=0
        while i < len(state):
            if state[i] == 1:
                start = i
                while i<state.shape[0] and state[i] == 1:
                    i += 1
                end = i
                if (end-start)*self.window>min_run:
                    long_run.append((start*self.window,min(end*self.window,self.L)))
            i += 1
        return long_run
    

class PropagatedBend:
    def __init__(self,n2p_zc_seq,p2n_zc_seq,n2p_zc_seq_next,pos_peak_seq,neg_peak_seq,range_start,range_end):
        self.n2p_zc_seq = n2p_zc_seq
        self.p2n_zc_seq = p2n_zc_seq
        self.n2p_zc_seq_next = n2p_zc_seq_next
        self.pos_peak_seq = pos_peak_seq
        self.neg_peak_seq = neg_peak_seq
        self.range_start = range_start
        self.range_end = range_end

    def get_bending_bias(self,full_curvature):
        '''
        Get the bending bias of each point in one cycle of propagated bend.

        Parameters
        ----------
        full_curvature : array of shape (T,N)
            Curvature of the whole worm.
        '''
        assert self.range_end - self.range_start == len(self.n2p_zc_seq)
        assert self.n2p_zc_seq is not None and self.n2p_zc_seq_next is not None
        bending_bias = np.zeros((self.range_end-self.range_start,))
        for i in range(self.range_end-self.range_start):
            bending_bias[i] = np.mean(full_curvature[self.n2p_zc_seq[i]:self.n2p_zc_seq_next[i],i+self.range_start])
            
        return bending_bias
    
    def get_peak_amp(self,full_curvature):
        '''
        Get the amplitude of each point in one cycle of propagated bend.

        Parameters
        ----------
        full_curvature : array of shape (T,N)
            Curvature of the whole worm. Smoothed curvature will be better
        '''
        assert self.range_end - self.range_start == len(self.pos_peak_seq)
        assert self.pos_peak_seq is not None and self.neg_peak_seq is not None

        pos_peak_amp = full_curvature[self.pos_peak_seq,np.arange(self.range_start,self.range_end)]
        neg_peak_amp = full_curvature[self.neg_peak_seq,np.arange(self.range_start,self.range_end)]

        self.pos_peak_amp = pos_peak_amp
        self.neg_peak_amp = neg_peak_amp
        return pos_peak_amp,neg_peak_amp
    
class CurvInfo:
    def __init__(self,curvature,**kwargs):
        self.range_min = kwargs.get('range_min',5)
        self.range_max = kwargs.get('range_max',95)
        self.filter_size = kwargs.get('filter_size',9)
        self.prominence = kwargs.get('prominence',5)
        self.distance = kwargs.get('distance',150)
        self.emd_min_period = kwargs.get('emd_min_period',120) # For discarding first few high frequency EMD modes
        self.max_period = kwargs.get('max_period',500) # For discarding atypical propagated bend
        
        self.curvature = curvature

    def get_propagated_peaks(self):
        pos_peak_seq_ls,neg_peak_seq_ls,pos_peak_ls,neg_peak_ls = propagated_peaks(self.curvature,self.prominence,self.distance,self.range_min,self.range_max,self.filter_size)
        self.pos_peak_seq_ls = pos_peak_seq_ls
        self.neg_peak_seq_ls = neg_peak_seq_ls
        self.pos_peak_ls = pos_peak_ls
        self.neg_peak_ls = neg_peak_ls
        return pos_peak_seq_ls,neg_peak_seq_ls,pos_peak_ls,neg_peak_ls

    def get_propagated_zero_crossing(self):
        p2n_zc_seq_ls,n2p_zc_seq_ls,p2n_zc_ls,n2p_zc_ls = propagated_zero_crossing(
                                self.curvature,self.filter_size,self.emd_min_period,
                                self.range_min,self.range_max)
        self.n2p_zc_seq_ls = n2p_zc_seq_ls
        self.p2n_zc_seq_ls = p2n_zc_seq_ls
        self.n2p_zc_ls = n2p_zc_ls
        self.p2n_zc_ls = p2n_zc_ls
        return p2n_zc_seq_ls,n2p_zc_seq_ls,p2n_zc_ls,n2p_zc_ls

    def get_propagated_bends_ls(self):
        range_min = self.range_min
        range_max = self.range_max
        filter_size = self.filter_size
        prominence = self.prominence
        distance = self.distance
        emd_min_period = self.emd_min_period
        max_period = self.max_period

        if hasattr(self,pos_peak_seq_ls):
            pos_peak_seq_ls = self.pos_peak_seq_ls
            neg_peak_seq_ls = self.neg_peak_seq_ls
            pos_peak_ls = self.pos_peak_ls
            neg_peak_ls = self.neg_peak_ls
        else:
            pos_peak_seq_ls,neg_peak_seq_ls,pos_peak_ls,neg_peak_ls = self.get_propagated_peaks()
        
        if hasattr(self,n2p_zc_seq_ls):
            n2p_zc_seq_ls = self.n2p_zc_seq_ls
            p2n_zc_seq_ls = self.p2n_zc_seq_ls
        else:
            n2p_zc_seq_ls,p2n_zc_seq_ls = self.get_propagated_zero_crossing()
        
        propagated_bends_ls = []
        for i in range(len(n2p_zc_seq_ls)-1):
            n2p_zc_seq = n2p_zc_seq_ls[i]
            n2p_zc_seq_next = n2p_zc_seq_ls[i+1]
            median1 = np.median(n2p_zc_seq)
            median2 = np.median(n2p_zc_seq_next)
            if median2 - median1 > max_period:
                continue
            
            p2n_zc_seq = None
            # Search for the p2n zero crossing sequence that is between n2p_zc_seq and n2p_zc_seq_next
            for j in range(max(0,i-3),min(i+3,len(p2n_zc_seq_ls))):
                median = np.median(p2n_zc_seq_ls[j])
                if median > median1 and median < median2:
                    p2n_zc_seq = p2n_zc_seq_ls[j]
                    break
            if p2n_zc_seq is None:
                continue

            # seach for the pos_peak_seq between n2p_zc_seq and n2p_zc_seq_next
            pos_peak_seq = None
            for j in range(max(0,i-3),min(i+3,len(pos_peak_seq_ls))):
                median = np.median(pos_peak_seq_ls[j])
                if median > median1 and median < median2:
                    pos_peak_seq = pos_peak_seq_ls[j]
                    break
            if pos_peak_seq is None:
                continue

            # seach for the neg_peak_seq between n2p_zc_seq and n2p_zc_seq_next
            neg_peak_seq = None
            for j in range(max(0,i-3),min(i+3,len(neg_peak_seq_ls))):
                median = np.median(neg_peak_seq_ls[j])
                if median > median1 and median < median2:
                    neg_peak_seq = neg_peak_seq_ls[j]
                    break
            if neg_peak_seq is None:
                continue

            propagated_bends = PropagatedBend(n2p_zc_seq,p2n_zc_seq,n2p_zc_seq_next,pos_peak_seq,neg_peak_seq,range_min,range_max)
            propagated_bends_ls.append(propagated_bends)
        self.propagated_bends_ls = propagated_bends_ls

        return propagated_bends_ls


def get_propagated_bends_ls(n2p_zc_seq_ls,p2n_zc_seq_ls,pos_peak_seq_ls,neg_peak_seq_ls):
    range_min = 5
    range_max = 95
    filter_size = 9
    prominence = 5
    distance = 120
    emd_min_period = 120
    max_period = 500


    propagated_bends_ls = []
    for i in range(len(n2p_zc_seq_ls)-1):
        n2p_zc_seq = n2p_zc_seq_ls[i]
        n2p_zc_seq_next = n2p_zc_seq_ls[i+1]
        median1 = np.median(n2p_zc_seq)
        median2 = np.median(n2p_zc_seq_next)
        if median2 - median1 > max_period:
            continue
        
        p2n_zc_seq = None
        # Search for the p2n zero crossing sequence that is between n2p_zc_seq and n2p_zc_seq_next
        for j in range(len(p2n_zc_seq_ls)):
            median = np.median(p2n_zc_seq_ls[j])
            if median > median1 and median < median2:
                p2n_zc_seq = p2n_zc_seq_ls[j]
                break
        if p2n_zc_seq is None:
            continue

        # seach for the pos_peak_seq between n2p_zc_seq and n2p_zc_seq_next
        pos_peak_seq = None
        for j in range(len(pos_peak_seq_ls)):
            median = np.median(pos_peak_seq_ls[j])
            if median > median1 and median < median2:
                pos_peak_seq = pos_peak_seq_ls[j]
                break
        if pos_peak_seq is None:
            continue

        # seach for the neg_peak_seq between n2p_zc_seq and n2p_zc_seq_next
        neg_peak_seq = None
        for j in range(len(neg_peak_seq_ls)):
            median = np.median(neg_peak_seq_ls[j])
            if median > median1 and median < median2:
                neg_peak_seq = neg_peak_seq_ls[j]
                break
        if neg_peak_seq is None:
            continue
        propagated_bends = PropagatedBend(n2p_zc_seq,p2n_zc_seq,n2p_zc_seq_next,pos_peak_seq,neg_peak_seq,range_min,range_max)
        propagated_bends_ls.append(propagated_bends)

    return propagated_bends_ls

def propagated_peaks(curvature, prominence=5, distance=120, range_min=5, range_max=95,filter_size=9):
    '''
    Find the timing of peaks of each points in one cycle of propagated bend.

    Parameters
    ----------
    curvature : array of shape (T,N)
    
    prominence : float, optional
        Minimum prominence of peaks. The default is 5.
    
    distance : int, optional
        Minimum distance between peaks. The default is 120.

    Returns
    -------
    pos_peak_seq_ls : list of array of shape (N,)
        List of sequences of positive peak for each point in one cycle of propagated bend.

    neg_peak_seq_ls : list of array of shape (N,)
        List of sequences of negative peak for each point in one cycle of propagated bend.
    '''
    pos_peak_seq_ls = []
    neg_peak_seq_ls = []
 
    curvature_sm = scipy.signal.medfilt(curvature,[filter_size,1]) # smooth curvature with median filter along the time axis
    pos_peak_ls = [scipy.signal.find_peaks(curvature_sm[:,i],prominence=prominence,distance=distance)[0] for i in range(range_min,range_max)]
    neg_peak_ls = [scipy.signal.find_peaks(-curvature_sm[:,i],prominence=prominence,distance=distance)[0] for i in range(range_min,range_max)]

    if len(pos_peak_ls[0]) == 0 or len(neg_peak_ls[0]) == 0:
        raise ValueError('No peak found in the first point of the cycle of propagated bend')


    num_pos_peak_first = len(pos_peak_ls[0]) # number of peaks of the first point, which is defined as the number of propagated bend
    # Register positive peaks in a cycle of propagated bend
    for i in range(num_pos_peak_first):
        pos_peak_seq = np.zeros((range_max-range_min,))
        ref = pos_peak_ls[0][i]
        for j in range(range_max-range_min):
            following_peak = pos_peak_ls[j][np.argmin(np.abs(pos_peak_ls[j]-ref))]
            if following_peak - ref < 50 and following_peak - ref >= -50:
                pos_peak_seq[j] = following_peak
                ref = following_peak
            else:
                pos_peak_seq[j] = np.nan

        # if the number of nan is smaller than 3% of the total number of points, then replace nan with their previous value and register the sequence
        if np.sum(np.isnan(pos_peak_seq)) < 0.03*(range_max-range_min):
            for j in range(1,range_max-range_min):
                if np.isnan(pos_peak_seq[j]):
                    pos_peak_seq[j] = pos_peak_seq[j-1]

            pos_peak_seq_ls.append(pos_peak_seq.astype(np.int32))
        else: 
            print('too many nan in pos_peak_seq')
    
    num_neg_peak_first = len(neg_peak_ls[0]) # number of peaks of the first point.
    # Register negative peaks in a cycle of propagated bend
    for i in range(num_neg_peak_first):
        neg_peak_seq = np.zeros((range_max-range_min,))
        ref = neg_peak_ls[0][i]
        for j in range(range_max-range_min):
            following_peak = neg_peak_ls[j][np.argmin(np.abs(neg_peak_ls[j]-ref))]
            if following_peak - ref < 50 and following_peak - ref >= -50:
                neg_peak_seq[j] = following_peak
                ref = following_peak
            else:
                neg_peak_seq[j] = np.nan
        # if the number of nan is smaller than 3% of the total number of points, then replace nan with their previous value and register the sequence
        if np.sum(np.isnan(neg_peak_seq)) < 0.03*(range_max-range_min):
            for j in range(1,range_max-range_min):
                if np.isnan(neg_peak_seq[j]):
                    neg_peak_seq[j] = neg_peak_seq[j-1]
            neg_peak_seq_ls.append(neg_peak_seq.astype(np.int32))
    return pos_peak_seq_ls,neg_peak_seq_ls,pos_peak_ls,neg_peak_ls

def propagated_zero_crossing(curvature,filter_size,p=100,range_min=5,range_max=95):
    '''
    Parameters
    ----------
    curvature : array of shape (T,N)
    '''

    p2n_zc_ls = []
    n2p_zc_ls = []
    curvature_sm = scipy.signal.medfilt(curvature,[filter_size,1]) # smooth curvature with median filter along the time axis
    for i in range(range_min,range_max):
        n2p_zc,p2n_zc = timeseries.zero_crossing_index(curvature_sm[:,i],p=p)
        p2n_zc_ls.append(p2n_zc)
        n2p_zc_ls.append(n2p_zc)
    # Register the zero crossing sequence along body
    n2p_zc_seq_ls = []
    p2n_zc_seq_ls = []

    num_n2p_zc_first = len(n2p_zc_ls[0]) # number of zero crossing of the first point, which is defined as the number of propagated bend
    # Register the negative to positive zero crossing in a cycle of propagated bend
    for i in range(num_n2p_zc_first):
        n2p_zc_seq = np.zeros((range_max-range_min,))
        ref = n2p_zc_ls[0][i]
        for j in range(range_max-range_min):
            following_zc = n2p_zc_ls[j][np.argmin(np.abs(n2p_zc_ls[j]-ref))]
            if following_zc - ref < 50 and following_zc - ref >= -50:
                n2p_zc_seq[j] = following_zc
                ref = following_zc
            else:
                n2p_zc_seq[j] = np.nan
        # if the number of nan is smaller than 3% of the total number of points, then replace nan with their previous value and register the sequence
        if np.sum(np.isnan(n2p_zc_seq)) < 0.03*(range_max-range_min):
            for j in range(1,range_max-range_min):
                if np.isnan(n2p_zc_seq[j]):
                    n2p_zc_seq[j] = n2p_zc_seq[j-1]
            n2p_zc_seq_ls.append(n2p_zc_seq.astype(np.int32))
    
    num_p2n_zc_first = len(p2n_zc_ls[0]) # number of zero crossing of the first point.
    # Register the positive to negative zero crossing in a cycle of propagated bend
    for i in range(num_p2n_zc_first):
        p2n_zc_seq = np.zeros((range_max-range_min,))
        ref = p2n_zc_ls[0][i]
        for j in range(range_max-range_min):
            following_zc = p2n_zc_ls[j][np.argmin(np.abs(p2n_zc_ls[j]-ref))]
            if following_zc - ref < 50 and following_zc - ref >= -50:
                p2n_zc_seq[j] = following_zc
                ref = following_zc
            else:
                p2n_zc_seq[j] = np.nan
        # if the number of nan is smaller than 3% of the total number of points, then replace nan with their previous value and register the sequence
        if np.sum(np.isnan(p2n_zc_seq)) < 0.03*(range_max-range_min):
            for j in range(1,range_max-range_min):
                if np.isnan(p2n_zc_seq[j]):
                    p2n_zc_seq[j] = p2n_zc_seq[j-1]
            p2n_zc_seq_ls.append(p2n_zc_seq.astype(np.int32))
        

    return p2n_zc_seq_ls,n2p_zc_seq_ls,p2n_zc_ls,n2p_zc_ls




    





# %%
