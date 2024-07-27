#%%
import scipy.signal
import numpy as np
from csaps import csaps
from scipy.signal import find_peaks,peak_prominences
import scipy.io
from vmdpy import VMD
import scipy.interpolate
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.interpolate import interp1d

def find_period_amp(k_t):
    # find the period and amplitude of undulatory signal k_t
    [pos,_] = scipy.signal.find_peaks(k_t)
    td = pos[1:]-pos[:-1]
    # half_ind = len(td)//2
    half_ind = 1 
    if len(td) < 2:
        half_ind = 0
    amplitude = np.mean(abs(k_t[pos[half_ind:]]))
    period = np.mean(td[half_ind:])  # mean interval between peaks' indices
    if np.isnan(period):
        period = 1e6
    return period,amplitude

def get_head_curve_traj(centerline_data,num=10,DeltaX=None):
    T = centerline_data.shape[2]
    head_curve_traj = np.zeros(T)
    curve_t = np.zeros((centerline_data.shape[0]-2,T))
    for t in range(T):
        x = centerline_data[:,0,t].astype(np.float64)
        y = centerline_data[:,1,t].astype(np.float64)
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]
        theta = np.arctan2(dy,dx)
        theta = np.unwrap(theta)
        if Delta_x is None:
            Delta_x = 1/(centerline_data.shape[0]-1)
        curve_t[:,t] = (theta[1:]-theta[:-1])/Delta_x
        head_curve_traj[t] = (theta[num:2*num]-theta[:num]).mean()/num/Delta_x
    return head_curve_traj,curve_t

def csaps_derivative(t,x,axis=0,smooth=0.95,upsample=3):
    '''
    1. upsample the time series using cubic spline
    2. calculate the derivative of the upsampled time series
    3. downsample the derivative

    Parameters
    ----------
    t : array of shape (T,)
        Time stamps.
    x : array of shape (T,) or (T,N) or (N,T)
        Time series.
    smooth : float, optional
        Smoothing parameter. The default is 0.95.
    upsample : int, optional
        Upsample the time series. The default is 3.

    Returns
    -------
    dx : array of shape (T,)
        Derivative of the time series.
    '''
    t_upsample = np.linspace(t[0],t[-1],int(upsample*len(t)))
    if len(x.shape) == 1:
        x_upsample = csaps(t,x,t_upsample,smooth=smooth)
        dx = np.gradient(x_upsample,t_upsample)
        dx = csaps(t_upsample,dx,t,smooth=smooth)
    else:
        assert len(x.shape) == 2
        transposed = False
        if axis == 0:
            assert x.shape[0] == len(t)
        elif axis == 1:
            assert x.shape[1] == len(t)
            x = x.T # (N,T) -> (T,N)
            transposed = True
        dx = np.zeros(x.shape)
        for i in range(x.shape[1]):
            x_upsample = csaps(t,x[:,i],t_upsample,smooth=smooth)
            dx_upsample = np.gradient(x_upsample,t_upsample)
            dx[:,i] = csaps(t_upsample,dx_upsample,t,smooth=smooth)
        if transposed:
            dx = dx.T
    return dx

def body_curv_dcurv(centerline,x_seg,t):
    ''' 
    Calculate the curvature and the time derivative of curvature

    Parameters
    ----------
    centerline : array of shape (N,2,T)
        Centerline of the worm.
    x_seg : array of shape (N,)
        The body length coordinate of the centerline.
    t : array of shape (T,)
        Time stamps.

    Returns
    -------
    curv_t : array of shape (N-2,T)
        Curvature of the worm.
    dcurv_t : array of shape (N-2,T)
        Time derivative of curvature of the worm.
    '''
    N = centerline.shape[0]
    T = centerline.shape[2]
    curv_t = np.zeros((N-2,T))
    dcurv_t = np.zeros((N-2,T))
    Dx = x_seg[1] - x_seg[0]
    t_upsample = np.linspace(t[0],t[-1],3*len(t))
    for ti in range(T):
        x = centerline[:,0,ti].astype(np.float64)
        y = centerline[:,1,ti].astype(np.float64)
        dx = x[1:]-x[:-1] # N-1
        dy = y[1:]-y[:-1] # N-1
        theta = np.arctan2(dy,dx) # angle to the lab frame
        theta = np.unwrap(theta)
        curv_t[:,ti] = (theta[1:] - theta[:-1])/Dx

    eps = 1e-3 # slightly smooth the derivative
    for xi in range(N-2):
        curv_xi_upsample = csaps(t,curv_t[xi,:],t_upsample,smooth=1-eps)
        dcurv_xi_upsample = np.gradient(curv_xi_upsample,t_upsample)
        dcurv_t[xi,:] = csaps(t_upsample,dcurv_xi_upsample,t,smooth=1-eps)

    return curv_t,dcurv_t

def centerline2angle_one(centerline):
    '''
    Compute the orientation of each segmeant 

    Parameter
    ---------
    Centerline: array of shape (N,2)
        Centerline of the Worm
    '''
    N = centerline.shape[0]
    dx = centerline[:-1,0] - centerline[1:,0]
    dy = centerline[:-1,1] - centerline[1:,1]
    angle = np.arctan2(dy,dx)
    angle = np.unwrap(angle)
    return angle

def centerline2angle(centerline):
    '''
    Compute the orientation of each segmeant
    ``
    Parameter
    ---------
    Centerline: array of shape (N,2,T)
        Centerline of the Worm

    Returns
    -------
    angle: array of shape (N-1,T)
        Orientation of each segment
    '''
    N = centerline.shape[0]
    T = centerline.shape[2]
    angle = np.zeros((N-1,T))
    for ti in range(T):
        angle[:,ti] = centerline2angle_one(centerline[:,:,ti])
    return angle

    

def interp_body_curv_dcurv(centerline,x_orig,x_targ,t_orig,t_targ,smooth=0.95):
    '''
    1, Interpolate the centerline in space. 
    2, calculate curvature and time derivative. 
    3, interpolate curvature and dcurv in time.    

    Parameters
    ----------
    centerline : array of shape (N,2,T)
        Centerline of the worm.
    x_orig : array of shape (N,)
        The body length coordinate of the centerline. e.g. np.linspace(0,1,100)
    x_targ : array of shape (N',)
        The body length coordinate of the centerline after interpolation. e.g. np.linspace(0,1,300)
    t_orig : array of shape (T,)
        Time stamps of the centerline.
    t_targ : array of shape (T,)
        Time stamps of the centerline after interpolation.
    smooth : float, optional
        Smoothing parameter for CSAPS. The default is 0.2.

    Returns
    -------
    curv_t_intp : array of shape (N'-2,T')
        Curvature of the worm after interpolation in space and time.
    dcurv_t_intp : array of shape (N'-2,T')
        Time derivative of curvature of the worm after interpolation in space and time.
    '''

    N = centerline.shape[0]
    T = centerline.shape[2]
    assert len(x_orig)==N
    assert len(t_orig)==T
    curv_t_intp = np.zeros((len(x_targ)-2,len(t_targ)))
    dcurv_t_intp = np.zeros((len(x_targ)-2,len(t_targ)))
    # 1,Interpolate the centerline in space using csaps
    centerline_interp = np.zeros((len(x_targ),2,T))
    for t in range(T):
        centerline_interp[:,:,t] = csaps(x_orig,centerline[:,:,t].T,x_targ,smooth=smooth).T
    # 2, calculate curvature and time derivative.
    curv_t,dcurv_t = body_curv_dcurv(centerline_interp,x_targ,t_orig) # (len(x_targ)-2,T)
    # 3, interpolate curvature and dcurv in time.
    for xi in range(len(x_targ)-2):
        curv_t_intp[xi,:] = csaps(t_orig,curv_t[xi,:],t_targ,smooth=1)
        dcurv_t_intp[xi,:] = csaps(t_orig,dcurv_t[xi,:],t_targ,smooth=1)
    return curv_t_intp,dcurv_t_intp

def interp_centerline(centerline,x_orig,x_targ,t_orig,t_targ,smooth=0.95):
    '''
    Interpolate the centerline in space and time. 

    Parameters
    ----------
    centerline : array of shape (N,2,T)
        Centerline of the worm.
    x_orig : array of shape (N,)
        The body length coordinate of the centerline. e.g. np.linspace(0,1,100)
    x_targ : array of shape (N',)
        The body length coordinate of the centerline after interpolation. e.g. np.linspace(0,1,300)
    t_orig : array of shape (T,)
        Time stamps of the centerline.
    t_targ : array of shape (T,)
        Time stamps of the centerline after interpolation.
    smooth : float, optional
        Smoothing parameter for CSAPS. The default is 0.2.

    Returns
    -------
    centerline_interp : array of shape (N',2,T')
        Centerline of the worm after interpolation in space and time.
    '''
    N = centerline.shape[0]
    T = centerline.shape[2]
    assert len(x_orig)==N
    assert len(t_orig)==T
    centerline_interp_space = np.zeros((len(x_targ),2,T))
    centerline_interp_spacetime = np.zeros((len(x_targ),2,len(t_targ)))
    for t in range(T):
        centerline_interp_space[:,:,t] = csaps(x_orig,centerline[:,:,t].T,x_targ,smooth=smooth).T
    for xi in range(len(x_targ)):
        centerline_interp_spacetime[xi,:,:] = csaps(t_orig,centerline_interp_space[xi,:,:],t_targ,smooth=1)
    return centerline_interp_spacetime

def centerline2curvature(centerline,smooth=0.99999,return_theta=False):
    '''
    Smooth the centerline witch csaps and calculate the curvature.
    
    Parameters
    ----------
    centerline : array of shape (N,2,T)
    
    Returns
    -------
    curv : array of shape (N,T)

    '''
    N = centerline.shape[0]
    T = centerline.shape[2]
    N_interp = 3*N
    centerline_interp = np.zeros((N_interp,2,T))
    curv_orig = np.zeros((N,T))
    x_orig = np.linspace(0,1,N)
    x_targ = np.linspace(0,1,N_interp)
    theta_orig = np.zeros((N,T))
    for t in range(T):
        centerline_interp[:,:,t] = csaps(x_orig,centerline[:,:,t].T,x_targ,smooth=smooth).T
        theta = centerline2angle_one(centerline_interp[:,:,t]) # angles of the interpolated centerline, N_interp-1
        curv_interp = (theta[1:]-theta[:-1])/(1/N_interp) # N_interp-2
        x_curv_interp = np.linspace(0,1,N_interp-2)
        x_curv_orig = np.linspace(0,1,N)
        curv_orig[:,t] = csaps(x_curv_interp,curv_interp,x_curv_orig,smooth=1)
        if return_theta:
            theta_orig[:,t] = csaps(np.linspace(0,1,N_interp-1),theta,np.linspace(0,1,N),smooth=1)

        # Use linear interpolation instead of csaps
        # curv_orig[:,t] = np.interp(x_curv_orig,x_curv_interp,curv_interp)
    if return_theta:
        return curv_orig,theta_orig
    else:
        return curv_orig
    
def boundary2centerline(boundary_A,boundary_B,direct_mean=False):
    '''
    Take the two boundaries of the worm to estimate the centerline.

    Parameters
    ----------
    boundary_A : array of shape (N,2,T)
        Boundary A of the worm.
    boundary_B : array of shape (N,2,T)
        Boundary B of the worm.
    
    Returns
    -------
    centerline : array of shape (N,2,T)
        Centerline of the worm.
    '''
    # csaps spine fit to boundary A and B
    N = boundary_A.shape[0]
    T = boundary_A.shape[2]
    # Transform to float64 if not
    if boundary_A.dtype != np.float64:
        boundary_A = boundary_A.astype(np.float64)
    if boundary_B.dtype != np.float64:
        boundary_B = boundary_B.astype(np.float64)
    
    if not direct_mean:
        centerline = np.zeros((N,2,T))
        for t in range(T):
            try:
                boundary_A_spine = csaps(np.linspace(0,1,N),boundary_A[:,:,t].T,np.linspace(0,1,N),smooth=0.99999).T
                boundary_B_spine = csaps(np.linspace(0,1,N),boundary_B[:,:,t].T,np.linspace(0,1,N),smooth=0.99999).T
                centerline_i = (boundary_A_spine + boundary_B_spine)/2
                # centerline_i = (boundary_A[:,:,t] + boundary_B[:,:,t])/2
                # centerline[:,:,t] = centerline_i
                # make centerline evenly spaced
                ds = np.sqrt(np.sum((centerline_i[1:,:]-centerline_i[:-1,:])**2,axis=1))
                s = np.cumsum(np.hstack((0,ds))) # s is the arclength
                centerline[:,:,t] = csaps(s/s[-1],centerline_i.T,np.linspace(0,1,N),smooth=0.99999).T
            except:
                centerline_i = (boundary_A[:,:,t] + boundary_B[:,:,t])/2
                ds = np.sqrt(np.sum((centerline_i[1:,:]-centerline_i[:-1,:])**2,axis=1))
                s = np.cumsum(np.hstack((0,ds))) # s is the arclength
                centerline[:,:,t] = csaps(s/s[-1],centerline_i.T,np.linspace(0,1,N),smooth=0.99999).T
        return centerline
    else:
        centerline = (boundary_A + boundary_B)/2
        return centerline

############################################################################################################
# Functions for eigenworms and Takens embedding
def proj_embedding(curvature,eigenworm,M):
    '''
    Project the curvature onto the eigenworms and take Takens embedding.
    
    Parameters
    ----------
    curvature : array of shape (T,N)
        Curvature of the worm.
    eigenworm : array of shape (N,K)
        Eigenworms basis
    M : int
        Embedding dimension.
    
    Returns
    -------
    embedding : array of shape (T-M+1,M*K)
        Takens embedding of the curvature.
    '''
    T = curvature.shape[0]
    N = curvature.shape[1]
    K = eigenworm.shape[1]
    embedding = np.zeros((T-M+1,M*K))
    for i in range(M):
        embedding[:,i*K:(i+1)*K] = curvature[i:T-M+i+1,:]@eigenworm
    return embedding

def embedding(proj,M):
    '''
    Take Takens embedding.
    
    Parameters
    ----------
    proj : array of shape (T,N)
        Projection of the curvature onto the eigenworms.
    M : int
        Embedding dimension.
    
    Returns
    -------
    embedding : array of shape (T-M+1,M*N)
        Takens embedding of the curvature.
    '''
    T = proj.shape[0]
    N = proj.shape[1]
    embedding = np.zeros((T-M+1,M*N))
    for i in range(M):
        embedding[:,i*N:(i+1)*N] = proj[i:T-M+i+1,:]
    return embedding

def delay_embedding(s,M,tau):
    '''
    Delay embedding of the time series and PCA.
    
    Parameters
    ----------
    s : array of shape (T,N)
        Time series.
    M : int
        Delay M times
    tau : int
        Delay interval.
    
    Returns
    -------
    embedding : array of shape (T-M*tau,(M+1)*N)
        Delay embedding of the time series.
    '''
    if len(s.shape)==1:
        s = s.reshape(-1,1)
    T = s.shape[0]
    N = s.shape[1] 
    embedding = np.zeros((T-M*tau,(M+1)*N))
    for i in range(M+1):
        embedding[:,i*N:(i+1)*N] = s[i*tau:T-M*tau+i*tau,:]
    return embedding

def PCA(x):
    '''
    PCA of the time series.

    Parameters
    ----------
    x : array of shape (T,N)
        Time series.

    Returns
    -------
    x_pca : array of shape (T,N)
        PCA of the time series.
    
    V : array of shape (N,N)
        Eigenvectors of the covariance matrix. Each column is an eigenvector.
    '''
    T = x.shape[0]
    N = x.shape[1]
    # Center the data
    x_centered = x - np.mean(x,axis=0)
    # Calculate the covariance matrix
    C = x_centered.T@x_centered/(T-1)
    # Calculate the eigenvalues and eigenvectors
    w, V = np.linalg.eig(C)
    # Sort the eigenvalues and eigenvectors
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:,idx]
    # Project the data onto the eigenvectors
    x_pca = x_centered@V
    return x_pca, V


def calc_amplitude(kt):
    '''
    Calculate the amplitude of the head curvature.
    Find the peaks with significant prominence and calculate the average hight of the peaks.
    Parameters
    ----------
    kt : array of shape (T,)
        Curvature of the head.

    Returns
    -------
    amplitude : float
        Amplitude of the head curvature.
    '''
    # smooth the head curvature
    kt_sm = np.convolve(kt,np.ones(7)/7,mode='same')
    # find positive peaks
    k_min = np.min(kt_sm)
    k_max = np.max(kt_sm)
    reference = (k_max-k_min)/2
    pos_peaks, _ = find_peaks(kt_sm,prominence=reference*0.8,height=reference*0.6)
    # find negative peaks
    neg_peaks, _ = find_peaks(-kt_sm,prominence=reference*0.8,height=reference*0.6)
    # Calculate the average height of the peaks
    if len(pos_peaks) == 0 and len(neg_peaks) == 0:
        return None
    
    peak_ave = np.mean(np.concatenate((kt_sm[pos_peaks],-kt_sm[neg_peaks])))
    # Calculate the amplitude

    return peak_ave

def peak_amplitude_distribution(s_ls,smooth_l=7,return_mean_std=True):
    L = len(s_ls)
    amp_ls = []
    for i in range(L):
        kt = s_ls[i]
        # smooth the head curvature
        kt_sm = np.convolve(kt,np.ones(smooth_l)/smooth_l,mode='same')
        # find positive peaks
        k_min = np.min(kt_sm)
        k_max = np.max(kt_sm)
        reference = (k_max-k_min)/2
        pos_peaks, _ = find_peaks(kt_sm,prominence=reference*0.7,distance=100)
        # find negative peaks
        neg_peaks, _ = find_peaks(-kt_sm,prominence=reference*0.7,distance=100)
        for j in range(len(pos_peaks)):
            amp_ls.append(kt_sm[pos_peaks[j]])
        for j in range(len(neg_peaks)):
            amp_ls.append(-kt_sm[neg_peaks[j]])
    amp_ls = np.array(amp_ls)
    if return_mean_std:
        return np.mean(amp_ls),np.std(amp_ls)
    else:
        return amp_ls

################### Find excursions which are defined the as local retrograding peaks ###################
def find_excursion(kt,smooth_l = 7,peak_width=7,max_excursion = 0.5,min_excursion = 0.0):
    kt_sm = np.convolve(kt,np.ones(smooth_l)/smooth_l,mode='same')
    # find positive peaks
    pos_peaks, _ = find_peaks(kt_sm,width=peak_width)
    # find negative peaks
    neg_peaks, _ = find_peaks(-kt_sm,width=peak_width)
    # merge positive and negative peaks
    signs_pos = np.ones(len(pos_peaks))
    signs_neg = -np.ones(len(neg_peaks))
    prom_pos = peak_prominences(kt_sm,pos_peaks)[0]
    prom_neg = peak_prominences(-kt_sm,neg_peaks)[0]
    signs = np.concatenate((signs_pos,signs_neg))
    peaks = np.concatenate((pos_peaks,neg_peaks))
    prom = np.concatenate((prom_pos,prom_neg))
    y_min = np.min(kt_sm)
    y_max = np.max(kt_sm)
    ave_amp = calc_amplitude(kt_sm)
    # sort peaks by time
    sort_ind = np.argsort(peaks)
    peaks = peaks[sort_ind]
    signs = signs[sort_ind]
    prom = prom[sort_ind]

    # Scan out excursions that obey following criteria:
    # 1. Postive excursion should follow +1 -1 +1 3 peaks pattern and negative excursion should follow -1 +1 -1 3 peaks pattern
    # 2. The first peak in the 3 peaks pattern should be larger than the last peak
    # 3. The second peak in the 3 peaks pattern should have similar width as the last peak
    # 4. The second peak and last peak should be close, decided by a threshold of max_excursion_width
    # 5. The second peak and last peak should have similar prominence.

    L = len(peaks)
    excursion_start_end = []
    max_excursion_width = 100
    # max_excursion_height = max_excursion*(y_max-y_min)
    # min_excursion_height = min_excursion*(y_max-y_min)
    max_excursion_height = max_excursion*ave_amp*2 # 2 is to be consistent with old analysis
    min_excursion_height = min_excursion*ave_amp*2
    for i in range(L-3):
        sign_i = signs[i]
        if signs[i+1]!= -1*sign_i or signs[i+2]!=sign_i:
            continue
        if sign_i == 1:
            if kt_sm[peaks[i]] < kt_sm[peaks[i+2]]:
                continue
        else:
            if kt_sm[peaks[i]] > kt_sm[peaks[i+2]]:
                continue
        if peaks[i+2]-peaks[i+1] > max_excursion_width:
            continue
        if abs(kt_sm[peaks[i+1]]-kt_sm[peaks[i+2]]) > max_excursion_height:
            continue
        if abs(kt_sm[peaks[i+1]]-kt_sm[peaks[i+2]]) < min_excursion_height:
            continue
        prom_diff = abs(prom[i+1]-prom[i+2])/min(prom[i+1],prom[i+2])
        if prom_diff > 0.1:
            continue 
        excursion = {'sign':sign_i,'pre_peak':peaks[i],'start':peaks[i+1],'end':peaks[i+2]}
        excursion_start_end.append(excursion)
    return excursion_start_end

################ Find excursion by sign match of time derivative between lf(VMD) and denoised signal
# 1. Find zero crossings of the derivative of the denoised signal, return the indices before the zero crossings
# 2. For loop of zero crossings; if not excursion start zeros crossing,continue; find next zero crossing, if excursion valid, append excursion to list else continue
# 3. Return excursion list

# What is a valid excursion?
# 1. The excursion is not too short
# 2. The excursion start with opposite sign and end with same sign
# 3. If start and end with the same sign, compare distance to lf zeros crossing.

def find_lf_denoised(head_curv):
    imfs,_,_ = VMD(head_curv, 600, 0, 10, 0, 1, 1e-7)
    stds = np.std(imfs,axis=1)
    lf = imfs[0:np.argmax(stds)+1,:].sum(axis=0)
    # if stds[0] > 2*stds[1]:
    #     lf = imfs[0,:]
    # else:
    #     lf = imfs[0:2,:].sum(axis=0)
    denoised = imfs[0:6,:].sum(axis=0)
    if len(denoised)<len(head_curv):
        denoised = np.pad(denoised,(0,len(head_curv)-len(denoised)),'edge')
        lf = np.pad(lf,(0,len(head_curv)-len(lf)),'edge')
    lf_phase = np.angle(scipy.signal.hilbert(lf))
    return lf,denoised,lf_phase

    
def find_zero_crossing_indices(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return zero_crossings

# Upsample denoised signal and lf signal by 10 via interpolation
def upsample(signal,upsample_factor=10):
    x = np.arange(0,len(signal))
    x_upsample = np.linspace(0,len(signal)-1,len(signal)*upsample_factor)
    f = scipy.interpolate.interp1d(x,signal)
    signal_upsample = f(x_upsample)
    return signal_upsample

def find_excursions_sign_match(signal,min_excursion_time=0,max_excursion_time=70,more_output=False):
    lf,denoised,phase_t = find_lf_denoised(signal)
    upsample_factor = 10
    lf_us = upsample(lf,upsample_factor=upsample_factor)
    denoised_us = upsample(denoised,upsample_factor=upsample_factor)
    dlf_us = np.gradient(lf_us,0.02/upsample_factor)
    dcurv_us = np.gradient(denoised_us,0.02/upsample_factor)
    # plt.figure(dpi=300,figsize=(12,5))
    # plt.plot(dlf_us,label='dlf')
    # plt.plot(dcurv_us,label='dcurv')
    # plt.hlines(0, xmin=0, xmax=len(dcurv_us), color='k', linestyle='--',linewidth=1)
    # plt.legend()
    # plt.show()

    zc_all = find_zero_crossing_indices(dcurv_us)
    zc_lf = find_zero_crossing_indices(dlf_us)
    dlf_sign = np.sign(dlf_us)
    dcurv_sign = np.sign(dcurv_us)
    excursion_ls = []
    zc_used = False

    for i,zc in enumerate(zc_all[:-2]):
        if zc_used:
            zc_used = False
            continue
        if dcurv_sign[zc+1] == dlf_sign[zc+1]: 
            continue
        if dcurv_sign[zc_all[i+1]+1] == dlf_sign[zc_all[i+1]+1]: 
            # There should only be at most dlf zero crossing between two dcurv zero crossing  
            if len(zc_lf[(zc_lf>zc) & (zc_lf<zc_all[i+1])]) > 1:
                continue
            # If the excursion is not too short and not too long, and the following peak is not too 'shallow', append it to the list
            if zc_all[i+1] - zc > min_excursion_time*upsample_factor and zc_all[i+1] - zc < max_excursion_time*upsample_factor and zc_all[i+2]-zc_all[i+1] > min_excursion_time*upsample_factor:
                excursion = {'start':zc//upsample_factor,'end':zc_all[i+1]//upsample_factor ,'sign':-dlf_sign[zc]}
                excursion_ls.append(excursion)
                zc_used = True
            continue
        # Find the zeros crossing of the lf signal that is between the two zero crossings of the denoised signal
        lf_zc = zc_lf[(zc_lf>zc) & (zc_lf<zc_all[i+1])]
        if len(lf_zc) == 0:
            continue
        lf_zc = lf_zc[0]
        # Compare which zero crossing is closer to the peak of low frequency signal
        if i == zc_all.shape[0]-2:
            continue
        if lf_zc - zc > zc_all[i+2] - lf_zc:
            if zc_all[i+1] - zc > min_excursion_time*upsample_factor and zc_all[i+1] - zc < max_excursion_time*upsample_factor and zc_all[i+2]-zc_all[i+1] > min_excursion_time*upsample_factor:
                excursion = {'start':zc//upsample_factor,'end':zc_all[i+1]//upsample_factor ,'sign':-dlf_sign[zc]}
                excursion_ls.append(excursion)
                zc_used = True

    # plt.figure(dpi=300,figsize=(12,5))
    # plt.plot(denoised,label='denoised')
    # for excursion in excursion_ls:
    #     color = 'r' if excursion['sign'] == 1 else 'b'
    #     plt.axvspan(excursion['start'],excursion['end'],alpha=0.3,color=color)
    # plt.legend()
    # plt.show()
    if more_output:
        return excursion_ls,[lf,denoised,phase_t,lf_us,denoised_us,dlf_us,dcurv_us]
    return excursion_ls

######################################################
def align_trajectory(source,target):
    '''
    Find the starting point of the target trajectory that is closest to the source trajectory.
    Parameters
    ----------
    source : 1D array of shape (T,)
        Source trajectory.
    target : 1D array of shape (M,)
        Target trajectory.

    '''
    corr = scipy.signal.correlate(source,target,mode='valid')
    start_ind = np.argmax(abs(corr))
    return start_ind

def local_delay_period(k1,k2,t=None,dt=0.02):
    assert len(k1)==len(k2)
    if t is None:
        t = np.arange(len(k1))*dt
    else:
        assert len(t)==len(k1)
    try:
        phase = np.angle(scipy.signal.hilbert(k2))
        phase_unwrap = np.unwrap(phase)
        f_time = interp1d(phase_unwrap,t)
    except:
        print('smoothing k2')
        k2_smooth = np.convolve(k2,np.ones(10)/10,mode='same')
        phase = np.angle(scipy.signal.hilbert(k2_smooth))
        phase_unwrap = np.unwrap(phase)
        f_time = interp1d(phase_unwrap,t)
    window = np.pi*3
    # delay_t = np.zeros(len(k1))
    # period_t = np.zeros(len(k1))
    n_period = int(phase_unwrap.max()/(2*np.pi))+1
    delays = np.zeros(n_period)
    periods = np.zeros(n_period)
    sample_ind = np.zeros((n_period,2),dtype=int)

    for i in range(n_period):
        # determine the start phase and the end phase of the window
        if i*2*np.pi - window/2 < phase_unwrap.min():
            start = phase_unwrap.min()
            end = start + window
        elif i*2*np.pi + window/2 > phase_unwrap.max():
            end = phase_unwrap.max()
            start = end - window
        else:
            start = i*2*np.pi - window/2
            end = i*2*np.pi + window/2
        start_t = f_time(start)
        start_ind = np.argmin(abs(t-start_t))
        end_t = f_time(end)
        end_ind = np.argmin(abs(t-end_t))
        periods[i] = (end_t - start_t)*2*np.pi/window
        autocorr = scipy.signal.correlate(k2[start_ind:end_ind],k1[start_ind:end_ind],mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peak = np.argmax(autocorr)
        delays[i] = t[start_ind+peak] - t[start_ind]
        sample_ind[i,0] = start_ind
        sample_ind[i,1] = end_ind
    return sample_ind,delays,periods

def sliding_window_delay(k1,k2,t=None,dt=0.02,step=20,windos_len=200):
    assert len(k1)==len(k2)
    if t is None:
        t = np.arange(len(k1))*dt
    else:
        assert len(t)==len(k1)
    sample_center = np.arange(0,len(k1),step)
    start_ind = np.maximum(sample_center-windos_len//2,0)
    end_ind = np.minimum(sample_center+windos_len//2,len(k1))
    delays = np.zeros(len(sample_center))
    for i in range(len(sample_center)):
        autocorr = scipy.signal.correlate(k2[start_ind[i]:end_ind[i]],k1[start_ind[i]:end_ind[i]],mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peak = np.argmax(autocorr)
        delays[i] = t[start_ind[i]+peak] - t[start_ind[i]]
    return t[sample_center],delays

    


##### Hilbert-Huang transform to extract phase#####
def HH_phase(s,p=50,shift=1):
    '''
    Calculate the instantaneous phase of the quasi-oscillatory signal vis Hilbert-Huang transform.
    
    Parameters
    ----------
    s : array of shape (T,)
        Quasi-oscillatory signal.
    
    p : int
        Modes with average period below 'p' will be subtracted from the signal before Hilbert transform.

    Returns
    -------
    phase : array of shape (T,)
        Instantaneous phase.
    '''
    # Calculate the empirical mode decomposition
    emd = EMD()
    imfs = emd(s)
    # sm_func = lambda x: np.convolve(x,np.ones(sm)/sm,mode='same')
    # period = [mean_peak_distance(imfs[i,:]) for i in range(imfs.shape[0])]
    period = [max_peak_distance(imfs[i,:]) for i in range(imfs.shape[0]-1)]
    # Find the modes with period below p
    n = np.sum(np.array(period)<p)
    print('truncating the first {} modes'.format(n))
    # Take the useful modes, in the head curvature case, the high frequency modes (imf 0,1,2,3) are discarded
    ss = imfs[n:,:].sum(axis=0)
    # hilbert transform
    analytic_signal = scipy.signal.hilbert(ss)
    if np.abs(shift) == 1:
        analytic_signal = analytic_signal*shift
    else:
        raise ValueError('Mode of shift should be 1 ')
    phase =np.angle(analytic_signal)
    return phase

def pad_convolve_smooth(s,sm=7):
    '''
    Pad the signal and then convolve with a smoothing kernel.
    '''
    s_pad = np.pad(s,(sm,sm),'edge')
    sm_func = lambda x: np.convolve(x,np.ones(sm)/sm,mode='same')
    return sm_func(s_pad)[sm:-sm]

def mean_peak_distance(s):
    '''
    The mean distance between peaks in the quasi-oscillatory signal.
    '''
    peaks = scipy.signal.find_peaks(s)[0]
    if len(peaks)<2:
        return 1e8
    else:
        return np.mean(np.diff(peaks))

def max_peak_distance(s):
    '''
    The max distance between peaks in the quasi-oscillatory signal.
    '''
    peaks = scipy.signal.find_peaks(s)[0]
    if len(peaks)<2:
        return 1e8
    else:
        return (np.diff(peaks)).max()

def plot_imf(s):
    emd = EMD()
    imfs = emd(s)
    std = np.std(s)
    plt.figure(dpi=300)
    for i in range(imfs.shape[0]):
        plt.plot(imfs[i,:]+i*std*2)
    plt.show()


###### Find zero-crossing points of the signal
def zero_crossing_index(curvature,p=50):
    '''
    Identify the positive-to-negative zero-crossing points of the curvature
    and the negative-to-positive zero-crossing points of the curvature
    
    Parameters
    ----------
    curvature : numpy.ndarray of shape (T,)
        The curvature time series
    '''

    phase = HH_phase(curvature,p=p)
    n90 = -np.pi/2
    phase_n90 = scipy.signal.find_peaks(-abs(phase-n90),width=50)[0]
    n2p_zc = find_zero_crossing_index(curvature,phase_n90) # negative to positive zero-crossing points with -pi/2 phase as reference
    p90 = np.pi/2
    phase_p90 = scipy.signal.find_peaks(-abs(phase-p90),width=50)[0]
    p2n_zc = find_zero_crossing_index(curvature,phase_p90) # positive to negative zero-crossing points with pi/2 phase as reference
    return n2p_zc,p2n_zc

def find_zero_crossing_index(sig,seed,srange=20):
    '''
    find the zero-crossing points near the seed points

    Parameters
    ----------
    sig : numpy.ndarray of shape (T,)
        The signal time series

    seed : numpy.ndarray of shape (N,)
        N seed points
    '''
    zero_crossing_index = []
    for i in range(len(seed)):
        seed_i = seed[i]
        sig_i = sig[seed_i-srange:seed_i+srange]
        zero_crossing_index_i = np.where(np.diff(np.sign(sig_i)))[0]
        if len(zero_crossing_index_i) == 0:
            continue
        zero_crossing_index_i = zero_crossing_index_i[np.argmin(abs(zero_crossing_index_i-srange))]
        zero_crossing_index_i = zero_crossing_index_i + seed_i - srange
        zero_crossing_index.append(zero_crossing_index_i)
    return np.array(zero_crossing_index)

def phase_polar_hist(s,phase,n,title=None):
    '''
    Plot the distribution of variable s over the phase space.
    Plot in polar coordinates.

    Parameters
    ----------
    s : array of shape (T,)
        The variable of interest.

    phase : array of shape (T,)
        -pi to pi

    n : int
        Number of bins.
    
    '''
    bins = np.linspace(-np.pi,np.pi,n+1)[1:]
    plot_bins = np.linspace(-np.pi,np.pi,n+1)[:-1]
    s_mean = np.zeros((n,))
    phase_dig = np.digitize(phase,bins)
    for i in range(n):
        s_mean[i] = s[phase_dig==i].mean()
    plt.figure(dpi=300)
    ax = plt.subplot(111, polar=True)
    ax.bar(plot_bins, s_mean, width=2*np.pi/n, bottom=0.0)
    if title is not None:
        plt.title(title)
    plt.show()

def phase_polar_hist_ax(ax,s,phase,n,title):
    '''
    Plot the distribution of variable s over the phase space.
    Plot in polar coordinates.

    Parameters
    ----------
    s : array of shape (T,)
        The variable of interest.

    phase : array of shape (T,)
        -pi to pi

    n : int
        Number of bins.
    
    '''
    bins = np.linspace(-np.pi,np.pi,n+1)[1:]
    plot_bins = np.linspace(-np.pi,np.pi,n+1)[:-1]
    s_mean = np.zeros((n,))
    phase_dig = np.digitize(phase,bins)
    for i in range(n):
        s_mean[i] = s[phase_dig==i].mean()
    ax.bar(plot_bins, s_mean, width=2*np.pi/n, bottom=0.0)
    if title is not None:
        ax.set_title(title)
    

def prob_x_cond_y(data, xbin, ybin):
    # Extract x and y values from data
    x_values, y_values = zip(*data)
    
    # Calculate the range of x and y values
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Calculate the bin widths
    x_width = (x_max - x_min) / xbin
    y_width = (y_max - y_min) / ybin
    
    # Create empty 2D array to store counts
    counts = np.zeros((ybin, xbin), dtype=int)
    
    # Iterate over data and increment counts
    for x, y in data:
        x_index = int((x - x_min) / x_width)
        y_index = int((y - y_min) / y_width)
        counts[y_index][x_index] += 1
    
    # Calculate probabilities
    x_cond_y = counts / np.sum(counts, axis=1, keepdims=True) # p(x|y) = p(x,y)/p(y)

    
    return x_cond_y

def plot_prob_x_cond_y(sig_x,sig_y,xbins,ybins,xclip=None,yclip=None):
    '''
    Plot the probability of x given y.
    '''
    if xclip is not None:
        sig_x = np.clip(sig_x,xclip[0],xclip[1])
    if yclip is not None:
        sig_y = np.clip(sig_y,yclip[0],yclip[1])
    data = zip(sig_x,sig_y)
    x_cond_y = prob_x_cond_y(data,xbins,ybins)
    plt.figure(dpi=300)
    plt.imshow(x_cond_y.T)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

def smooth_data(s,window):
    '''
    Smooth the data with a window.
    '''
    return np.convolve(np.pad(s,(window,window),'edge'),np.ones(window*2+1)/(window*2+1),mode='same')[window:-window]   


#%%    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import resforce
    from ipywidgets import interact, interactive, fixed, interact_manual
    t = np.linspace(0,10,1000)
    x = np.linspace(0,1,100)
    t_grid,x_grid = np.meshgrid(t,x)
    f = 0.5
    lamb = 0.6
    AMP = 7
    y = AMP*np.sin(2*np.pi*f*t_grid + 2*np.pi*lamb*x_grid)
    dy = np.gradient(y,t,axis=1)
    y_noise = y + np.random.randn(*y.shape)*0.1*AMP
    dy_noise = csaps_derivative(t,y_noise,axis=1,smooth=0.98)
    rft = resforce.ResisForce()
    Dt = t[1]-t[0]
    Dx = x[1]-x[0]
    centerline = rft.curv2centerline(y,dy,Dt,Dx)
    x_min = np.min(centerline[:,0,:])
    x_max = np.max(centerline[:,0,:])
    y_min = np.min(centerline[:,1,:])
    y_max = np.max(centerline[:,1,:])
    x_range = [x_min, x_min + max(x_max-x_min,y_max-y_min)]
    y_range = [y_min, y_min + max(x_max-x_min,y_max-y_min)]


#%%
if __name__ == '__main__':
    def plot_centerline(i):
        plt.figure(dpi=300,figsize=(5,5))
        plt.plot(centerline[:,0,i],centerline[:,1,i])
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.show()
    interact(plot_centerline,i=(0,centerline.shape[2]-1,1))

#%%
if __name__ == '__main__':
    curv_t,dcurv_t = body_curv_dcurv(centerline,x,t)
    plt.figure(dpi=300,figsize=(5,5))
    plt.plot(curv_t[0,:])
    plt.plot(y[0,:])
    plt.show()

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N2data = scipy.io.loadmat('data/N2_centerline_old.mat')
    N2_centerline = N2data['centerline_data']
    N2_curvature = N2data['curvature_data']
    ind = 0
    curvature = N2_curvature[ind,0]
    centerline = N2_centerline[ind,0]
    curvature_calc = centerline2curvature(centerline)
 #%%   
if __name__ == '__main__':
    plt.figure(dpi=300,figsize=(5,5))
    plt.plot(-curvature[:,:15].mean(axis=1)*100,label='processed data')
    plt.plot(curvature_calc[:15,:].mean(axis=0),label='my calculation')
    plt.legend()
    plt.show()








# %%
# Plot the interpolated centerline and original centerline
if __name__ == '__main__':
    N2data = scipy.io.loadmat('../WormSim/excursion/N2_centerline_old.mat')
    N2_centerline = N2data['centerline_data']
    N2_curvature = N2data['curvature_data']
    ind = 0
    curvature = N2_curvature[ind,0] 
    centerline = N2_centerline[ind,0].astype(np.float64)/20  # N*2*T
    def plot_centerline_interp(t):
        N = centerline.shape[0]
        centerline.shape[0]
        T = centerline.shape[2]
        N_interp = 3*N
        x_orig = np.linspace(0,1,N)
        x_targ = np.linspace(0,1,N_interp)
        centerline_interp_t = csaps(x_orig,centerline[:,:,t].T,x_targ,smooth=0.99999).T
        plt.figure(dpi=300,figsize=(5,5))
        plt.plot(centerline_interp_t[:,0],centerline_interp_t[:,1],label='interp',linewidth=1)
        plt.plot(centerline[:,0,t],centerline[:,1,t]+2,label='orig',alpha=0.5,linewidth=1)
        plt.legend()
        plt.show()
    interact(plot_centerline_interp,t=(0,centerline.shape[2]-1,1))
# %%
# Example of converting centerline to curvature
if __name__ == '__main__':
    name = 'N2'
    centerline_data = scipy.io.loadmat('../neuron_ablations/centerline/{}.mat'.format(name))
    centerline_data = centerline_data[name+'_centerline_data']
    curvature_data_ls = []
    for i in range(centerline_data.shape[0]):
        print(i)
        curvature = centerline2curvature(centerline_data[i,0])
        curvature_data_ls.append(curvature)


    curvature_data = np.empty((len(curvature_data_ls),1),dtype=object)
    for i in range(len(curvature_data_ls)):
        curvature_data[i,0] = curvature_data_ls[i]
    np.save('../neuron_ablations/curvature/{}.npy'.format(name),curvature_data)

#%%
if __name__ == '__main__':
    N2_w6_boundary = scipy.io.loadmat('../WormSim/neuron_ablations/rawdata/20190811_1706_w6_Boundary.mat')
    N2_w6_boundary = scipy.io.loadmat('../neuron_ablations/rawdata/20190811_1706_w6_Boundary.mat')
    BoundaryA = N2_w6_boundary['BoundaryA'].astype(np.float64)
    BoundaryB = N2_w6_boundary['BoundaryB'].astype(np.float64)
    centerline = N2_w6_boundary['centerline'].astype(np.float64)
    # Swap axis 0 and 2
    BoundaryA = np.swapaxes(BoundaryA,0,2)
    BoundaryB = np.swapaxes(BoundaryB,0,2)
    centerline = np.swapaxes(centerline,0,2)
    WormLength = np.sqrt((centerline[:,0,0]-centerline[:,0,-1])**2+(centerline[:,1,0]-centerline[:,1,-1])**2).mean()
    i = 0
    plt.plot(centerline[i,0,:],centerline[i,1,:])
    centerline_csaps = csaps(np.linspace(0,1,centerline.shape[0]),centerline[i,:,:].T,smooth=0.99999)

# %% Check how calc_amplitude works
if __name__ == '__main__':
    def plot_amp(i):
        data = scipy.io.loadmat('../WormSim/neuron_ablations/N2.mat')['N2_hb_dynamics']
        kt = data[i,0][:,1]
        kt_sm = np.convolve(kt,np.ones(7)/7,mode='same')
        # find positive peaks
        k_min = np.min(kt_sm)
        k_max = np.max(kt_sm)
        reference = (k_max-k_min)/2
        pos_peaks, _ = find_peaks(kt_sm,prominence=reference*1.2,height=reference*0.4)
        # find negative peaks
        neg_peaks, _ = find_peaks(-kt_sm,prominence=reference*1.2,height=reference*0.4)
        # Calculate the average height of the peaks
        peak_ave = np.mean(np.concatenate((kt_sm[pos_peaks],-kt_sm[neg_peaks])))
        plt.figure(dpi=300,figsize=(12,5))
        plt.plot(kt_sm)
        plt.vlines(pos_peaks,k_min,k_max,color='r')
        plt.vlines(neg_peaks,k_min,k_max,color='b')
        plt.hlines([-peak_ave,0,peak_ave],xmin=0,xmax=len(kt_sm),color='k',linestyle='--')
        amp_old = calc_amplitude(kt_sm)
        plt.hlines([-amp_old,amp_old],xmin=0,xmax=len(kt_sm),color='g',linestyle='--')
        plt.show()
    interact(plot_amp,i=(0,100,1))
    ## It seems that height requirement is necessary to avoid false peaks

