#%%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def extract_phase_scale(sig,scale,omega=3,plot=False):
    '''
    Extract phase from a signal using wavelet transform using a fixed scale.
    
    Parameters
    ----------
    sig : 1d array
        Signal to be analyzed.
    scale : int
        Scale of the wavelet transform.
    omega : int
        Omega of the wavelet transform.

    Returns
    -------
    phase_t : 1d array
        Phase of the signal.
    amp_t : 1d array
    '''
    wt = signal.cwt(sig, signal.morlet2, [scale,],w=omega)
    phase = np.angle(wt[0])
    phase = np.mod(phase+3/2*np.pi,2*np.pi) -np.pi
    if not plot:
        return phase,wt[0]
    else:
        plt.figure(figsize=(13,6))
        plt.plot(sig/np.max(np.abs(sig)))
        plt.plot(np.real(wt[0])/np.real(wt[0]).max())
        plt.plot(phase/np.pi)
        plt.show()
        return phase,wt[0]

def extract_phase(sig,omega=3,scale_range=np.arange(10,100),plot=False,mark_phase=0):
    '''
    Extract phase from a signal using wavelet transform. The scale is determined by the maximum power.

    Parameters
    ----------
    sig : 1d array
        Signal to be analyzed.
    omega : int
        Omega of the wavelet transform.
    scale_min : int 
        Minimum scale of the wavelet transform.
    scale_max : int
        Maximum scale of the wavelet transform.
    mark_phase : float
        Phase to be marked in horizontal lines in the plot.
    Returns
    -------
    phase_t : 1d array
        Phase of the signal.
    amp_t : 1d array
    '''
    wt = signal.cwt(sig, signal.morlet2, scale_range,w=omega)
    wt_abs = np.abs(wt)
    phase_t = np.zeros_like(sig)
    amp_t = np.zeros_like(sig)
    for i in range(len(sig)):
        phase_t[i] = np.angle(wt[np.argmax(wt_abs[:,i]),i])
        amp_t[i] = np.max(wt_abs[:,i])
    if not plot:
        return phase_t,amp_t
    else:
        plt.figure(figsize=(13,6),dpi=300)
        plt.plot(sig/np.max(np.abs(sig)))
        plt.plot(phase_t/np.pi)
        phase0_ind = np.where(np.abs(phase_t-mark_phase)<0.1)[0]
        plt.vlines(phase0_ind,-1,1,'r',alpha=0.5)
        plt.show()
#%%
if __name__ == '__main__':
    import scipy.io
    N2data = scipy.io.loadmat('data/N2.mat')
    sig = N2data['N2_hb_dynamics'][71,0][:,1]
    sig_sm = np.convolve(sig,np.ones(7)/7,mode='same') # Smooth the signal
    phase,wt0 = extract_phase(sig_sm,50,omega=3,plot=True)
# %%
if __name__ == '__main__':
    scale = np.linspace(50,100,51)
    wt = signal.cwt(sig, signal.morlet2, scale,w=3)
    wt_power = np.abs(wt)
    max_power_ind = np.argmax(wt_power,axis=0)
    contour_sig = wt[max_power_ind,np.arange(wt.shape[1])]
    plt.figure(dpi=300)
    plt.imshow(wt_power,cmap='coolwarm')
    plt.colorbar()
    plt.show()
    plt.plot(np.real(contour_sig)/np.real(contour_sig).max())
    plt.plot(sig/sig.max())
    plt.plot(np.angle(contour_sig)/np.pi)
    plt.show()
# %%
