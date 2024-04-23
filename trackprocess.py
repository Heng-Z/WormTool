#%%
import numpy as np
import scipy.io
import WormTool
from tqdm import tqdm
from csaps import csaps
from ipywidgets import interact,fixed
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
import logging
from WormTool.motionstate import MotionState

logging.basicConfig(level=logging.INFO,format='%(name)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normal_boundary(boundaryA,boundaryB,verbose=False):
    # boundaryA and boundaryB are 2D arrays with shape (N,2)
    dist = np.linalg.norm(boundaryA-boundaryB,axis=1)
    dist_top50 = np.sort(dist)[int(0.7*len(dist)):]
    dist_mean = dist_top50.mean()
    dist_std = dist_top50.std()
    boundaryA_length = np.linalg.norm(np.diff(boundaryA,axis=0),axis=1).sum()
    boundaryB_length = np.linalg.norm(np.diff(boundaryB,axis=0),axis=1).sum()
    worm_length = (boundaryA_length+boundaryB_length)/2
    small_mean = dist_mean/worm_length < 0.15
    small_std = dist_std/worm_length < 0.2
    closed_head = np.linalg.norm(boundaryA[0]-boundaryB[0]) < 0.05*worm_length
    closed_tail = np.linalg.norm(boundaryA[-1]-boundaryB[-1]) < 0.07*worm_length
    if verbose:
        print('dist_mean:',dist_mean)
        print('dist_std:',dist_std)
        print('boundaryA_length:',boundaryA_length)
        print('boundaryB_length:',boundaryB_length)
        print('worm_length:',worm_length)
        print('small_mean:',small_mean)
        print('small_std:',small_std)
        print('closed_head:',closed_head)
        print('closed_tail:',closed_tail)
    return small_mean and small_std and closed_head and closed_tail

def sliding_speed(boundA,boundB,time):
    # boundA and boundB are boundaries of the worms, of shape 100x2xT
    # return the sliding speed of the worm
    # 
    # ds_A = np.sqrt(np.sum(np.diff(boundA,axis=2)**2,axis=1))
    # ds_B = np.sqrt(np.sum(np.diff(boundB,axis=2)**2,axis=1))
    # v_A = ds_A/np.diff(time) # (100,T-1)
    # v_B = ds_B/np.diff(time) # (100,T-1)
    # v_A = np.concatenate([v_A,v_A[:,-1]],axis=1)
    # v_B = np.concatenate([v_B,v_B[:,-1]],axis=1)
    diff_A = boundA[:,:,2:]-boundA[:,:,:-2]
    diff_B = boundB[:,:,2:]-boundB[:,:,:-2]
    ant2post_vec_A = boundA[0:-1,:,1:-1]-boundA[1:,:,1:-1]
    ant2post_vec_B = boundB[0:-1,:,1:-1]-boundB[1:,:,1:-1]
    sign_A = np.sign(np.sum(diff_A[1:]*ant2post_vec_A,axis=1).sum(axis=0,keepdims=True)) # (1,T-2)
    sign_B = np.sign(np.sum(diff_B[1:]*ant2post_vec_B,axis=1).sum(axis=0,keepdims=True)) # (1,T-2)
    sign_A = scipy.signal.medfilt(sign_A[0],kernel_size=5) 
    sign_B = scipy.signal.medfilt(sign_B[0],kernel_size=5)
    sign_AorB = np.sign(sign_A+sign_B+1e-6)
    ds_A = np.sqrt(np.sum(diff_A**2,axis=1))*sign_AorB # (100,T-2)
    ds_B = np.sqrt(np.sum(diff_B**2,axis=1))*sign_AorB # (100,T-2)
    v_A = ds_A/(time[2:] - time[:-2]).T 
    v_B = ds_B/(time[2:] - time[:-2]).T
    v_A = np.concatenate([v_A[:,0:1],v_A,v_A[:,-1:]],axis=1)
    v_B = np.concatenate([v_B[:,0:1],v_B,v_B[:,-1:]],axis=1)
    return v_A, v_B

def head_tail_reverse_correction(centerline_t,boundA_t,boundB_t,worm_len):
    # centerline_t is a 3D array of shape (N,2,T)
    diff = np.diff(centerline_t[0],axis=1)
    diff_norm = np.linalg.norm(diff,axis=0)/worm_len
    reverse = np.where(diff_norm > 0.5)[0]
    logger.debug('Reverse points:{}'.format(reverse))
    if len(reverse) ==1 and reverse[0] < centerline_t.shape[2]/2:
        # reverse back along the first dimension
        centerline_t[:,:,:reverse[0]+1] = centerline_t[::-1,:,:reverse[0]+1]
        boundA_before = boundA_t[::-1,:,:reverse[0]+1].copy()
        boundA_t[:,:,:reverse[0]+1] = boundB_t[::-1,:,:reverse[0]+1]
        boundB_t[:,:,:reverse[0]+1] = boundA_before
    if len(reverse) == 2 and reverse[1]-reverse[0] < centerline_t.shape[2]/2:
        centerline_t[:,:,reverse[0]+1:reverse[1]+1] = centerline_t[::-1,:,reverse[0]+1:reverse[1]+1]
        boundA_before = boundA_t[::-1,:,reverse[0]+1:reverse[1]+1].copy()
        boundA_t[:,:,reverse[0]+1:reverse[1]+1] = boundB_t[::-1,:,reverse[0]+1:reverse[1]+1]
        boundB_t[:,:,reverse[0]+1:reverse[1]+1] = boundA_before

    return centerline_t,boundA_t,boundB_t

def segment_processing(segment):

    boundA = segment['BoundaryA']
    boundB = segment['BoundaryB']
    stage_position = segment['stage_position']
    timestamp = segment['timestamp']
    T = boundA.shape[2]
    # process and correct centerline
    centerline = WormTool.timeseries.boundary2centerline(boundA,boundB,direct_mean=False) # (N,2,T)
    centerline = centerline*1.683 - stage_position*0.05 # (N,2,T)
    boundA = boundA*1.683 - stage_position*0.05
    boundB = boundB*1.683 - stage_position*0.05
    worm_len = np.sum(np.linalg.norm(np.diff(centerline,axis=0),axis=1),axis=0)# average worm length (T,)
    avg_worm_len = worm_len.mean()
    centerline,boundA,boundB = head_tail_reverse_correction(centerline,boundA,boundB,avg_worm_len)
    # Sliding speed
    v_A,v_B = sliding_speed(boundA,boundB,timestamp[:,0])
    v_A_middle_mean = v_A[20:80,:].mean(axis=0) # (T,)
    v_B_middle_mean = v_B[20:80,:].mean(axis=0) # (T,)
    slide_speed = (v_A_middle_mean + v_B_middle_mean)/2 # (T,)
    slide_speed_smooth = csaps(timestamp[:,0],slide_speed,timestamp[:,0],smooth=0.999)
    speed_sign = np.sign(slide_speed)
    curvature_t = WormTool.timeseries.centerline2curvature(centerline) # (N,T)
    CoM = centerline.mean(axis=0) # center of mass (2,T)
    ds = CoM[:,2:]-CoM[:,:-2] # (2,T-2)
    ds_norm = np.linalg.norm(ds,axis=0) # (T-2,)
    dt =  timestamp[2:,0]-timestamp[:-2,0] # (T-2,)
    velocity = ds_norm*speed_sign[1:-1]/dt # (T-2,)
    velocity = np.concatenate([velocity[0:1],velocity,velocity[-1:]]) # (T,)
     
    #phase speed on eigenworm basis
    eigen_body_basis = np.load('/Users/hengzhang/projects/WormSim/excursion/Eigenworm_basis/eigenworms_N2_body.npy')
    eigen_body = curvature_t[30:].T@eigen_body_basis[:,:2] # (T,2)
    body_phase = np.unwrap(np.arctan2(eigen_body[:,1],-eigen_body[:,0]))
    dbody_phase = (body_phase[2:]-body_phase[:-2])/(timestamp[2:,0]-timestamp[:-2,0])
    smooth_dbody_phase = csaps(timestamp[1:-1,0],dbody_phase,timestamp[1:-1,0],smooth=0.999)
    smooth_dbody_phase = np.concatenate([smooth_dbody_phase[0:1],smooth_dbody_phase,smooth_dbody_phase[-1:]],axis=0)
    # digitize to 3 states 1 0 and -1: forward, stop, backward
    # if phase smooth_dbody_phase > 0.4, forward; if smooth_dbody_phase speed < -0.4, backward; else stop
    state = state_calc(smooth_dbody_phase,max_break=10,threshold=0.4)
    return {'centerline':centerline,'boundA':boundA,'boundB':boundB,'curvature':curvature_t,'velocity':velocity,'CoM':CoM,'worm_len':worm_len,
            'slide_speed_smooth':slide_speed_smooth,'state':state,'body_phase_speed':smooth_dbody_phase}

def state_calc(body_phase_speed_smooth,max_break=10,threshold=0.4):
    '''
    Parameters: 
    body_phase_speed_smooth: the smoothed body phase speed; phase is the angle of the eigen-body(30%-100%) a1-a2 plane
    max_break: the maximum break time steps of a state. If the break time steps of a state is larger than max_break, the break period will be considered as a new state.
    threshold: the threshold to determine the forward, backward and stop states.
    Return:
    state: a 1D array of 1,0,-1, representing forward, stop and backward
    '''
    assert max_break > 0, 'max_break should be larger than 0'
    assert threshold > 0, 'threshold should be larger than 0'
    state = np.zeros_like(body_phase_speed_smooth)
    forward = (body_phase_speed_smooth > threshold).astype(np.int32)
    forward = scipy.signal.medfilt(forward,kernel_size=2*max_break+1) # median filter to remove short breaks
    backward = (body_phase_speed_smooth < -threshold).astype(np.int32)
    backward = scipy.signal.medfilt(backward,kernel_size=2*max_break+1)
    state[forward>0] = 1
    state[backward>0] = -1
    return state



def ones_segment(zero_ones):
    '''
    Parameters:
    zero_ones: a 1D array of 0s and 1s
    Return:
    segments: a list of tuples, each tuple is the start and end index of a segment of 1s
    '''
    segments = []
    t = 0
    while t < len(zero_ones):
        if zero_ones[t] == 1:
            start = t
            while t < len(zero_ones) and zero_ones[t] == 1:
                t += 1
            end = t-1
            segments.append((start,end))
        t += 1
    return segments

def whole_track_processing(track_mat_file):
    '''
    track_mat_file: the mat file containing the tracking data: BoundaryA, BoundaryB, stage_position, timestamp
    '''
    track_mat = scipy.io.loadmat(track_mat_file)
    BoundaryA = track_mat['BoundaryA'].astype(np.float64)
    BoundaryB = track_mat['BoundaryB'].astype(np.float64)
    stage_position = track_mat['stage_position'].astype(np.float64) 
    timestamp = track_mat['timestamp'].astype(np.float64)
    T = len(timestamp)
    break_points = np.ones(T)
    for i in range(T):
        if not normal_boundary(BoundaryA[:,:,i],BoundaryB[:,:,i]):
            break_points[i] = 0
    segments = ones_segment(break_points)
    centerline = np.nan*np.zeros_like(BoundaryA)
    sliding_speed = np.nan*np.zeros(T)
    sliding_speed_smooth = np.nan*np.zeros(T)
    state = np.nan*np.zeros(T)
    curvature = np.nan*np.zeros((100,T))
    velocity = np.nan*np.zeros(T)
    CoM = np.nan*np.zeros((2,T))
    worm_len = np.nan*np.zeros(T)
    BoundaryA_corrected = BoundaryA.copy()
    BoundaryB_corrected = BoundaryB.copy()
    body_phase_speed = np.nan*np.zeros(T)
    for i in range(len(segments)):
        start,end = segments[i]
        logger.debug('Segment processing: start:{}, end:{}'.format(start,end))
        if end-start < 50:
            logger.debug('Segment{}-{} too short, skip'.format(start,end))
            continue
        segment = {'BoundaryA':BoundaryA[:,:,start:end+1],'BoundaryB':BoundaryB[:,:,start:end+1],
                    'stage_position':stage_position[:,:,start:end+1],'timestamp':timestamp[start:end+1]}
        result = segment_processing(segment)
        centerline[:,:,start:end+1] = result['centerline']
        # sliding_speed[start:end+1] = result['slide_speed']
        sliding_speed_smooth[start:end+1] = result['slide_speed_smooth']
        curvature[:,start:end+1] = result['curvature']
        velocity[start:end+1] = result['velocity']
        CoM[:,start:end+1] = result['CoM']
        worm_len[start:end+1] = result['worm_len']
        BoundaryA_corrected[:,:,start:end+1] = result['boundA']
        BoundaryB_corrected[:,:,start:end+1] = result['boundB']
        state[start:end+1] = result['state']
        body_phase_speed[start:end+1] = result['body_phase_speed']
        # if logger.level == logging.DEBUG:
        #     break
        # if i==1:
        #     break
    if logger.level == logging.DEBUG:
        return {'timestamp':timestamp,'centerline':centerline,'sliding_speed':sliding_speed,
            'curvature':curvature,'sliding_speed_smooth':sliding_speed_smooth,
            'velocity':velocity,'CoM':CoM,'worm_len':worm_len,'break_points':break_points,'segments':segments,
           'BoundaryA':BoundaryA_corrected,'BoundaryB':BoundaryB_corrected,'state':state}
    else:
        return {'timestamp':timestamp,'centerline':centerline,
            'curvature':curvature,'sliding_speed':sliding_speed_smooth,
            'velocity':velocity,'CoM':CoM,'worm_len':worm_len,'break_points':break_points,'segments':segments,
           'state':state,'body_phase_speed':body_phase_speed}
    

#%%
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    mat_file = '/Users/hengzhang/projects/WormSim/neuron_ablations/rawdata/N2_processed/20190811_1503_w0.mat'
    result = whole_track_processing(mat_file)

    
# %%
if __name__ == '__main__':
    curvature = result['curvature']
    slide_speed_smooth = result['sliding_speed_smooth']/np.nanmean(result['worm_len'])
    slide_speed = result['sliding_speed']/np.nanmean(result['worm_len'])
    segments = result['segments']
    start,end = segments[7]
    end = start+2000
    sign = np.sign(slide_speed_smooth[start:end+1])
    fig,ax = plt.subplots(3,1,figsize=(5,10))
    ax[0].imshow(curvature[:,start:end+1],aspect='auto',cmap='coolwarm',vmax=10,vmin=-10,interpolation='None')
    ax[1].plot(sign)
    ax[2].plot(slide_speed[start:end+1])
    ax[2].plot(slide_speed_smooth[start:end+1])
    plt.show()
#%%
if __name__ == '__main__':
    curvature = result['curvature']
    segments = result['segments']
    start,end = segments[7]
    start +=8000
    end = start+2000
    state = result['state']
    fig,ax = plt.subplots(3,1,figsize=(5,10))
    ax[0].imshow(curvature[:,start:end+1],aspect='auto',cmap='coolwarm',vmax=10,vmin=-10,interpolation='None')
    ax[1].plot(state[start:end+1])
    ax[2].plot(result['sliding_speed_smooth'][start:end+1])
    plt.show()

# %%
if __name__ == '__main__':
    # plot boundary A and B
    from ipywidgets import interact,fixed
    boundaryA = result['BoundaryA']
    boundaryB = result['BoundaryB']
    xmin,xmax = boundaryA[:,0,start:end+1].min(),boundaryA[:,0,start:end+1].max()
    ymin,ymax = boundaryA[:,1,start:end+1].min(),boundaryA[:,1,start:end+1].max()
    sidel_length = max(xmax-xmin,ymax-ymin)
    def plot(i):
        i = i+start
        plt.plot(boundaryA[:,0,i],boundaryA[:,1,i],'r')
        plt.plot(boundaryA[0,0,i],boundaryA[0,1,i],'ro')
        plt.plot(boundaryB[:,0,i],boundaryB[:,1,i],'b')
        plt.plot(boundaryB[0,0,i],boundaryB[0,1,i],'bo')

        plt.plot(boundaryA[:,0,i-1],boundaryA[:,1,i-1],'r',alpha=0.5)
        plt.plot(boundaryA[0,0,i-1],boundaryA[0,1,i-1],'ro',alpha=0.5)
        plt.plot(boundaryB[:,0,i-1],boundaryB[:,1,i-1],'b',alpha=0.5)
        plt.plot(boundaryB[0,0,i-1],boundaryB[0,1,i-1],'bo',alpha=0.5)
        plt.xlim([xmin-sidel_length*0.2,xmin+sidel_length*1.2])
        plt.ylim([ymin-sidel_length*0.2,ymin+sidel_length*1.2])
        # equal axis
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        # plt.plot()
        # plt.show()
    # interact(plot,i=(start+400,start+500))
    interact(plot,i=(0,end-start))
# %%
if __name__ == '__main__':
    curvature = result['curvature']
    # start,end = segments[7]
    monitor = MotionState(curvature[:,start:end+1])
    time = result['timestamp'][start:end+1,0]
    phase = monitor.phase
    dphase = np.diff(np.unwrap(phase))/np.diff(time)
    smooth_dphase = csaps(time[1:],dphase,time[1:],smooth=0.999)
    # eigen body basis
    eigen_body_basis = np.load('../WormSim/excursion/Eigenworm_basis/eigenworms_N2_body.npy')
    eigen_body = curvature[30:,start:end+1].T@eigen_body_basis[:,:2]
    body_phase = np.arctan2(eigen_body[:,1],-eigen_body[:,0])
    dbody_phase = np.diff(np.unwrap(body_phase))/np.diff(time)
    smooth_dbody_phase = csaps(time[1:],dbody_phase,time[1:],smooth=0.999)
    plt.figure(dpi=200)
    plt.plot(dphase)
    plt.plot(smooth_dphase,'r',label='whole body')
    plt.plot(smooth_dbody_phase,'g',label='body')
    plt.hlines(0.5,0,len(dphase),linestyles='dashed',colors='m')
    plt.show()

# %%
if __name__ == '__main__':
    # plot centerline
    centerline = result['centerline']
    segments = result['segments']
    start,end = segments[7]
    end = start+2000
    slide_speed = result['sliding_speed']/np.nanmean(result['worm_len'])
    speed_sign = (slide_speed[start:end+1]>0.1 )
    eigen_body_basis = np.load('../WormSim/excursion/Eigenworm_basis/eigenworms_N2_body.npy')
    eigen_body = curvature[:70,start:end+1].T@eigen_body_basis[:,:2]
    body_phase = np.arctan2(eigen_body[:,1],-eigen_body[:,0])
    dbody_phase = np.diff(np.unwrap(body_phase))/np.diff(time)
    smooth_dbody_phase = csaps(time[1:],dbody_phase,time[1:],smooth=0.999)
    smooth_dbody_phase = np.concatenate([smooth_dbody_phase[0:1],smooth_dbody_phase],axis=0)
    phase_sign = (smooth_dbody_phase >0.4)
    # united sign
    sign = np.logical_or(speed_sign,phase_sign)
    sign = sign.astype(np.int)
    fig,ax = plt.subplots(4,1,figsize=(5,10))
    ax[0].imshow(curvature[:,start:end+1],aspect='auto',cmap='coolwarm',vmax=10,vmin=-10,interpolation='None')
    ax[1].plot(sign)
    ax[2].plot(slide_speed[start:end+1])
    ax[2].hlines(0.1,0,len(slide_speed[start:end+1]),linestyles='dashed',colors='m')
    ax[3].plot(smooth_dbody_phase)
    ax[3].hlines(0.4,0,len(smooth_dbody_phase),linestyles='dashed',colors='m')



    
# %%
