import numpy as np
import numpy.linalg as la

def state_identify_func(segment1,segment2):
    '''
    This function returns a function that takes in two signals and their posiotion 
    defined by the segment1 and segment2.
    For example, if segment1 = [-10,-5,0,5,10] and signal1 falls in the range [-5,0],
    the position of signal1 is 1.
    '''
    def func(sig1,sig2):
        pos1 = np.digitize(sig1,segment1)
        pos2 = np.digitize(sig2,segment2)
        return pos1,pos2, pos1*(len(segment2)+1)+pos2
    return func

def joint_state_transition(sig1,sig2,segment1,segment2,downsample=None):
    '''
    Estimate the joint state transition matrix.
    
    sig1 and sig2 are the 1D signals
    segment1 and segment2 are the state discretizing thresholds.
    There will be a number of (len(segment1)-1)*(len(segment2)-1) joint states.
    '''
    assert len(sig1)==len(sig2)
    if downsample is not None and downsample>1:
        sig1 = sig1[::downsample]
        sig2 = sig2[::downsample]
    state_ind_func = state_identify_func(segment1,segment2)
    num_states = (len(segment1)+1)*(len(segment2)+1)
    joint_state_transition = np.zeros((num_states,num_states))
    for i in range(len(sig1)-1):
        _,_,state1 = state_ind_func(sig1[i],sig2[i])
        _,_,state2 = state_ind_func(sig1[i+1],sig2[i+1])
        joint_state_transition[state1,state2] += 1
    return joint_state_transition/joint_state_transition.sum(axis=1,keepdims=True)

def RBFkernel(t1,t2,**kwargs):
    try: 
        sig = kwargs['sig']
        l = kwargs['l']
    except:
        raise ValueError('sig and l must be specified in the kwargs for RBF kernel')

    return sig**2*np.exp(-np.linalg.norm(t1-t2)**2/(2*l**2))

def LocalPeriodicKernel(t1,t2,**kwargs):
    try: 
        sig = kwargs['sig']
        lp = kwargs['lp']
        le = kwargs['le']
        p = kwargs['p']
    except:
        raise ValueError('sig, lp, le, p must be specified in the kwargs for Local Periodic kernel')
    return sig**2*np.exp(-(t1-t2)**2/(2*le**2))*np.exp(-2*np.sin(np.pi*abs(t1-t2)/p)**2/lp**2)

def PeriodicKernel(t1,t2,**kwargs):
    try:
        sig = kwargs['sig']
        l = kwargs['l']
        p = kwargs['p']
    except:
        raise ValueError('sig, l, p must be specified in the kwargs for Periodic kernel')
    return sig**2*np.exp(-2*np.sin(np.pi*abs(t1-t2)/p)**2/l**2)

def SVD_inv(M):
    U,S,V = np.linalg.svd(M)
    S_inv = np.diag(1/S)
    return V.T@S_inv@U.T
def Cholesky_inv(M,eps=1e-6):
    L = la.cholesky(M+eps*np.eye(M.shape[0]))
    L_inv = la.inv(L)
    return L_inv.T@L_inv

class GPdeconv:
    '''
    Noise is no longer white noise but a GP
    Example:
    deconv_torque = GPdeconv('RBF',{'tau_k':0.3,'tau_m':0.1,'noise_sig':0.4,'noise_l':0.1,'sig':5,'l':0.1})
    T = len(y)
    dt = 0.02
    t = np.arange(0,T*dt,dt)
    deconv_torque.get_A_curv(t)
    mu = deconv_torque.deconv(y,dt)
    x_sample = deconv_torque.sample_post(mu,dt)
    '''
    def __init__(self,kernel,param):
        self.kernel = kernel
        self.tau_k = param['tau_k']
        self.tau_m = param['tau_m']
        # self.dt = param['dt']
        self.param = param
    
    def get_cov_mat(self,t,param):
        N = len(t)
        K = np.zeros((N,N))
        if param['kernel'] == 'RBF':
            kernel_func = RBFkernel
            try:
                sig = param['sig']
                l = param['l']
            except:
                print('sig and l not specified for RBF kernel')
                
        for i in range(N):
            for j in range(N):
                K[i,j] = kernel_func(t[i],t[j],sig,l)
        return K
    
    def get_A_mus_curv(self,t):
        # get the transform matrix A
        N = len(t)
        dt = t[1]-t[0]
        A = np.zeros((N,N))
        kernel_t = np.arange(0,2,dt)
        kernel =1/(self.tau_k-self.tau_m)*(np.exp(-kernel_t/self.tau_k) - np.exp(-kernel_t/self.tau_m))
        for j in range(N):
            end = min(j+len(kernel),N)
            A[j:end,j] = kernel[:end-j]*dt
        self.A = A

    def get_A_curv(self,t):
        N = len(t)
        dt = t[1]-t[0]
        A = np.zeros((N,N))
        kernel_t = np.arange(0,2,dt)
        kernel =1/(self.tau_k)*np.exp(-kernel_t/self.tau_k)
        for j in range(N):
            end = min(j+len(kernel),N)
            A[j:end,j] = kernel[:end-j]*dt
        self.A = A

    def deconv(self,y,dt):
        T = len(y)
        kernel_t = np.arange(0,T)*dt
        param_X = {'kernel':'RBF','sig':self.param['sig'],'l':self.param['l']}
        param_noise = {'kernel':'RBF','sig':self.param['noise_sig'],'l':self.param['noise_l']}
        K = self.get_cov_mat(kernel_t,param_X)
        N = self.get_cov_mat(kernel_t,param_noise)
        self.cov_mat_prior = K
        if not hasattr(self,'A'):
            # raise Exception('A not defined')
            raise Exception('A not defined')
        A = self.A
        # K_deconv_inv = la.pinv(self.cov_mat_prior)+ ATA/self.noise_sig**2
        # self.cov_mat_post = la.pinv(K_deconv_inv)
        # inv_func = SVD_inv # function to do inversion
        inv_func = Cholesky_inv
        N_inv = inv_func(N)
        K_deconv_inv = inv_func(K) + A.T@N_inv@A
        self.cov_mat_post = inv_func(K_deconv_inv)
        self.mu = self.cov_mat_post.T@A.T@N_inv@y
        return self.mu 

    def sample_prior(self,t=None):
        if t is None: 
            if not hasattr(self,'cov_mat_prior'):
                # warning('cov_mat_prior not defined')
                raise Exception('cov_mat_prior not defined')
            else:
                cov_mat = self.cov_mat_prior
        else:
            param_X = {'kernel':'RBF','sig':self.param['sig'],'l':self.param['l']}
            cov_mat = self.get_cov_mat(t,param_X)

        return np.random.multivariate_normal(np.zeros(cov_mat),cov_mat)
    
    def sample_post(self):
        if not hasattr(self,'cov_mat_post'):
            raise Exception('cov_mat_post not defined')
        return np.random.multivariate_normal(self.mu,self.cov_mat_post)


# class GP:
#     def __init__(self,kernel,param):
#         self.kernel = kernel
#         self.param = param

#     def get_cov_mat(self,t):
#         N = len(t)
#         K = np.zeros((N,N))
#         if self.kernel == 'RBF':
#             kernel_func = RBFkernel
#             try:
#                 sig = self.param['sig']
#                 l = self.param['l']
#             except:
#                 print('sig and l not specified for RBF kernel')
        
#             for i in range(N):
#                 for j in range(N):
#                     K[i,j] = kernel_func(t[i],t[j],sig,l)
#         else:
#             kernel_func = self.kernel
#             for i in range(N):
#                 for j in range(N):
#                     K[i,j] = kernel_func(t[i],t[j])
#         self.K = K
    
#     def sampleGP(self,t=None):
#         if t is None:
#             if not hasattr(self,'K'):
#                 # warning('cov_mat_prior not defined')
#                 raise Exception('K not defined')
#             K = self.K
#         else:
#             self.get_cov_mat(t)
#             K = self.K
#         return np.random.multivariate_normal(np.zeros(len(self.K)),K)
    
class GP:
    def __init__(self,kernel,**kwargs):
        self.kernel = kernel
        self.kwargs = kwargs
    
    def get_cov_mat(self,t):
        N = len(t)
        K = np.zeros((N,N))
        if self.kernel == 'RBF':
            kernel_func = RBFkernel
        elif self.kernel == 'LocalPeriodic':
            kernel_func = LocalPeriodicKernel
        elif self.kernel == 'Periodic':
            kernel_func = PeriodicKernel
        elif callable(self.kernel):
            kernel_func = self.kernel
        else:
            raise Exception('kernel not defined')
        for i in range(N):
            for j in range(N):
                K[i,j] = kernel_func(t[i],t[j],**self.kwargs)
        self.K = K
        
    def sampleGP(self,t=None):
        if t is None:
            if not hasattr(self,'K'):
                # warning('cov_mat_prior not defined')
                raise Exception('K not defined')
            else:
                K = self.K
        else:
            self.get_cov_mat(t)
            K = self.K
        return np.random.multivariate_normal(np.zeros(len(self.K)),K)
    
    