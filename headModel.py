import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Bistable():
    def __init__(self,param=None,**kwargs):
        if param is not None:
            self.tau_k = param['tau_k']
            self.tau_m = param['tau_m']
            self.M0 = param['M0']
            self.dt = param['dt']
            self.prop_th = param['prop_th']
            self.v_prop_coef = param['v_prop_coef']
        else:
            self.tau_k = kwargs.get('tau_k',0.4)
            self.tau_m = kwargs.get('tau_m',0.1)
            self.M0 = kwargs.get('M0',1)
            self.dt = kwargs.get('dt',0.01)
            self.prop_th = kwargs.get('prop_th',0.9)
            self.v_prop_coef = kwargs.get('v_prop_coef',0.17)
        self.M = 0
        self.k = 0
        self.state = 1
        
    def update(self):
        dMdt = (self.M0*self.state - self.M)/self.tau_m
        dkdt = (self.M-self.k)/self.tau_k
        self.M += dMdt*self.dt
        self.k += dkdt*self.dt
        prop_sig = self.prop_function(self.k)
        if self.state == 1 and prop_sig > self.prop_th:
            self.state = -1
        elif self.state == -1 and prop_sig < -self.prop_th:
            self.state = 1

    def update_prop_input(self,prop_input):
        dMdt = (self.M0*self.state - self.M)/self.tau_m
        dkdt = (self.M-self.k)/self.tau_k
        self.M += dMdt*self.dt
        self.k += dkdt*self.dt
        prop_sig = prop_input
        if self.state == 1 and prop_sig > self.prop_th:
            self.state = -1
        elif self.state == -1 and prop_sig < -self.prop_th:
            self.state = 1
        
    def prop_function(self,k):
        dkdt = (self.M-self.k)/self.tau_k
        prop_sig = k+self.v_prop_coef*dkdt
        return prop_sig

    def run(self,T):
        t = np.arange(0,T,self.dt)
        k_t = np.zeros((len(t),))
        for ti in range(len(t)):
            self.update()
            k_t[ti] = self.k
        return k_t

    def plot(self,k_t):
        plt.figure(dpi=300)
        plt.plot(k_t)
        plt.show()

def head_kernel(t,tau_k,tau_m):
    N = len(t)
    dt = t[1]-t[0]
    A = np.zeros((N,N))
    kernel_t = np.arange(0,2,dt)
    kernel =1/(tau_k-tau_m)*(np.exp(-kernel_t/tau_k) - np.exp(-kernel_t/tau_m))
    for j in range(N):
        end = min(j+len(kernel),N)
        A[j:end,j] = kernel[:end-j]*dt
    return A

##############################################
class FHNneuron:
    def __init__(self,**kwargs):
        self.tau_fast = kwargs.get('tau_fast',0.02)
        self.tau_slow = kwargs.get('tau_slow',0.2)
        self.a = kwargs.get('a',0.7)
        self.dt = kwargs.get('dt',0.005)
        self.v0 = kwargs.get('v0',0)
        self.v = np.random.randn()
        self.w = 0
        self.vt = None
        self.wt = None
        self.It = None
    
    def vector_field(self,v,w,I):
        '''
        Return the vector field of the FHN neuron.
        '''
        v0 = self.v0
        dvdt = ((v-v0) - (v-v0)**3/3 - w + I)/self.tau_fast
        dwdt = (v - v0 - self.a*w)/self.tau_slow
        return dvdt,dwdt

    def update(self,I):
        v = self.v
        w = self.w
        v0 = self.v0
        dvdt = ((v-v0) - (v-v0)**3/3 - w + I)/self.tau_fast
        dwdt = (v-v0 - self.a*w)/self.tau_slow
        self.v = v + dvdt*self.dt
        self.w = w + dwdt*self.dt

    def update_noisy(self,I,sigma):
        v = self.v
        w = self.w
        v0 = self.v0
        dvdt = ((v-v0) - (v-v0)**3/3 - w + I)/self.tau_fast
        dwdt = (v - v0 - self.a*w)/self.tau_slow
        self.v = v + dvdt*self.dt + sigma*np.random.randn()*np.sqrt(self.dt)
        self.w = w + dwdt*self.dt + sigma*np.random.randn()*np.sqrt(self.dt)

    
    def run(self,It,noise=None):
        '''
        Run FHN neuron simulation with input current It.
        If noise is not None, add noise to the neuron dynamics.(std of noise = the value of variable 'noise')
        '''
        t = np.arange(len(It))*self.dt
        v_t = np.zeros((len(t),))
        w_t = np.zeros((len(t),))
        if noise is None:
            for ti in range(len(t)):
                self.update(It[ti])
                v_t[ti] = self.v
                w_t[ti] = self.w
        else:
            for ti in range(len(t)):
                self.update_noisy(It[ti],noise)
                v_t[ti] = self.v
                w_t[ti] = self.w
        self.vt = v_t
        self.wt = w_t
        self.It = It
        return v_t,w_t

        
    def plot(self,flag='1D'):
        '''
        If flag = '1D',Plot the trajectory of the neuron voltage and current.
        If flag = '2D',Plot the nullcline and vt and wt in phase space.

        '''
        if self.vt is not None:
            if flag == '1D':
                plt.figure(dpi=300)
                plt.plot(self.vt)
                if self.It is not None:
                    plt.plot(self.It)
                plt.show()
            elif flag == '2D':
                v_range = np.arange(-2,2,0.01)
                v_nullcline_w = (v_range-self.v0) - (v_range-self.v0)**3/3 
                w_nullcline_w = 1/self.a * (v_range-self.v0)
                plt.figure(dpi=300)
                plt.plot(v_range,v_nullcline_w,'b',label='v nullcline',linestyle='dashed')
                plt.plot(v_range,w_nullcline_w,'b',label='w nullcline',linestyle='dashed')
                plt.plot(self.vt,self.wt,'r',label='trajectory')
                plt.legend()
                plt.xlim(-2,2)
                plt.ylim(-2,2)
                plt.show()
            else:
                print('flag must be 1D or 2D')

    def nullclines(self, v_range, I=None):
        I = I if I is not None else 0
        v_nullcline_w = (v_range-self.v0) - (v_range-self.v0)**3/3 +I
        w_nullcline_w = 1/self.a * (v_range-self.v0)
        return v_nullcline_w, w_nullcline_w

    def make_animation(self,name,downsample=10,interval=50):
        '''
        Make an animation of the moving nullclines and state trajectory.
        '''
        vt = self.vt
        wt = self.wt
        It = self.It
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        v_range = np.arange(-2,2,0.01)
        w_nullcline, = ax.plot([],[],'b',label='w nullcline',linestyle='dashed')
        v_nullcline, = ax.plot([],[],'b',label='v nullcline',linestyle='dashed')
        trajectory, = ax.plot([],[],'or',label='trajectory')
        def anim_func(frame):
            v = vt[frame]
            w = wt[frame]
            I = It[frame]
            v_nullcline.set_data(v_range,v_range - v_range**3/3-I)
            w_nullcline.set_data(v_range,1/self.a * v_range)
            trajectory.set_data(v,w)
            return v_nullcline,w_nullcline,trajectory
        anim = animation.FuncAnimation(fig,anim_func,frames=np.arange(0,len(vt),downsample),interval=interval,blit=True)
        anim.save(name, writer='imagemagick')
        plt.close(fig)
        print('Gif saved to %s'%name)

class BistableNeuron:
    def __init__(self,**kwargs) -> None:
        self.th1 = kwargs.get('th1',0.9)
        self.th2 = kwargs.get('th2',0.2)
        self.x = 0
    def update(self,I):
        if self.x == 0 and I > self.th1:
            self.x = 1
        elif self.x == 1 and I < self.th2:
            self.x = 0


class Head:
    def __init__(self,**kwargs) -> None:
        self.tau_m = kwargs.get('tau_m',0.1)
        self.tau_k = kwargs.get('tau_k',0.4)
        self.M = kwargs.get('M',1)
        self.K = kwargs.get('K',0)
        self.dt = kwargs.get('dt',0.01)
        
    def update(self,X):
        dMdt = (X-self.M)/self.tau_m
        dKdt = (self.M-self.K)/self.tau_k
        self.M += dMdt*self.dt
        self.K += dKdt*self.dt

if __name__ == '__main__':
    model = Bistable()
    k_t = model.run(10)
    model.plot(k_t)
