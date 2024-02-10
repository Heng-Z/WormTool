#%%
import os
import signal
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import warnings
import psutil

class Worm:
    '''Worm locomotion simulator based on the neuro-mechanical model and code by Cohen 2012
    
    TODO
    ----
    0. Add more detailed comments
    '''
    def __init__(self,model_path='./program',controller_port = '/dev/cu.usbmodem144101',arduino_port=None,**kwargs):
        self.NSEG = 48
        self.NBAR = self.NSEG + 1
        self.model_path = model_path
        self.dt = 0.01
        self.fb_strength = 1.45
        self.fb_weight = self.get_fb_weight()*self.fb_strength
        self.mean_mat = self.get_mean_mat()
        self.params = {'th1':-0.3,'th2':0.3,'tau_m':0.1,'dt':self.dt}
        self.neuron_scale = kwargs.get('neuron_scale',1)
        # Load the mechanical model
        self.mech_model = subprocess.Popen([self.model_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True)
        if arduino_port is not None: # Some old codes used old name, arduino_port, instead of controller_port
            controller_port = arduino_port
        if controller_port is None:
            self.controller = None
        # Connect to the arduino
        else:
            try:
                controller = serial.Serial(controller_port,9600) 
            except:
                controller = None
                warnings.warn('Arduino not connected')
            self.controller = controller
        self.V = np.zeros(12)
        self.yval = np.zeros((self.NBAR,3))
        self.D = 80e-6
        self.R = self.D/2.0*abs(np.sin(np.arccos((np.arange(self.NBAR)-self.NSEG/2.0)/(self.NSEG/2.0 + 0.2))))

    def get_fb_weight(self):
        neighb = 1
        local = 1
        fb_weight = np.zeros((47,12))
        fb_weight[0:3,0] = 1/3*0.5
        fb_weight[3:7,0] = 1/4*0.5
        fb_weight[0:3,1] = -1/3
        fb_weight[3:7,1] = 1/4
        for i in range(2,12):
            fb_weight[4*(i-1)-1:4*i-1,i] = -1/4
            fb_weight[4*i-1:4*(i+1)-1,i] = 1/4
        return fb_weight

    def get_mean_mat(self):
        mean_mat = np.zeros((47,12))
        mean_mat[0:3,0] = 1/3
        for i in range(1,12):
            mean_mat[4*i-1:4*i+3,i] = 1/4
        return mean_mat

    def update_worm(self,head_input=None):
        # Update the voltage and the position of the worm
        try:
            from_model = self.mech_model.stdout.readline().strip()
            if from_model == "-1":
                pass
            yval_str, sr_str, neuron_str = from_model.split('\t')
            yval_str = yval_str.strip(' ,').split(',')
        except:
            warnings.warn('Mechanical model not responding')
        yval = self.yval
        for i in range(self.NBAR):
            yval[i][0] = float(yval_str[i*3]) # x position
            yval[i][1] = float(yval_str[i*3+1]) # y position
            yval[i][2] = float(yval_str[i*3+2]) # angle
        #### Calculate feedback and update neuron ####
        DX = yval[1:,0] - yval[:-1,0]
        DY = yval[1:,1] - yval[:-1,1]
        theta = np.arctan2(DY,DX)
        theta = np.unwrap(theta)
        curvature = theta[1:] - theta[:-1]  
        fb = curvature@self.fb_weight
        # Update nueron the head [0] and body [1:11] are updated separately
        V = self.V
        for i in range(1,12):
            V[i] = self.update_neuron(V[i],self.params,fb[i]) * self.neuron_scale
        # Update the head
        if self.controller is not None:
            # Get the input from arduino
            # print('Get control signal')
            self.controller.write(b'g') 
            try:
                readOut = self.controller.readline().decode()
            except:
                # raise Warning('Arduino Controller not responding')
                readOut = 512
            V[0] = (int(readOut)-512.0)/512.0
        elif head_input is not None:
            V[0] = head_input
        else:
            raise Exception('No input to the head')
        ################# Write to the mechanical model #################
        V_all = np.concatenate((V,-V))
        # print(frame,V_all)
        for i in range(self.NSEG*2):
            self.mech_model.stdin.write(str(V_all[i//4])) # 4 means the 4 segments per unit
            self.mech_model.stdin.write(' ')
        self.mech_model.stdin.write('\n')
        self.mech_model.stdin.flush()

    def follow_preset_head_input(self,head_input,save_tempdata=False):
        T = len(head_input)
        DV_XY = np.zeros((T,self.NBAR,4))
        yval_t = np.zeros((T,self.NBAR,3))
        k_t = np.zeros((T,self.NBAR-2))
        for t in range(T):
            self.update_worm(head_input[t])
            yval_t[t] = self.yval
            DX = yval_t[t,1:,0] - yval_t[t,:-1,0]
            DY = yval_t[t,1:,1] - yval_t[t,:-1,1]
            theta = np.arctan2(DY,DX)
            theta = np.unwrap(theta)
            k_t[t] = theta[1:] - theta[:-1]
            DorX,DorY,VenX,VenY = self.get_drosal_ventral_curve()
            DV_XY[t,:,0] = DorX
            DV_XY[t,:,1] = DorY
            DV_XY[t,:,2] = VenX
            DV_XY[t,:,3] = VenY
        if save_tempdata is True: 
            self.DV_XY = DV_XY
        return yval_t,k_t,DV_XY 

    def update_neuron(self,V,param,fb):
        out,V = neural_hyteresis(V,fb,param)
        return V
    
    def get_drosal_ventral_curve(self):
        yval = self.yval
        R = self.R
        Dorsal_X = yval[:,0] + R*np.cos(yval[:,2])
        Dorsal_Y = yval[:,1] + R*np.sin(yval[:,2])
        Ventral_X = yval[:,0] - R*np.cos(yval[:,2])
        Ventral_Y = yval[:,1] - R*np.sin(yval[:,2])
        return Dorsal_X,Dorsal_Y,Ventral_X,Ventral_Y
    
    def save_gif(self,name,dpi=150,size=3,interval=100,downsample=10):
        if not hasattr(self,'DV_XY'):
            raise Exception('No DV_XY data, please run follow_preset_head_input(save_tempdata=True) first')
        DV_XY = self.DV_XY
        T = DV_XY.shape[0]
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim([-size*1e-3,size*1e-3])
        ax.set_ylim([-size*1e-3,size*1e-3])
        line_dorsal, = ax.plot([],[],'k')
        line_ventral, = ax.plot([],[],'k')
        textb = ax.text(0,0, '')
        def anim_func(frame):
            line_dorsal.set_data(DV_XY[frame,:,0],DV_XY[frame,:,1])
            line_ventral.set_data(DV_XY[frame,:,2],DV_XY[frame,:,3])
            textb.set_text('frame:%d'%frame)
            return [line_dorsal,line_ventral,textb]
        anim = animation.FuncAnimation(fig,anim_func,frames=np.arange(0,T,downsample),interval=interval,blit=True)
        anim.save(name, writer='imagemagick')
        plt.close(fig)
        print('Gif saved to %s'%name)

    def close(self):
        os.killpg(os.getpgid(self.mech_model.pid), signal.SIGTERM)
        self.controller.close()

def neural_hyteresis(x,k,params):
    th1 = params['th1']
    th2 = params['th2']
    if k < th1:
        x = 1
        out = 1
    elif k > th2:
        x = -1
        out = -1
    else:
        out = x
    return out,x

class WormPuppet:
    def __init__(self,model_path='./program') -> None:
        self.NSEG = 48
        self.NBAR = self.NSEG + 1
        self.model_path = model_path
        self.dt = 0.01
        # Load the mechanical model
        self.mech_model = subprocess.Popen([self.model_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True)
        
        self.V = np.zeros(12) # Voltage of muscle
        self.yval = np.zeros((self.NBAR,3))
        self.D = 80e-6
        self.R = self.D/2.0*abs(np.sin(np.arccos((np.arange(self.NBAR)-self.NSEG/2.0)/(self.NSEG/2.0 + 0.2))))
    
    def update_worm(self,V_muscle):
        # Update the voltage and the position of the worm
        try:
            from_model = self.mech_model.stdout.readline().strip()
            if from_model == "-1":
                pass
            yval_str, sr_str, neuron_str = from_model.split('\t')
            yval_str = yval_str.strip(' ,').split(',')
        except:
            warnings.warn('Mechanical model not responding')
        yval = self.yval
        for i in range(self.NBAR):
            yval[i][0] = float(yval_str[i*3]) # x position
            yval[i][1] = float(yval_str[i*3+1]) # y position
            yval[i][2] = float(yval_str[i*3+2]) # angle
        
        ################# Write to the mechanical model #################
        assert len(V_muscle) == 12
        V_all = np.concatenate((V_muscle,-V_muscle))
        # print(frame,V_all)
        for i in range(self.NSEG*2):
            self.mech_model.stdin.write(str(V_all[i//4])) # 4 means the 4 segments per unit
            self.mech_model.stdin.write(' ')
        self.mech_model.stdin.write('\n')
        self.mech_model.stdin.flush()

    def follow_muscle_input(self,muscle_input,save_tempdata=False):
        # Muscle input is of shape (T,12)
        assert muscle_input.shape[1] == 12
        T = len(muscle_input)
        DV_XY = np.zeros((T,self.NBAR,4))
        yval_t = np.zeros((T,self.NBAR,3))
        k_t = np.zeros((T,self.NBAR-2))
        for t in range(T):
            self.update_worm(muscle_input[t])
            yval_t[t] = self.yval
            DX = yval_t[t,1:,0] - yval_t[t,:-1,0]
            DY = yval_t[t,1:,1] - yval_t[t,:-1,1]
            theta = np.arctan2(DY,DX)
            theta = np.unwrap(theta)
            k_t[t] = theta[1:] - theta[:-1]
            DorX,DorY,VenX,VenY = self.get_drosal_ventral_curve()
            DV_XY[t,:,0] = DorX
            DV_XY[t,:,1] = DorY
            DV_XY[t,:,2] = VenX
            DV_XY[t,:,3] = VenY
        if save_tempdata is True: 
            self.DV_XY = DV_XY
        return yval_t,k_t,DV_XY 
    
    def get_drosal_ventral_curve(self):
        yval = self.yval
        R = self.R
        Dorsal_X = yval[:,0] + R*np.cos(yval[:,2])
        Dorsal_Y = yval[:,1] + R*np.sin(yval[:,2])
        Ventral_X = yval[:,0] - R*np.cos(yval[:,2])
        Ventral_Y = yval[:,1] - R*np.sin(yval[:,2])
        return Dorsal_X,Dorsal_Y,Ventral_X,Ventral_Y
    
    def save_gif(self,name,dpi=150,size=3,interval=100,downsample=10):
        if not hasattr(self,'DV_XY'):
            raise Exception('No DV_XY data, please run follow_preset_head_input(save_tempdata=True) first')
        DV_XY = self.DV_XY
        T = DV_XY.shape[0]
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim([-size*1e-3,size*1e-3])
        ax.set_ylim([-size*1e-3,size*1e-3])
        line_dorsal, = ax.plot([],[],'k')
        line_ventral, = ax.plot([],[],'k')
        textb = ax.text(0,0, '')
        def anim_func(frame):
            line_dorsal.set_data(DV_XY[frame,:,0],DV_XY[frame,:,1])
            line_ventral.set_data(DV_XY[frame,:,2],DV_XY[frame,:,3])
            textb.set_text('frame:%d'%frame)
            return [line_dorsal,line_ventral,textb]
        anim = animation.FuncAnimation(fig,anim_func,frames=np.arange(0,T,downsample),interval=interval,blit=True)
        anim.save(name, writer='imagemagick')
        plt.close(fig)
        print('Gif saved to %s'%name)

    def close(self):
        os.killpg(os.getpgid(self.mech_model.pid), signal.SIGTERM)
        self.controller.close()

class Connect:
    def __init__(self,neurons,body,**kwargs):
        self.neurons = neurons
        self.body = body
        self.N_Input = body.N_Input
        self.N_Curv = body.N_Curv
        self.N_Segment = body.N_Segment
        self.Ma = np.zeros((self.N_Input,))
        # self.tau_m = kwargs.get('tau_m',0.1)
        # self.neuron2muscle = kwargs.get('neuron2muscle',self.get_neuron2muscle())
        self.curvature = np.zeros((self.N_Curv,))
        self.centerline = np.zeros((self.N_Segment+1,2))
        self.neuron_output = np.zeros((self.N_Input,))
        self.neurons.get_mat(self.N_Curv,self.N_Input//2)
  

    def update_worm(self,control):
        neuron_output = self.neurons.update_neuron(self.curvature,control) # neuron output number must match the number of muscles
        assert len(neuron_output) == self.N_Input # the first half of Input to the dorsal muscle of the body and the second half to the ventral muscle
        centerline,curvature = self.body.update_body(neuron_output) # centerline is of shape (N_Segment+1,2)
        self.curvature = curvature
        self.centerline = centerline
        self.neuron_output = neuron_output
    
    def run(self,time,control=None):
        T = len(time)
        centerline_t = np.zeros((self.N_Segment+1,2,T))
        curvature_t = np.zeros((self.N_Curv,T))
        neuron_output_t = np.zeros((self.N_Input,T))
        if control is not None:
            assert control.shape[0] == T
        for t in range(T):
            self.update_worm(control[t])
            centerline_t[:,:,t] = self.centerline
            curvature_t[:,t] = self.curvature
            neuron_output_t[:,t] = self.neuron_output

        return centerline_t,curvature_t,neuron_output_t
    
class CohenWorm:
    def __init__(self,model_path,**kwargs) -> None:
        self.N_Input = 12*2
        self.N_Segment = 48
        self.NBAR = self.N_Segment + 1  
        self.N_Curv = self.N_Segment 
        self.model_path = model_path
        self.dt = 0.01
        # Load the mechanical model
        self.mech_model = subprocess.Popen([self.model_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True)
        
        self.V = np.zeros(12) # Voltage of muscle
        self.yval = np.zeros((self.NBAR,3))
        self.D = 80e-6
        self.R = self.D/2.0*abs(np.sin(np.arccos((np.arange(self.NBAR)-self.N_Segment/2.0)/(self.N_Segment/2.0 + 0.2))))
    
    def update_body(self,V_muscle):
        # Update the voltage and the position of the worm
        try:
            from_model = self.mech_model.stdout.readline().strip()
            if from_model == "-1":
                pass
            yval_str, sr_str, neuron_str = from_model.split('\t')
            yval_str = yval_str.strip(' ,').split(',')
            yval = self.yval
            for i in range(self.NBAR):
                yval[i][0] = float(yval_str[i*3]) # x position
                yval[i][1] = float(yval_str[i*3+1]) # y position
                yval[i][2] = float(yval_str[i*3+2]) # angle
        except:
            warnings.warn('Mechanical model not responding, use the previous yval')
            yval = self.yval
        
        angle_unwrap = np.unwrap(yval[:,2])
        curvature = angle_unwrap[1:] - angle_unwrap[:-1]
        centerline = yval[:,:2]
        
        ################# Write to the mechanical model #################
        assert len(V_muscle) == self.N_Input
        # V_all = np.concatenate((V_muscle,-V_muscle))
        # print(frame,V_all)
        try:
            for i in range(self.N_Segment*2):
                self.mech_model.stdin.write(str(V_muscle[i//4])) # 4 means the 4 segments per unit
                self.mech_model.stdin.write(' ')
            self.mech_model.stdin.write('\n')
            self.mech_model.stdin.flush()
        except:
            raise Exception('Mechanical model not responding')
        return centerline,curvature
    
    def close(self):
        psutil.Process(self.mech_model.pid).kill()
        
#%%
if __name__ == '__main__':
    worm = CohenWorm('../WormSim/WormSim_RL_demo_3/Model/program_dur100s_dt10ms')
    T = 1000
    centerline_t = np.zeros((worm.N_Segment+1,2,T))
    curvature_t = np.zeros((worm.N_Segment,T))
    for t in range(T):
        x = np.linspace(0,1,12)
        input = np.concatenate((np.sin(x*2*np.pi-t*0.01*2*np.pi/3),-np.sin(x*2*np.pi-t*0.01*2*np.pi/3)))
        centerline,curvature = worm.update_body(input)
        centerline_t[:,:,t] = centerline
        curvature_t[:,t] = curvature
        



        
    


        

        
# %%
