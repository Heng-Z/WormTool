import numpy as np
import scipy.optimize

class ResisForce:
    def __init__(self):
        self.K = 40
        self.Dx = None
        self.segment_length = 1

    def dcurv_force_torque(self,curv,dcurv,w0):
        '''
        By solving the force and torque balance given the curvature and the time derivative of the curvature, 
        this function returns the force environment excerts on the head and the internal torque the body muscle has to provide.

        Parameters
        ----------
        curv : array: (L,)
            Curvature of the head.
        dcurv : array (L,)
            Time derivative of the curvature.
        w : array of shape (3,)
            Initial point for solving the velocity of angular velocity of the center of mass of the worm. Typically use the solution from previous time step.
        
        Returns
        -------
        torque : array of shape (L,)
            Body torque the worm has to provide.
        Fn : array of shape (L,)
            Normal force the environment excerts on the worm
        Fl : array of shape (L,)
            Longitudinal force the environment excerts on the worm
        w : array of shape (3,)
            Velocity of angular velocity of the center of mass of the worm.

        TODO
        ----
        1, Remove redundant calculations.
        '''
        def total_torque_force(w):
            comvx = w[0]
            comvy = w[1]
            ome = w[2] # angular velocity with respect to the head 
            K = self.K
            seg_len = self.segment_length
            L = len(curv)
            assert len(dcurv) == L
            if self.Dx is None:
                Dx = 1/L
            else:
                Dx = self.Dx
            oangle = np.cumsum(curv)*Dx
            xx = np.cumsum(np.cos(oangle))*Dx
            yy = np.cumsum(np.sin(oangle))*Dx

            vvx = np.cumsum(-np.sin(oangle)*Dx*np.cumsum(dcurv)*Dx)
            vvy = np.cumsum(np.cos(oangle)*Dx*np.cumsum(dcurv)*Dx)
            vvx = vvx - ome*yy + comvx
            vvy = vvy + ome*xx + comvy

            mid_angle = oangle
            vl = np.cos(mid_angle)*vvx + np.sin(mid_angle)*vvy
            vn = -np.sin(mid_angle)*vvx + np.cos(mid_angle)*vvy
            Fn = -vn
            Fl = -vl/K
            Fx = Fl*np.cos(mid_angle) - Fn*np.sin(mid_angle)
            Fy = Fl*np.sin(mid_angle) + Fn*np.cos(mid_angle)
            # calculate the torque and force
            torque_sum = np.sum(xx*Fy-yy*Fx)*Dx
            xforce_sum = Dx*L*np.sum(Fx)
            yforce_sum = Dx*L*np.sum(Fy)

            return [torque_sum,xforce_sum,yforce_sum]
        
        def segment_torque_force(w):
            comvx = w[0]
            comvy = w[1]
            ome = w[2]
            K = self.K
            seg_len = self.segment_length
            L = len(curv)
            assert len(dcurv) == L
            if self.Dx is None:
                Dx = 1/L
            else:
                Dx = self.Dx
            oangle = np.cumsum(curv)*Dx
            xx = np.cumsum(np.cos(oangle))*Dx
            yy = np.cumsum(np.sin(oangle))*Dx

            vvx = np.cumsum(-np.sin(oangle)*Dx*np.cumsum(dcurv)*Dx)
            vvy = np.cumsum(np.cos(oangle)*Dx*np.cumsum(dcurv)*Dx)
            vvx = vvx - ome*yy + comvx
            vvy = vvy + ome*xx + comvy

            mid_angle = oangle
            vl = np.cos(mid_angle)*vvx + np.sin(mid_angle)*vvy
            vn = -np.sin(mid_angle)*vvx + np.cos(mid_angle)*vvy
            Fn = -vn
            Fl = -vl/K
            Fx = Fl*np.cos(mid_angle) - Fn*np.sin(mid_angle)
            Fy = Fl*np.sin(mid_angle) + Fn*np.cos(mid_angle)
            # calculate the torque 
            torque = np.zeros(L)
            for i in range(L):
                torque[i] = np.sum((xx[i:]-xx[i])*Fy[i:] - (yy[i:]-yy[i])*Fx[i:])*Dx
            return [torque,Fn,Fl]


        # solve the torque and force balance

        w,_,flag,_ = scipy.optimize.fsolve(total_torque_force,w0,xtol=1e-4,maxfev=10000,full_output=True)
        if flag != 1:
            print('Torque and force balance not solved')
            w = w0
        tor,Fn,Fl = segment_torque_force(w)
        return tor,Fn,Fl,w
    
        
    def curv2force(self,curv,dcurv):
        '''
        Solve force and torque timeseries given curvature and time derivative of curvature.
        
        Parameters
        ----------
        curv : array: (L,T)
            Curvature of the head.
        dcurv : array (L,T)
            Time derivative of the curvature.

        Returns
        -------
        fn : array of shape (L,T)
            Normal force the environment excerts on the worm
        fl : array of shape (L,T)
            Longitudinal force the environment excerts on the worm
        torque : array of shape (L,T)  
            Body torque the worm has to provide.
        w : array of shape (3,T)
            Velocity of angular velocity of the center of mass of the worm.
        
        '''
        L,T = curv.shape
        fn = np.zeros((L,T))
        fl = np.zeros((L,T))
        torque = np.zeros((L,T))
        w = np.zeros((3,T))
        w0 = np.zeros(3)
        for i in range(T):
            tor,Fn,Fl,w[:,i] = self.dcurv_force_torque(curv[:,i],dcurv[:,i],w0)
            fn[:,i] = Fn
            fl[:,i] = Fl
            torque[:,i] = tor
            w0 = w[:,i]
        return fn,fl,torque,w

    def curv2centerline(self,curv,dcurve,DX,DT,downsample=1):
        '''
        Reconstruct the centerline trajectory in the lab frame.

        Parameters
        ----------
        curv : array of shape (L,T)
            Curvature of the worm.
        dcurv : array of shape (L,T)
            Time derivative of the curvature.
        DX : float
            Segment width of the centerline.
        DT : float
            Time step of the centerline.
        downsample : int, optional
            Downsample the centerline in time. The default is 1.
        
        Returns
        -------
        centerline : array of shape (L,2,T)
            Centerline trajectory in the lab frame.
        '''
        _,_,_,w = self.curv2force(curv,dcurve)
        orientation = np.cumsum(w[2,:])*DT 
        dispx = np.cumsum(w[0,:]*np.cos(orientation) - w[1,:]*np.sin(orientation))*DT
        dispy = np.cumsum(w[0,:]*np.sin(orientation) + w[1,:]*np.cos(orientation))*DT
        L,T = curv.shape
        centerline = np.zeros((L,2,T))
        for i in range(T):
            oangel = np.cumsum(curv[:,i])*DX
            xx = np.cumsum(np.cos(oangel))*DX
            yy = np.cumsum(np.sin(oangel))*DX
            x_lab = xx*np.cos(orientation[i]) - yy*np.sin(orientation[i]) + dispx[i]
            y_lab = xx*np.sin(orientation[i]) + yy*np.cos(orientation[i]) + dispy[i]
            centerline[:,:,i] = np.vstack((x_lab,y_lab)).T
        return centerline[:,:,::downsample]
    
    



        





