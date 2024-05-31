'''
Visualize 2D or 3D trajectories using matplotlib and save as mp4 or gif

Example:

animate2D(x,y,save='test.mp4',fps=30,interval=10)
animate3D(x,y,z,save='test.mp4',fps=30,interval=10)
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from ipywidgets import interact

def animate2D(x,y,save=None,interval=10,dpi=200,extend=0):
    '''
    x and y are 1D arrays
    '''
    assert len(x)==len(y)
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(y),np.max(y))
    line_all, = ax.plot([],[],'g-',alpha=0.4)
    dot_t, = ax.plot([],[],'r.')
    line, = ax.plot([],[],'k')
    def init():
        line_all.set_data(x,y)
        dot_t.set_data([],[])
        line.set_data([],[])
        return line,dot_t,line_all,
    def animate(i):
        dot_t.set_data(x[i],y[i])
        if extend>0:
            line.set_data(x[max(0,i-extend):min(i+extend,len(x))],y[max(0,i-extend):min(i+extend,len(y))])
        return line,dot_t,line_all,
        
    ani = animation.FuncAnimation(fig,animate,init_func=init,frames=len(x),interval=interval,blit=True)
    if save is not None:
        ani.save(save)
        plt.close()
    else:
        plt.show()

def animate3D(x,y,z,save=None,fps=30,interval=10,views=None,dpi=200):
    '''
    x,y,z are 1D arrays
    '''
    assert len(x)==len(y)==len(z)
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(y),np.max(y))
    ax.set_zlim(np.min(z),np.max(z))
    if views is not None:
        ax.view_init(views[0],views[1])
    line, = ax.plot([],[],[],'k')
    def init():
        line.set_data([],[])
        line.set_3d_properties([])
        return line,
    def animate(i):
        line.set_data(x[:i],y[:i])
        line.set_3d_properties(z[:i])
        return line,
    ani = animation.FuncAnimation(fig,animate,init_func=init,frames=len(x),interval=interval,blit=True)
    if save is not None:
        ani.save(save,fps=fps)
    plt.close()

def plot_posture_head_curve(centerline,head_curv,orientation,i):
    '''
    Function to use with ipywedget interact; Plot the ith frame posture of the head and the head curve

    Parameters
    ----------
    centerline : 3D array
        Centerline of worm in the camera frame (n_segment,2,n_frame)
    head_curv : 1D array
        Head curvature time serie of the worm (n_frame,)
    i : int
        Frame number
    '''
    # plot in two subplots with two specfied figure size
    fig,ax = plt.subplots(4,1,figsize=(8,10),dpi=100)
    # plot the head curvature
    ax[0].hlines(0,0,len(head_curv))
    ax[0].plot(head_curv)
    ax[0].plot(i,head_curv[i],'ro')
    ax[0].set_xlabel('time index')
    ax[0].set_ylabel('head curvature')
    ax[0].set_ylim(-15,15)
    # positioning the plot in the upper 1/3 of the figure
    ax[0].set_position([0.1,0.78,0.8,0.2]) # [left,bottom,width,height]
    # plot orientation 
    ax[1].plot(orientation)
    ax[1].plot(i,orientation[i],'ro')
    ax[1].set_position([0.1,0.52,0.8,0.2]) # [left,bottom,width,height]
    # plot the centerline
    x_min = np.min(centerline[:,0,:])
    x_max = np.max(centerline[:,0,:])
    y_min = np.min(centerline[:,1,:])
    y_max = np.max(centerline[:,1,:])
    sidelen = max(x_max-x_min,y_max-y_min)
    ax[2].plot(centerline[:,0,i],centerline[:,1,i],'k')
    ax[2].plot(centerline[0,0,i],centerline[0,1,i],'ro')
    ax[2].set_xlim(x_min,x_min+sidelen)
    ax[2].set_ylim(y_min,y_min+sidelen)
    ax[2].invert_yaxi2()
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    # positioning the plot in the lower 2/3 of the figure, plot should be in square shape
    ax[2].set_position([0.1,0.1,0.4,0.4])
    # head 15% curvature
    head_posture = centerline[:int(0.15*len(centerline)),:,i] - centerline[int(0.15*len(centerline)),:,i]
    neck_theta = np.arctan2(centerline[16,1,i]-centerline[12,1,i],centerline[16,0,i]-centerline[12,0,i])
    # rotate the head posture so that the neck is always vertical, use numpy to do the rotation
    head_posture = np.dot(head_posture,-np.array([[np.sin(neck_theta),np.cos(neck_theta)],[-np.cos(neck_theta),np.sin(neck_theta)]]))
    ax[3].plot(head_posture[:,0],head_posture[:,1],'k')
    ax[3].plot(head_posture[0,0],head_posture[0,1],'ro')
    ax[3].set_xlim(-45,45)
    ax[3].set_ylim(-10,80)
    ax[3].set_position([0.6,0.1,0.4,0.4])


def plot_orientation_head_curve(centerline,head_curv,orientation,i):
    '''
    Function to use with ipywedget interact; Plot the head curvature trajectory, mean angle trajectory, centerline and COM

    Parameters
    ----------
    centerline : 3D array
        Centerline of worm in the camera frame (n_segment,2,n_frame)
    head_curv : 1D array
        Head curvature time serie of the worm (n_frame,)
    orientation : 1D arra
        
    i : int
        Frame number
    '''
    # plot in two subplots with two specfied figure size
    fig,ax = plt.subplots(4,1,figsize=(8,15),dpi=300)
    # plot the head curvature
    ax[0].plot(head_curv)
    # flip the negative part and set the positive part to zero
    head_curv_flip = -np.clip(head_curv,None,0)
    ax[0].plot(head_curv_flip,alpha=0.4)
    ax[0].plot(i,head_curv[i],'ro')
    ax[0].set_xlabel('time index')
    ax[0].set_ylabel('head curvature')
    # positioning the plot in the upper 1/3 of the figure
    ax[0].set_position([0.1,0.8,0.8,0.15])  # [left,bottom,width,height]
    # Plot the mean angle
    ax[1].plot(orientation)
    ax[1].plot(i,orientation[i],'ro')
    ax[1].set_xlabel('time index')
    ax[1].set_ylabel('mean angle')
    # positioning the plot in the middle 1/3 of the figure
    ax[1].set_position([0.1,0.6,0.8,0.15])  # [left,bottom,width,height]
    # plot the centerline
    x_min = np.min(centerline[:,0,:])
    x_max = np.max(centerline[:,0,:])
    y_min = np.min(centerline[:,1,:])
    y_max = np.max(centerline[:,1,:])
    sidelen = max(x_max-x_min,y_max-y_min)
    com_x = np.mean(centerline[:,0,:],axis=0)
    com_y = np.mean(centerline[:,1,:],axis=0)

    # Cumsum of the curvature
    head_curv_cumsum = np.cumsum(head_curv)
    ax[2].plot(head_curv_cumsum)
    ax[2].plot(i,head_curv_cumsum[i],'ro')
    ax[2].set_xlabel('time index')
    ax[2].set_ylabel('cumulative head curvature')
    ax[2].set_position([0.1,0.4,0.8,0.15])  # [left,bottom,width,height]

    ax[3].plot(centerline[:,0,i],centerline[:,1,i],'k')
    ax[3].plot(centerline[0,0,i],centerline[0,1,i],'ro')
    # plot a line represent the mean angle of centerline
    x_range = np.linspace(x_min,x_min+sidelen,100)
    ax[3].plot(x_range,com_y[i]+np.tan(orientation[i])*(x_range-com_x[i]),'b')
    ax[3].set_xlim(x_min,x_min+sidelen)
    ax[3].set_ylim(y_min,y_min+sidelen)
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('y')
    # positioning the plot in the lower 2/3 of the figure, plot should be in square shape
    ax[3].set_position([0.2,0.05,0.6,0.3]) # [left,bottom,width,height]

def plot_head_curve_eigenworm (centerline,head_curv,eigenworm,i):
    '''
    Function to use with ipywedget interact; Plot the ith frame posture, head curvature and eigenworm

    Parameters
    ----------
    centerline : 3D array
        Centerline of worm in the camera frame (n_segment,2,n_frame)
    head_curv : 1D array
        Head curvature time serie of the worm (n_frame,)
    eigenworm : 2D array
        First two components projection of eigenworm basis (n_frame,2)
    i : int
        Frame number
    '''
    # plot in two subplots with two specfied figure size
    fig,ax = plt.subplots(3,1,figsize=(8,8),dpi=100)
    # plot the head curvature
    ax[0].hlines(0,0,len(head_curv))
    ax[0].plot(head_curv)
    ax[0].plot(i,head_curv[i],'ro')
    ax[0].set_xlabel('time index')
    ax[0].set_ylabel('head curvature')
    ax[0].set_ylim(-15,15)
    # positioning the plot in the upper 1/3 of the figure
    ax[0].set_position([0.1,0.55,0.8,0.2]) 
    # plot the centerline
    x_min = np.min(centerline[:,0,:])
    x_max = np.max(centerline[:,0,:])
    y_min = np.min(centerline[:,1,:])
    y_max = np.max(centerline[:,1,:])
    sidelen = max(x_max-x_min,y_max-y_min)
    ax[1].plot(centerline[:,0,i],centerline[:,1,i],'k')
    ax[1].plot(centerline[0,0,i],centerline[0,1,i],'ro')
    ax[1].set_xlim(x_min,x_min+sidelen)
    ax[1].set_ylim(y_min,y_min+sidelen)
    ax[1].invert_yaxis()
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    # positioning the plot in the lower 2/3 of the figure, plot should be in square shape
    ax[1].set_position([0.1,0.1,0.4,0.4])
    
    # plot the eigenworm
    ax[2].plot(eigenworm[:,0],eigenworm[:,1],'k')
    ax[2].plot(eigenworm[i,0],eigenworm[i,1],'ro')
    ax[2].set_xlabel('eigenworm 1')
    ax[2].set_ylabel('eigenworm 2')
    ax[2].set_position([0.6,0.1,0.4,0.4])
    

def plot_centerline_interact(centerline):
    '''
    Function to use with ipywedget interact; Plot the ith frame posture of the worm

    Parameters
    ----------
    centerline : 3D array
        Centerline of worm in the camera frame (n_segment,2,n_frame)
    i : int
        Frame number
    '''
    def plot_centerline(i):
        fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=100)
        x_min = np.min(centerline[:,0,:])
        x_max = np.max(centerline[:,0,:])
        y_min = np.min(centerline[:,1,:])
        y_max = np.max(centerline[:,1,:])
        sidelen = max(x_max-x_min,y_max-y_min)
        ax.plot(centerline[:,0,i],centerline[:,1,i],'k')
        ax.plot(centerline[0,0,i],centerline[0,1,i],'ro')
        ax.set_xlim(x_min,x_min+sidelen)
        ax.set_ylim(y_min,y_min+sidelen)
        ax.invert_yaxis()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #eqaul aspect ratio
        ax.set_aspect('equal')
        plt.show()
    interact(plot_centerline,i=(0,centerline.shape[2]-1,1))
    