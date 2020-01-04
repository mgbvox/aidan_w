import numpy as np
import stmpy
import stmpy.driftcorr as dfc
import scipy as sp
from numpy import *
import random as rd
import matplotlib.pyplot as plt

def quick_show_single(A, en, thres=5, rspace=False, saveon=False, imgName=''):
    # to fold
    # Quick show single images  
    layers = len(A)
    if rspace is False:
        imgsize = np.shape(A)[-1]
        bp_x = np.min(dfc.findBraggs(np.mean(A, axis=0), min_dist=int(imgsize/10), rspace=rspace))
        ext = imgsize / (imgsize - 2*bp_x)
    
    for i in range(layers):
        plt.figure(figsize=[4,4])
        c = np.mean(A[i])
        s = np.std(A[i])
        if rspace is True:
            plt.imshow(A[i], clim=[c-thres*s,c+thres*s],cmap=stmpy.cm.jackyPSD)
        else:
            plt.imshow(A[i],extent=[-ext,ext,-ext,ext,],clim=[0,c+thres*s],cmap=stmpy.cm.gray_r)
            plt.xlim(-1.2,1.2)
            plt.ylim(-1.2,1.2)
        stmpy.image.add_label("${}$mV".format(int(en[i])), ax=plt.gca())
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().set_frame_on(False)
        if saveon is True:
            plt.savefig("{} at ${}$mV.png".format(imgName, int(en[i])),dpi=200, bbox_inches='tight',pad_inches=0)

# quick_display function
def quick_display(A, sigma=3):
    A_fft = stmpy.tools.fft(A, zeroDC=True)
    L = np.shape(A)[-1]
    c = np.mean(A_fft)
    s = np.std(A_fft)
    fig,ax=plt.subplots(2,2,figsize=[8, 8])
    ax[0, 0].imshow(A, cmap=stmpy.cm.blue2, origin='lower')
    ax[0, 0].axhline(L/2, color='r', linestyle='--')
    ax[0, 1].imshow(A_fft, cmap=stmpy.cm.gray_r, origin='lower', clim=[0,c+sigma*s])
    ax[0, 1].axhline(L/2, color='b', linestyle='--')
    ax[1, 0].plot(A[int(L/2)], 'r')
    ax[1, 0].plot(A[int(L/2)], 'rx', ms=4, mew=1)   
    ax[1, 1].plot(A_fft[int(L/2)], 'b')
    ax[1, 1].plot(A_fft[int(L/2)], 'bx', ms=4, mew=1) 

# generate impurity function
def generate_map(size, L, a0):
    s = np.linspace(0, size, L)
    x, y = np.meshgrid(s[:-1], s[:-1])
    k0 = 2*np.pi/a0
    z0 = np.cos(k0 * x) * np.cos(k0 * y)
    return z0

# add impurity function
def add_imp(A, size, center=None, decay=None, a=None, ratio=None, itype='0', flip=False, activate=True):
    if activate==False:
        return A
    
    L = shape(A)[-1]
    s = linspace(0, size, L)
    x, y = meshgrid(s, s)
    if itype == '0':
        k = 2*pi / a
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        if flip:
            z1 = -exp(-decay * r) * sin(k * r)
        else: 
            z1 = exp(-decay * r) * sin(k * r)
        return A + z1
    elif itype == '1':
        k = 2*pi / a
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        r_x = sqrt(ratio*(x-center[0])**2+(y-center[1])**2)
        r_y = sqrt((x-center[0])**2+ratio*(y-center[1])**2)
        if flip:
            z2 = -((exp(-decay * r) * sin(k * r_x) +\
             exp(-decay * r) * sin(k * r_y)) / 2)
        else:
            z2 = (exp(-decay * r) * sin(k * r_x) +\
             exp(-decay * r) * sin(k * r_y)) / 2
        return A + z2
    elif itype == '2':
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        if flip:
            z3 = -exp(-decay * r**2)
        else:
            z3 = exp(-decay * r**2)
        return A + z3
    elif itype == '3':
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        r_x = sqrt(ratio*(x-center[0])**2+(y-center[1])**2)
        r_y = sqrt((x-center[0])**2+ratio*(y-center[1])**2)
        if flip:
            z4 = -((exp(-decay * r_x) +\
             exp(-decay * r_y)) / 2)
        else:
            z4 = (exp(-decay * r_x) +\
                 exp(-decay * r_y)) / 2
        return A + z4
    elif itype == '4': #Decay constant around 0.5
        rx = x-center[0]
        ry = y-center[1]
        r=sqrt(rx**2+ry**2)
        if flip:
            z5 = -exp(-decay*(rx*ry)**2)*exp(-decay*r)
        else:
            z5 = exp(-decay*(rx*ry)**2)*exp(-decay*r)
        return A + z5
    elif itype == '5':      #Adds a very faint outline; would be used to add another band to an impurity
                            #Decay constant around 0.3
        rx = x-center[0]
        ry = y-center[1]
        r=sqrt(rx**2+ry**2)
        if flip:
            z6 = -exp(-decay*(rx*ry)**2)*sin((rx*ry)**2)*exp(-decay*r)
        else:
            z6 = exp(-decay*(rx*ry)**2)*sin((rx*ry)**2)*exp(-decay*r)
        return A + z6
    elif itype == '6':      #Decay constant around 0.1
        rx = x-center[0]
        ry = y-center[1]
        r=sqrt(rx**2+ry**2)
        if flip:
            z7 = -exp(-decay*(rx**6+ry**6))*sin(r**2)
        else:
            z7 = exp(-decay*(rx**6+ry**6))*sin(r**2)
        return A + z7
    else:
        print("Error: please input itype as one of '0', '1', '2', '3'.")
        
def generate_image():
    
    import random as rd
    
    z0 = generate_map(50, 320, 1)
    
    bboxes = []
    
    points = []
    widths=[]
    gamma=320/50
    
    num = rd.randint(1,7) #This can be randomly selected. 
    pts = np.zeros((num, 2))

    x0 = random.random(num) * 50
    y0 = random.random(num) * 50

    #Rescale points to record.
    for x in range(num):

        pts[x][0]=x0[x]*gamma-x0[x]//(20)
        pts[x][1]=y0[x]*gamma-y0[x]//(20)

    #Record centers.
    points.append(pts)

    #grab lattice
    z1=copy(z0)
    zm=copy(z0)

    #Stores widths for bboxes for image n.
    wtemp=[]
    #num = n_impurities
    for i in range(num):
        decayconstants = [0]*7
        decayconstants[0]=random.randint(20,80)/100 #Decay constant for type 0 impurity
        decayconstants[1]=random.randint(20,80)/100 
        decayconstants[2]=random.randint(20,80)/100
        decayconstants[3]=random.randint(20,80)/100
        decayconstants[4]=random.randint(20,80)/100
        decayconstants[5]=random.randint(20,40)/100
        decayconstants[6]=random.randint(80,120)/1000

        '''
        #Future random point generation: data augmentation.
        for tpe in np.random.randint(0,4,(4,)):
            z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=d1, a=10, ratio=5, itype=f'{tpe}')
        '''
        
        
        
        flip = bool(rd.getrandbits(1)) #Whether or not to flip the impurity (applied to ALL impurity 
                                       #types as long as they are in one location)
        
        actbools = [] #Randomly determine which impurities we want to activate
        for a in range(7):
            val = bool(rd.getrandbits(1))
            actbools.append(val)
            if val == False:
                decayconstants[a] = 0 # If there is no impurity, do not contribute to bbox calculation
                
        # Add circular QPI signal
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[0], a=10, ratio=5, itype='0',
                     flip=flip, activate=actbools[0])
        
        # Add splitting 4-fold symmetric QPI signal
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[1], a=4, ratio=5, itype='1'
                     ,flip=flip,activate=actbools[1]) 
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[1], a=3.2, ratio=5, itype='1',
                     flip=flip, activate=actbools[1]) 
        
        # Add 4-fold symmetric impurity center
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[3], a=10, ratio=5, itype='3', 
                     flip=flip, activate=actbools[3])
        
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[4], itype='4', 
                     flip=flip, activate=actbools[4])
    
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[5], itype='5', 
                     flip=flip, activate=actbools[5])
        
        z1 = add_imp(z1, 50, center=[x0[i], y0[i]], decay=decayconstants[6], itype='6', 
                     flip=flip, activate=actbools[6])
        
        alpha=0.08
        
        radii = [0]*7
        if decayconstants[0]==0:
            radii[0]=0
        else:
            radii[0] = -log(alpha)/decayconstants[0]
            
        if decayconstants[1]==0:
            radii[1]=0
        else:
            radii[1] = -log(alpha)/decayconstants[1] 
            
        if decayconstants[2]==0:
            radii[2]=0
        else:
            radii[2] = -log(alpha)/decayconstants[2]
            
        if decayconstants[4]==0:
            radii[4]=0
        else:
            radii[4] = -log(alpha)/decayconstants[4]
            
        if decayconstants[5]==0:
            radii[5]=0
        else:
            radii[5] = -log(alpha)/decayconstants[5]
    
        if decayconstants[6]==0:
            radii[6]=0
        else:
            radii[6] = (-log(alpha)/decayconstants[6])**(1/6)
        
        #print(radii)
    
        w=max(radii)*gamma
        wtemp.append(w)

    # Add overall noise
    noise = random.random(shape(z1)) * 0.5
    z1 = z1 + noise
    
    z1 = np.interp(z1, (z1.min(), z1.max()), (0, 255))
    widths.append(wtemp)
    
    return z1, np.stack(points, axis=0)[0], np.stack(widths, axis=0).ravel()

def get_bboxes(p, w):
    bboxes = []
    for i,c in enumerate(p):
        xmin = max(0,c[0]-w[i])   #ReLU; sets any negative coordinates to 0.
        xmax = min(319,c[0]+w[i]) #If coordinate is greater than image size, sets value to end of image.
        ymin = max(0,c[1]-w[i])
        ymax = min(319,c[1]+w[i])
        box = [xmin, xmax, ymin, ymax]
        bboxes.append(box)
    return np.array(bboxes)


def get_imp(A, size, center=None, decay=None, a=None, ratio=None, itype='0', flip=False, activate=True):
    if activate==False:
        return A
    
    L = shape(A)[-1]
    s = linspace(0, size, L)
    x, y = meshgrid(s, s)
    if itype == '0':
        k = 2*pi / a
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        if flip:
            z1 = -exp(-decay * r) * sin(k * r)
        else: 
            z1 = exp(-decay * r) * sin(k * r)
        return (A + z1, z1)
    elif itype == '1':
        k = 2*pi / a
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        r_x = sqrt(ratio*(x-center[0])**2+(y-center[1])**2)
        r_y = sqrt((x-center[0])**2+ratio*(y-center[1])**2)
        if flip:
            z2 = -((exp(-decay * r) * sin(k * r_x) +\
             exp(-decay * r) * sin(k * r_y)) / 2)
        else:
            z2 = (exp(-decay * r) * sin(k * r_x) +\
             exp(-decay * r) * sin(k * r_y)) / 2
        return (A + z2, z2)
    elif itype == '2':
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        if flip:
            z3 = -exp(-decay * r**2)
        else:
            z3 = exp(-decay * r**2)
        return (A + z3, z3)
    elif itype == '3':
        r = sqrt((x-center[0])**2+(y-center[1])**2)
        r_x = sqrt(ratio*(x-center[0])**2+(y-center[1])**2)
        r_y = sqrt((x-center[0])**2+ratio*(y-center[1])**2)
        if flip:
            z4 = -((exp(-decay * r_x) +\
             exp(-decay * r_y)) / 2)
        else:
            z4 = (exp(-decay * r_x) +\
                 exp(-decay * r_y)) / 2
        return (A + z4, z4)
    
    elif itype == '4': #Decay constant around 0.5
        rx = x-center[0]
        ry = y-center[1]
        r=sqrt(rx**2+ry**2)
        if flip:
            z5 = -exp(-decay*(rx*ry)**2)*exp(-decay*r)
        else:
            z5 = exp(-decay*(rx*ry)**2)*exp(-decay*r)
        return (A + z5, z5)
    elif itype == '5':      #Adds a very faint outline; would be used to add another band to an impurity
                            #Decay constant around 0.3
        rx = x-center[0]
        ry = y-center[1]
        r=sqrt(rx**2+ry**2)
        if flip:
            z6 = -exp(-decay*(rx*ry)**2)*sin((rx*ry)**2)*exp(-decay*r)
        else:
            z6 = exp(-decay*(rx*ry)**2)*sin((rx*ry)**2)*exp(-decay*r)
        return (A + z6, z6)
    elif itype == '6':      #Decay constant around 0.1
        rx = x-center[0]
        ry = y-center[1]
        r=sqrt(rx**2+ry**2)
        if flip:
            z7 = -exp(-decay*(rx**6+ry**6))*sin(r**2)
        else:
            z7 = exp(-decay*(rx**6+ry**6))*sin(r**2)
        return (A + z7, z7)
    else:
        print("Error: please input itype as one of '0', '1', '2', '3','4','5','6', or '7'.")