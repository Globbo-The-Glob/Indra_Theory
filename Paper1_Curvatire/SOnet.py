import numpy as np
import networkx as nx
import scipy.integrate as odeint
import scipy as sp
from functions import *
from functions import RK4Coeffs
import matplotlib as mpl
import matplotlib.pyplot as plt
import math 
import csv
import colour 


'''

MAIN  CLASS

'''


class Net:
    def __init__(self,name,Adj,n,k,dt,var,mean,peaks_phase,means_phase,peaks_freq,means_freq):
        self.Adj = Adj#Adjacency with weighted and dircetional couplings
        self.n = n
        self.dt = dt
        self.k = float(k)
        self.coupling = self.k/self.n
        self.nat_hist = []
        self.noise_A = 0.1
        self.space = [] #3 dimensional vectors in node based vector. 
        # NEEDS GENERATOR
        if sum(means_phase) == 0:    
            means_phase = np.random.uniform(-np.pi,np.pi,(peaks_phase,1))
        i = 1
        self.phase = np.zeros((n,1))
        nn = int(np.ceil(n/peaks_phase))
       
        i = 0
        for kk in range(0,peaks_phase):
            # distribute a proportion of oscillators about mean
            a = 0 + nn*i
            b = nn + nn*i
            self.phase[a:b] = np.random.normal(means_phase[i],0.01,(nn,1))
            i += 1
        
        #self.state = 0 + self.noise_A*np.random.randn(self.n,1)#np.random.uniform(-np.pi,np.pi,(n,1))#0 + self.noise_A*np.random.randn(self.n,1) # # MAJOR DIFFERENCE BETWEEN INTERESTING AND NOT# 0 + self.noise_A*np.random.randn(self.n,1) #np.random.uniform(-np.pi,np.pi,(n,1))  # nx1 vector of current oscillator  states
        self.phase_init = self.phase
        #print('init state = {}'.format(self.state))
        self.nats = 0.1*np.ones((n,1))+ self.noise_A*0.01*np.random.randn(self.n,1) #np.random.uniform(-2*np.pi,2*np.pi,(n,1))#np.zeros((n,1))+ self.noise_A*0.1*np.random.randn(self.n,1) #np.array([0.1, -0.1])#   #dt*np.random.uniform(-2*np.pi,2*np.pi,(n,1)) #2*np.pi*dt*np.random.normal(mean,var,(n,1)) #  np.zeros((n,1)) + self.noise_A*0.01*np.random.randn(self.n,1) # 2*np.pi*dt*np.random.normal(mean,var,(n,1)) # Vector of natural freqs, currently all unity for simplicity. 
        self.state_nats = self.nats 
        #print('init nats = {}'.format(self.nats))
        self.I = np.zeros((n,1))
        self.Iw = np.zeros((n,1))
        self.graph = nx.from_numpy_array(self.Adj)
        self.pos = nx.kamada_kawai_layout(self.graph)
        self.nat_hist = self.nats
        self.phase_hist = self.phase
        self.name = name
        self.state = np.block([[self.phase], [self.nats]])
    
    def Update(self,T,res):
        
        # This version does the whole net in one go
        # State vector is [phase,nat]^T
        # Equation becomes [phase,nat]^T_dt = [phase,nat]^T_0 + [nat,del_nat]^T
        dt = self.dt # Iteration
        Adj = self.Adj
        state = self.state
        n = self.n
        # RK4 coeffs
        k1 = RK4Coeffs(Adj,n,state,0)
        k2 = RK4Coeffs(Adj,n,state+k1*(dt/2),dt/2)
        k3 = RK4Coeffs(Adj,n,state+k2*(dt/2),dt/2)
        k4 = RK4Coeffs(Adj,n,state+k3*dt,dt)
        # RK4 baby!
        self.state = state + dt*k4 # + dt / 6 * ( k1 + 2 * k2 + 2 * k3 + 
        
        
        # for i in range(self.n):
        #     I_n = 0
        #     I_f = 0
        #     for j in range(self.n):
        #         ## SECOND ORDER UPDATING
                
        #         if self.Adj[i,j] != 0: # if not itself
        #             I_f += self.Adj[i,j]*np.sin(self.state[j] - self.state[i])*self.dt # np.sin
        #     self.I[i] = I_n
        #     self.Iw[i] = I_f #check change!
    
        #     self.nats +=  self.Iw # *1/self.n 
        #     self.state += self.nats + self.I
            
        
        self.phase_hist = np.append(self.phase_hist,self.state[0:n],1)
        self.nat_hist= np.append(self.nat_hist,self.state[n:],1)
       
     
    def reinit(self):
        self.nats = self.state_nats  
        self.state = self.state_init 
        self.nat_hist = self.nats
        
    def Gauss_Space(self,space_mean,space_var):

        self.space = np.random.normal(space_mean,space_var,(self.n,2))
        self.pos = self.space
 
    def distribute_adj(self,distribution):
        if distribution == 'uni':
            self.Adj = self.Adj * np.random.uniform(-1,1,self.Adj.shape)

    def R2Connect(self):
        for i in range(self.n-1):
            for j in range(i+1,self.n): 
                    if j != i: 
                        xi = self.space[i,:]
                        xj = self.space[j,:]
                        r2 = (np.linalg.norm(xi-xj))**2
                        self.Adj[i,j] = self.k/(1 + r2)
                        self.Adj[j,i] = self.k/(1 + r2)
                        
    def ProbConnect(self,a,P_inhib):
        # -a is power law exponent
        for i in range(self.n-1):
            for j in range(i+1,self.n): 
                if j != i:
                    pool = np.zeros([10000,1])
                    inhib_pool = np.ones([10000,1])
                    xi = self.space[i,:]
                    xj = self.space[j,:]
                    s = np.abs(xi-xj)
                    mag = np.sqrt(s[0]**2 + s[1]**2)# + s[2]**2 )
                    prob_connect = 1/(mag**a+1) # connection law. +1 is offset to normalise ## CHANGE to control connectivity
                    connect_num = np.round(len(pool)*prob_connect)
                    pool[1:int(connect_num)] = 1
                    np.random.shuffle(pool)
                    inhib_num = np.round(len(pool)*P_inhib)
                    pool[1:int(inhib_num)] = -1
                    np.random.shuffle(inhib_pool)
                    self.Adj[i,j] = (1/mag)*pool[np.random.randint(0,len(pool))]*inhib_pool[np.random.randint(0,len(pool))]
                    self.Adj[j,i] = (1/mag)*pool[np.random.randint(0,len(pool))]*inhib_pool[np.random.randint(0,len(pool))]
                else:
                    continue
    
    def Mesh(self):
        
        x = np.linspace(0,100,self.rtn)
        y = np.linspace(0,100,self.rtn)
        xv,yv = np.meshgrid(x,y,indexing = 'ij')
        
        # Build Adj for grid 
        # Check all 4 directions
        self.Adj = np.zeros((self.rtn,self.rtn)) # RESET ADJ
        for i in range(self.rtn):
            for j in range(self.rtn):
                self.pos[i,j] = [xv[i,j],yv[i,j]] # update position for plotting
                if i - 1 >= 0:
                    self.Adj[i-1,j] = self.k
                if i + 1 < self.rtn: 
                    self.Adj[i+1,j] = self.k
                if j - 1 >= 0:
                    self.Adj[i,j-1] = self.k  
                if j + 1 < self.rtn: 
                    self.Adj[i,j+1] = self.k
        print(f"Grid{self.Adj}")                 
        # Couple if exist. 
        
    def View(self):
        self.graph = nx.from_numpy_array(self.Adj)
        #edge colours should be realted to normalised weights 
        
        #node colours should be current phase
        color_map = []
        # for i in range(self.n):
        #     color_map.append(cyclic_rgb(self.state[i]))
        # print(color_map)  
        Order = np.arange(0, self.n, 2)
        Order = np.hstack((Order, np.arange(self.n-1, 0, -2)))
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        ax = axs[0]
        # colormap(jet)
        ax.pcolormesh(self.Adj[Order, :][:, Order],cmap=mpl.colormaps['Spectral'])

        ax.set_xlabel('Nodes')
        ax.set_ylabel('Nodes')
        ax.set_title('Coupling Matrix')
        ax.axis('square')
        # red = Color("red")
        # colors = list(red.range_to(Color("green"),10))
        #nx.draw(self.graph,node_color=color_map,pos=self.pos) # BROKEN WITH EVOLUTIONS 
        
        
        ax = axs[1]
        # normalise weights
        norm_adj = np.zeros_like(self.Adj) 
        signed_norm_adj = np.zeros_like(self.Adj) 
        for i in range(0,self.n):
            for j in range(0,self.n):    
                norm_adj[i,j] = (abs(self.Adj[i,j]) - abs(self.Adj).min())/(abs(self.Adj).max() - abs(self.Adj).min())*(1)
            signed_norm_adj = norm_adj*np.sign(self.Adj)
        #print(norm_adj)
        for i in range(0,self.n):
            
            for j in range(i,self.n):
                if self.Adj[i,j] != 0:
                    #print(rgb_spectrum_line(norm_adj[i,j]))
                
                    plt.plot([self.pos[i][0],self.pos[j][0]],[self.pos[i][1],self.pos[j][1]],linestyle = '-',color = rgb_spectrum_line(signed_norm_adj[i,j]),alpha = norm_adj[i,j])
            plt.scatter(self.pos[i][0],self.pos[i][1])