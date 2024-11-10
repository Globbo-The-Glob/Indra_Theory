import numpy as np
import networkx as nx
import scipy.integrate as odeint
import scipy as sp
# from SOnet import Net
import matplotlib as mpl
import matplotlib.pyplot as plt
import math 
import csv
import colour 
import torch

def rgb_spectrum_line(value): # NEEDS OPACITY AND TUNING 
    if value == 0: 
        rgb = np.array([1.0,1.0,1.0]) # white 
    elif value < 0:
        rgb = abs(value)*np.array([1.0,0.0,0.0]) # yellow if < 0 
    else:
        rgb = abs(value)*np.array([0.0,0.0,1.0]) # blue >0
    return rgb
 
def RK4Coeffs(Adj,n,state):
    deltaPhase = np.dot(Adj,state[0:n]) - state[0:n]# Dot adj w/ state vector (Only using first half of state vector for phase)
    deltaNat = 1/n*np.sin(deltaPhase) # Kuramoto update rule for natural freq change
    state[0:n] = state[0:n] + state[n:] + deltaNat# Add nats to phase 
    state[n:] = state[n:] + deltaNat # Add acc to nats
    return state 
        
def RunAndPlot(time,net,T,res,N,k,var,mean,connectivity):
    results = np.zeros((res,N))
    theta  = np.zeros((res,N))
    for i in range(0,len(time)):
        theta[i,:] = net.state[0:net.n,0]
        results[i,:] = np.sin(net.state[0:net.n,0]) 

        net.Update(T,res)
    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(time,results[:,i],color='black', alpha=1/net.n) 
        
    ax.set_ylim([-1, 1.1])

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(r"sin $\theta $")
    ax.set_title(f'k = {k}, Var = {var}, Mean = {mean} , P = {connectivity} ,N = {N}')
    fig, ax = plt.subplots()
    # print(np.exp(1j*results))
    # print(np.mean(np.exp(1j*results)))
    OP = np.abs(np.mean(np.exp(1j*theta), axis=1))
    ax.plot(time, OP)
    ax.set_ylim([-0, 2])


    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Order Parameter')

    fig, ax = plt.subplots()
    for i in range(0,net.n):
        ax.plot(time,net.nat_hist[i][0:-1])
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(r"Natural Frequencies")
    ax.set_title(f'k = {k}, Var = {var}, Mean = {mean} , P = {connectivity} ,N = {N}')
    
    # fig, ax = plt.subplots()
    # ax = plt.axes(projection='3d')
    # for i in range(0,N):
    #     ax.plot3D(np.cos(theta[:,i]), np.sin(theta[:,i]), time, 'black',alpha = 1/net.n)
    return results, theta

def Load(dir,name):
    file = dir + name
    f = open(file,'w')
    csv_read = csv.reader(file,delimiter=',')
    i = 0 
    netty = Net(name,0,0,0,0,0,0)
    states = []
    nats = [] 
    adj_vec = []
    details = []
    for row in csv_read:
        states[i] = row[0]
        nats[i] = row[1]
        adj_vec[i] = row[2]
        details[i] = row[3]
        i += 1
        