from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si

class MFW_plot:
    '''Create a simulation environment to track a simulated trading strategy'''
    def __init__(self):
        self.timeSteps = 100
        self.T = 1
        self.S0 = 1.
        self.delta_0 = 0
        self.B0 = 10
        self.realDataPaths=20
        
        realDataTemp = np.fromfile("Data/testData.txt")
        self.realData = np.zeros((self.realDataPaths,self.timeSteps+1))
        for i in range(self.realDataPaths):
            if i==0:
                self.realData[i] = realDataTemp[0:101]/realDataTemp[0]
            else:
                self.realData[i] = realDataTemp[i*100-1:(i+1)*100]/realDataTemp[i*100-1]
        
    def stockPath(self,mu,sigma):
        St = np.zeros(self.timeSteps+1)
        dt = float(self.T) / float(self.timeSteps)
        St[0] = self.S0
        for i in range(self.timeSteps):
            St[i+1] = St[i] * np.exp( (mu - 0.5*sigma*sigma) *dt + np.sqrt(dt)*sigma*np.random.normal() )
        return St
    
    def f_plot(self,mu, sigma):
        fig = plt.figure(figsize=(16, 12), dpi=80)
        ax1 = plt.subplot(2,2,1 )
        ax2 = plt.subplot(2,2,2 )
        ax3 = plt.subplot(2,2,3 )
        ax4 = plt.subplot(2,2,4 )
        # get the time axes for the bottom
        t = np.linspace(0, self.T, num=(self.timeSteps+1))
        
        dS_real = np.zeros(self.timeSteps*self.realDataPaths)
        dS_model = np.zeros(self.timeSteps*self.realDataPaths)
        
        for i in range(self.realDataPaths):
            St = self.realData[i]
            for j in range(self.timeSteps):
                dS_real[i*self.timeSteps + j] = np.log(St[j+1]) - np.log(St[j])
            # generate a plot of the stock price 
            ax1.plot(t, St )
            St = self.stockPath(mu,sigma)
            for j in range(self.timeSteps):
                dS_model[i*self.timeSteps + j] = np.log(St[j+1]) - np.log(St[j])
            # generate a plot of the stock price 
            ax2.plot(t, St )
        my_bins = np.linspace(-0.05, 0.05,40 )
        ax3.hist(dS_real,bins=my_bins)
        ax4.hist(dS_model,bins=my_bins)
        plt.show()

    def f_plot_interactive(self):
        return interact_manual(self.f_plot, mu=(-0.1, 0.1 , 0.01), sigma=(0., 0.5, 0.005))

    def plotRealData(self):
        realDataTemp = np.fromfile("Data/testData.txt")
        realDataTemp = realDataTemp[0:2000]
        # get the time axes for the bottom
        t = np.linspace(0, self.T*20, num=(20*self.timeSteps+1))
        t = t[0:2000]
        
        plt.figure(figsize=(16, 12), dpi=80)
        plt.plot(t, realDataTemp )
        plt.show()
    
