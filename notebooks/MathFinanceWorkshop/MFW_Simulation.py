from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si

class MFW_Simulation:
    '''Create a simulation environment to track a simulated trading strategy'''
    def __init__(self):
        self.timeSteps = 100
        self.T = 1
        self.S0 = 1.
        self.delta_0 = 10
        self.B0 = 10
        self.realDataPaths=20
        self.competitionDataPaths=2
        
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
    
    def tradingStrategy(self,St,Wt,Bt,delta_t):
        changeInDelta = 1
        return changeInDelta
    
    def tradingPath(self,St):
        # Wt is the wealth process
        # W_t = \Delta S_t + B_t
        Wt = np.zeros(self.timeSteps+1)
        # Delta is the numbe of shares
        delta_t = np.zeros(self.timeSteps+1)
        # B is the current bank account
        Bt = np.zeros(self.timeSteps+1)
        dt = float(self.T) / float(self.timeSteps)
        # Initialise wealth
        delta_t[0] = self.delta_0
        Bt[0] = self.B0
        Wt[0] = delta_t[0]*St[0] + Bt[0]
        for i in range(self.timeSteps):
            
            changeInDelta=self.tradingStrategy(St[0:i+1],Wt[0:i+1],Bt[0:i+1],delta_t[0:i+1])
            changeInDelta=max(-1,min(1,changeInDelta))
            
            if delta_t[i] + changeInDelta < 0.:
                changeInDelta = -delta_t[i]
            # get +ve cost (-ve profit) to purchase (sell)
            cost = changeInDelta * St[i+1]
            
            if Bt[i] - cost < 0:
                changeInDelta = Bt[i] / St[i+1]
                cost = Bt[i]
            
            delta_t[i+1] = delta_t[i] + changeInDelta
            Bt[i+1] = Bt[i] - cost 
            Wt[i+1] = delta_t[i+1]*St[i+1] + Bt[i+1]
        return [ delta_t, Bt , Wt ]
    
    def plotSimulation(self,simulationRuns,mu,sigma):
        # create a figure with two subplots
        fig = plt.figure(figsize=(16, 12), dpi=80)
        ax1 = plt.subplot(2,2,1 )
        ax2 = plt.subplot(2,2,2 )
        ax3 = plt.subplot(2,2,3 )
        ax4 = plt.subplot(2,2,4 )
        # set graph titles
        ax1.set_title(r'Stock Price')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$S_t$')
        # -- 
        ax2.set_title(r'Number of Stocks')
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$\Delta_t$')
        # -- 
        ax3.set_title(r'Bank Account')
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$B_t$')
        # -- 
        ax4.set_title(r'Wealth Process')
        ax4.set_xlabel(r'$t$')
        ax4.set_ylabel(r'$W_t$')
        # -- 
        
        
        # get the time axes for the bottom
        t = np.linspace(0, self.T, num=(self.timeSteps+1))
        
        
        for i in range(simulationRuns):
            St = self.stockPath(mu,sigma)
            # generate a plot of the stock price 
            ax1.plot(t, St )
            delta_t, Bt , Wt = self.tradingPath(St)
            # generate a plot of the stock price 
            ax2.plot(t, delta_t )
            # generate a plot of the stock price 
            ax3.plot(t, Bt )
            # generate a plot of the stock price 
            ax4.plot(t, Wt )
            
    def statsSimulation(self,simulationRuns,mu,sigma):
        
        # get the time axes for the bottom
        t = np.linspace(0, self.T, num=(self.timeSteps+1))
        
        # store stats
        wealthReturns = np.zeros(simulationRuns)
        
        for i in range(simulationRuns):
            St = self.stockPath(mu,sigma)
            delta_t, Bt , Wt = self.tradingPath(St)            
            wealthReturns[i] = (Wt[-1] - Wt[0])/Wt[0]
        
        print(" Strategy generates average return ", 100*np.mean(wealthReturns) , "% and a volatility of ", 100*np.std(wealthReturns),"%" )
        
        plt.figure(figsize=(12, 8), dpi=80)
        plt.hist(wealthReturns,bins='auto')


    def plotRealData(self):
        # create a figure with two subplots
        fig = plt.figure(figsize=(16, 12), dpi=80)
        ax1 = plt.subplot(2,2,1 )
        ax2 = plt.subplot(2,2,2 )
        ax3 = plt.subplot(2,2,3 )
        ax4 = plt.subplot(2,2,4 )
        # set graph titles
        ax1.set_title(r'Stock Price')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$S_t$')
        # -- 
        ax2.set_title(r'Number of Stocks')
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$\Delta_t$')
        # -- 
        ax3.set_title(r'Bank Account')
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$B_t$')
        # -- 
        ax4.set_title(r'Wealth Process')
        ax4.set_xlabel(r'$t$')
        ax4.set_ylabel(r'$W_t$')
        # -- 
        
        
        # get the time axes for the bottom
        t = np.linspace(0, self.T, num=(self.timeSteps+1))
        
        for i in range(self.realDataPaths):
            St = self.realData[i]
            # generate a plot of the stock price 
            ax1.plot(t, St )
            delta_t, Bt , Wt = self.tradingPath(St)
            # generate a plot of the stock price 
            ax2.plot(t, delta_t )
            # generate a plot of the stock price 
            ax3.plot(t, Bt )
            # generate a plot of the stock price 
            ax4.plot(t, Wt )
            
    def statsRealData(self):
        
        # get the time axes for the bottom
        t = np.linspace(0, self.T, num=(self.timeSteps+1))
        
        # store stats
        wealthReturns = np.zeros(self.realDataPaths)
        
        for i in range(self.realDataPaths):
            St = self.realData[i]
            delta_t, Bt , Wt = self.tradingPath(St)            
            wealthReturns[i] = (Wt[-1] - Wt[0])/Wt[0]
        
        print(" Strategy generates average return ", 100*np.mean(wealthReturns) , "% and a volatility of ", 100*np.std(wealthReturns),"%" )
        
        plt.figure(figsize=(12, 8), dpi=80)
        plt.hist(wealthReturns,bins='auto')

    def runCompetitionData(self,dataFile):
        
        competitionDataTemp = np.fromfile("Data/"+dataFile)
        competitionData = np.zeros((self.competitionDataPaths,self.timeSteps+1))
        for i in range(self.competitionDataPaths):
            if i==0:
                competitionData[i] = competitionDataTemp[0:101]/competitionDataTemp[0]
            else:
                competitionData[i] = competitionDataTemp[i*100-1:(i+1)*100]/competitionDataTemp[i*100-1]
        
        # get the time axes for the bottom
        t = np.linspace(0, self.T, num=(self.timeSteps+1))
        
        # store stats
        wealthReturns = np.zeros(self.competitionDataPaths)
        
        # create a figure with two subplots
        fig = plt.figure(figsize=(16, 12), dpi=80)
        ax1 = plt.subplot(2,2,1 )
        ax2 = plt.subplot(2,2,2 )
        ax3 = plt.subplot(2,2,3 )
        ax4 = plt.subplot(2,2,4 )
        
        # set graph titles
        ax1.set_title(r'Stock Price')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$S_t$')
        # -- 
        ax2.set_title(r'Number of Stocks')
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r'$\Delta_t$')
        # -- 
        ax3.set_title(r'Bank Account')
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$B_t$')
        # -- 
        ax4.set_title(r'Wealth Process')
        ax4.set_xlabel(r'$t$')
        ax4.set_ylabel(r'$W_t$')
        
        for i in range(self.competitionDataPaths):
            St = competitionData[i]
            delta_t, Bt , Wt = self.tradingPath(St)            
            wealthReturns[i] = (Wt[-1] - Wt[0])/Wt[0]
            # generate a plot of the stock price 
            ax1.plot(t, St )
            # generate a plot of the stock price 
            ax2.plot(t, delta_t )
            # generate a plot of the stock price 
            ax3.plot(t, Bt )
            # generate a plot of the stock price 
            ax4.plot(t, Wt )
        
        
        
        print(" Strategy generates average return ", 100*np.mean(wealthReturns) , "% and a volatility of ", 100*np.std(wealthReturns),"%" )
        
    
