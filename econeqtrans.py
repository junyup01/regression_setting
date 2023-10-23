# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 02:08:07 2023

@author: henry
"""

import numpy as np
import pandas as pd

class Econeq:
    def __init__(self,name,name_withgroup,timeStart,timeEnd,macro_x,micro_x,rets,xsys,tgroup,lag):
        self.name = name
        self.name_withgroup = name_withgroup
        if self.name != self.name_withgroup: self.IE=True
        else: self.IE=False
        if tgroup != []: self.TE=True
        else: self.TE=False
        self.group = [str(i) for i in range(len(self.name_withgroup))]
        self.timeStart = pd.to_datetime(timeStart)
        self.timeEnd = pd.to_datetime(timeEnd)
        self.time = pd.DataFrame(rets[(rets['time']>=self.timeStart)&(rets['time'] <= self.timeEnd)]['time'],columns=['time'])
        self.timeLag = pd.DataFrame(rets.iloc[self.time.index - lag]['time'])
        self.timeslag = rets.iloc[self.time.index[0] - lag]['time']
        self.timeelag = rets.iloc[self.time.index[-1] - lag]['time']
        self.time, self.timeLag = self.time.reset_index(drop=True), self.timeLag.reset_index(drop=True)
        self.macro_x = macro_x.loc[(macro_x['time'] >= self.timeslag) & (macro_x['time'] <= self.timeelag)].drop(columns=['time']).reset_index(drop='True')
        self.micro_x = {}
        for i in range(len(micro_x)):
          self.micro_x[i] = micro_x[i].loc[(micro_x[i]['time'] >= self.timeslag) & (micro_x[i]['time'] <= self.timeelag),name].reset_index(drop='True')          
                    
        def doIGroup():  
            self.renamed_columns = {}
            for group_idx, group_columns in enumerate(self.name_withgroup):
                prefix = str(group_idx)
                for column in group_columns:
                    self.renamed_columns[column] = prefix + column
            self.grouped_n = list(self.renamed_columns.values())                          
            for i in range(len(micro_x)):  
              self.micro_x[i].rename(columns=self.renamed_columns, inplace=True)
              self.micro_x[i] = pd.concat([self.timeLag,self.micro_x[i]],axis=1)
              self.micro_x[i] = self.micro_x[i].set_index('time')
              self.micro_x[i] = self.micro_x[i].T.stack()
              self.micro_x[i] = pd.DataFrame(self.micro_x[i],columns=[f'indiv_{i}'])
            self.micro_x = pd.concat(list(self.micro_x.values()),axis=1)
            self.micro_x['group'] = [idx[0] for idx in self.micro_x.index.get_level_values(0)]
            self.micro_x = pd.get_dummies(self.micro_x, columns=['group'],drop_first=True)
            for col in self.micro_x.columns:
                if col.startswith('group_'):  
                    self.micro_x[col] = self.micro_x[col].astype(int)
            self.stackid1 = self.micro_x.index.get_level_values(0)
            self.stackid2 = self.micro_x.index.get_level_values(1)
            self.stackid = self.micro_x.index
            self.micro_x = self.micro_x.reset_index(drop=True)            
           
        def doTGroup(): 
            self.micro_x['time'] = pd.concat([self.time['time']]*len(self.name), ignore_index=True)
            for t in tgroup[1:]:
                tstart = tgroup[tgroup.index(t)-1]
                tstart = pd.to_datetime(tstart)
                tend = pd.to_datetime(t)
                self.micro_x[t] = 0
                self.micro_x.loc[(self.micro_x['time']<=tend) & (self.micro_x['time']>tstart),t] = 1                     
            self.micro_x = self.micro_x.drop(columns=['time'])
            
        if not self.IE:
            self.grouped_n = self.name
            for i in range(len(micro_x)):
                self.micro_x[i].columns = [j+f'indiv_{i}' for j in list(micro_x[i].columns[1:])]
            self.micro_x = pd.concat(list(self.micro_x.values()),axis=1)           
                
        if self.IE:
            doIGroup()  
            if self.TE:
                self.micro_x['time'] = pd.concat([self.time]*len(self.name), axis=0, ignore_index=True)['time']
        if self.TE:
            doTGroup()
        
        self.micro_x = self.micro_x.reset_index(drop=True)
       
        self.rets = rets.loc[(rets['time'] >= self.timeStart) & (rets['time'] <= self.timeEnd),name].reset_index(drop='True')
        self.xsys = xsys.loc[(xsys['time'] >= self.timeStart) & (xsys['time'] <= self.timeEnd)].reset_index(drop='True')
        if self.IE:
            self.xsys = np.vstack([self.xsys] * len(self.name))
            self.xsys = pd.DataFrame(self.xsys)

            state_col = self.macro_x.columns
            self.macro_x = np.vstack([self.macro_x] * len(self.name))
            self.macro_x = pd.DataFrame(self.macro_x,columns=state_col)
            self.rets.rename(columns=self.renamed_columns, inplace=True)
            self.rets = pd.concat([self.time,self.rets],axis=1)
            self.rets = self.rets.set_index('time')
            self.rets = self.rets.T.stack()
            self.rets = pd.DataFrame(self.rets,columns=['returns'])
            self.xsys = self.xsys.drop(columns=[0])
            self.xsys[1] = pd.to_numeric(self.xsys[1], errors='coerce')
        else:
            self.xsys = self.xsys.drop(columns=['time'])
        self.rets = self.rets.reset_index(drop=True)