# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:56:55 2023

@author: henry
"""

import numpy as np
import pandas as pd

class Poolset():
    """
    macro_x for variables do not change within cross-section;
    micro_x for variables change within cross-section;
    y for dependent variables of individuals;
    micro_x is a list containing dataframes of individual features, their columns are the same with y's, will be named self.y_name;
    the first column in all of them are 'time'
    e.g. macro_x's columns ['time','GDP','exchange_rate',...]; micro_x's list [size, ROE,...], their columns ['time','bank_A',...];
    y's columns ['time','bank_A','bank_B',...]
    
    """
    def __init__(self, macro_x:pd.DataFrame, micro_x:list, y:pd.DataFrame):       
        self.y_name = y.columns[1:]
        self.timeall = pd.DataFrame(y['time'])
        self.y = y.copy()
        self.macro_x = macro_x.copy()
        self.micro_x = {}
        self.micro_x0 = micro_x.copy()
        for i in range(len(micro_x)):
            self.micro_x[i] = micro_x[i].iloc[:,1:].copy()
        for i in range(len(micro_x)):    
            self.micro_x[i].columns = [j+f'indiv_{i}' for j in list(micro_x[i].columns[1:])]               
        self.micro_x = pd.concat(list(self.micro_x.values()),axis=1) 
        self.macro_x = self.macro_x.drop(columns=['time'])
        self.y = self.y.drop(columns=['time'])
        
        self.lag = 0
        self.timelag = pd.DataFrame(y['time'])
        self.IE, self.TE = False, False
        self.time_start, self.time_end = y.iloc[0]['time'], y.iloc[-1]['time']
    
   
    def set_lag(self,lag):
        """
        periods lag behind, non-negative integer;
        the data indices should be enough to lag;
        y will not be affected
        
        """
        self.lag = lag      
    
    def set_time(self,time_start,time_end): 
        """
        e.g. time_start = '2010-01-01'
        
        """
        self.time_start = pd.to_datetime(time_start)
        self.time_end = pd.to_datetime(time_end)
        self.time = pd.DataFrame(self.timeall[(self.timeall['time']>=time_start) & (self.timeall['time']<=time_end)])
        self.timelag = pd.DataFrame(self.timeall.iloc[self.time.index - self.lag]['time'])
        self.time_startlag = self.timeall.iloc[self.timeall[self.timeall['time']==self.time_start].index.values - self.lag]
        self.time_startlag = pd.to_datetime(self.time_startlag.squeeze())
        self.time_endlag = self.timeall.iloc[self.timeall[self.timeall['time']==self.time_end].index.values - self.lag]
        self.time_endlag = pd.to_datetime(self.time_endlag.squeeze())
        
        self.y['time'] = self.time['time']
        self.y = self.y.loc[(self.y['time'] >= self.time_start) & (self.y['time'] <= self.time_end)].reset_index(drop='True')
        self.y = self.y.drop(columns=['time'])
        self.macro_x['time'] = self.timelag['time']
        self.macro_x = self.macro_x.loc[(self.macro_x['time'] >= self.time_startlag) & (self.macro_x['time'] <= self.time_endlag)].reset_index(drop='True')
        self.macro_x = self.macro_x.drop(columns=['time'])
        self.micro_x={}
        for i in range(len(self.micro_x0)):
          self.micro_x[i] = self.micro_x0[i].loc[(self.micro_x0[i]['time'] >= self.time_startlag) & (self.micro_x0[i]['time'] <= self.time_endlag),self.y_name].reset_index(drop='True')          
          self.micro_x0[i] = self.micro_x0[i].loc[(self.micro_x0[i]['time'] >= self.time_startlag) & (self.micro_x0[i]['time'] <= self.time_endlag)].reset_index(drop='True')   
        for i in range(len(self.micro_x0)):
          self.micro_x[i].columns = [j+f'indiv_{i}' for j in list(self.micro_x0[i].columns[1:])]
        self.micro_x = pd.concat(list(self.micro_x.values()),axis=1)
        
    def set_Ieffect(self,name_withgroup): 
        """
        group individuals by a nested list with y_name. 
        e.g. name_withgroup = [[a,b],[c,d,e]]
        
        """
        if self.TE:
            raise RuntimeError('set_Ieffect should be done before set_Teffect, set again')
        self.name_withgroup = name_withgroup
        self.group = [str(i) for i in range(len(self.name_withgroup))]
        self.IE = True
        self.renamed_columns = {}
        for group_idx, group_columns in enumerate(self.name_withgroup):
            prefix = str(group_idx)
            for column in group_columns:
                self.renamed_columns[column] = prefix + column
        self.grouped_n = list(self.renamed_columns.values())    
        self.micro_x=self.micro_x0.copy()        
        for i in range(len(self.micro_x)):  
          self.micro_x[i] = self.micro_x[i].drop(columns=['time'])
          self.micro_x[i] = self.micro_x[i].rename(columns=self.renamed_columns)
          self.micro_x[i] = pd.concat([self.timelag,self.micro_x[i]],axis=1)
          self.micro_x[i] = self.micro_x[i].set_index('time')
          self.micro_x[i] = self.micro_x[i].T.stack()
          self.micro_x[i] = pd.DataFrame(self.micro_x[i],columns=[f'indiv_{i}'])
        self.micro_x = pd.concat(self.micro_x,axis=1)
        self.micro_x['group'] = [idx[0] for idx in self.micro_x.index.get_level_values(0)]
        self.micro_x = pd.get_dummies(self.micro_x, columns=['group'],drop_first=True)
        for col in self.micro_x.columns:
            if col.startswith('group_'):  
                self.micro_x[col] = self.micro_x[col].astype(int)
        self.microIE_id1 = self.micro_x.index.get_level_values(0)
        self.microIE_id2 = self.micro_x.index.get_level_values(1)
        self.microIE_id = self.micro_x.index
        self.micro_x = self.micro_x.reset_index(drop=True)   
        
        state_col = self.macro_x.columns
        self.macro_x = np.vstack([self.macro_x] * len(self.y_name))
        self.macro_x = pd.DataFrame(self.macro_x,columns=state_col)
        self.y.rename(columns=self.renamed_columns, inplace=True)
        self.y['time'] = self.time['time']
        self.y = self.y.set_index('time')
        self.y = self.y.T.stack()
        self.y = pd.DataFrame(self.y,columns=['y'])   
        self.yIE_id1 = self.y.index.get_level_values(0)
        self.yIE_id2 = self.y.index.get_level_values(1)
        self.yIE_id = self.y.index
        self.y = self.y.reset_index(drop=True)
        
        
    def set_Teffect(self,tgroup): 
        """
        add time effect by a list with time periods written in their last dates
        e.g. tgroup = ['2010-12-01', '2018-12-01']
        
        """
        self.TE = True
        self.micro_x['time'] = pd.concat([self.time['time']]*len(self.y_name), ignore_index=True)
        for t in tgroup[1:]:
            tstart = tgroup[tgroup.index(t)-1]
            tstart = pd.to_datetime(tstart)
            tend = pd.to_datetime(t)
            self.micro_x[t] = 0
            self.micro_x.loc[(self.micro_x['time']<=tend) & (self.micro_x['time']>tstart),t] = 1                     
        self.micro_x = self.micro_x.drop(columns=['time'])
           
                
    def get_state(self,show):
        if show == 'lag':           
            return self.lag
        if show == 'time':           
            return self.time_start, self.time_end
        if show == 'IE':
            return self.IE
        if show == 'TE':
            return self.TE
        