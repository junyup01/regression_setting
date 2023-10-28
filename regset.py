# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 02:13:19 2023

@author: henry
"""

import numpy as np
import pandas as pd

class MicroX():
    """
    micro_x for variables change within cross-section;
    micro_x is a list containing dataframes of individual features, their columns are the same with y's in YforI, will be named self.name;
    the first column in all of them are 'time'
    e.g. micro_x's list [size, ROE,...], their columns ['time','bank_A',...]
    
    """
    def __init__(self, micro_x:list):       
        self.name = micro_x[0].columns[1:]
        self.timeall = pd.DataFrame(micro_x[0]['time'])
        self.time = pd.DataFrame(micro_x[0]['time'])
        self.group = [str(i) for i in range(len(micro_x))]
        self.microlen = len(micro_x)
        self.micro_x = {}
        self.micro_x0 = micro_x.copy()
        for i in range(len(micro_x)):
            self.micro_x[i] = micro_x[i].iloc[:,1:].copy()
        for i in range(len(micro_x)):    
            self.micro_x[i].columns = [j+f'indiv_{i}' for j in list(micro_x[i].columns[1:])]
        self.micro_x = pd.concat(list(self.micro_x.values()),axis=1) 
        
        self.lag = 0
        self.timelag = pd.DataFrame(micro_x[0]['time'])
        self.IE, self.TE, self.isfold = False, False, False
        self.time_start, self.time_end = micro_x[0].iloc[0]['time'], micro_x[0].iloc[-1]['time']    
    
   
    def set_lag(self,lag):
        """
        periods lag behind, non-negative integer;
        the data indices should be enough to lag.
        
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
        
        self.micro_x={}
        for i in range(len(self.micro_x0)):
          self.micro_x[i] = self.micro_x0[i].loc[(self.micro_x0[i]['time'] >= self.time_startlag) & (self.micro_x0[i]['time'] <= self.time_endlag),self.name].reset_index(drop='True')          
          self.micro_x0[i] = self.micro_x0[i].loc[(self.micro_x0[i]['time'] >= self.time_startlag) & (self.micro_x0[i]['time'] <= self.time_endlag)].reset_index(drop='True')   
        for i in range(len(self.micro_x0)):
          self.micro_x[i].columns = [j+f'indiv_{i}' for j in list(self.micro_x0[i].columns[1:])]
        self.micro_x = pd.concat(list(self.micro_x.values()),axis=1)
        
    def set_Ieffect(self,name_withgroup): 
        """
        group individuals by a nested list with self.name. 
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
        
        
    def set_Teffect(self,tgroup): 
        """
        add time effect by a list with time periods written in their last dates
        e.g. tgroup = ['2010-12-01', '2018-12-01']
        
        """
        self.TE = True
        self.tgroup = tgroup
        self.micro_x['time'] = pd.concat([self.time['time']]*len(self.name), ignore_index=True)
        for t in tgroup[1:]:
            tstart = tgroup[tgroup.index(t)-1]
            tstart = pd.to_datetime(tstart)
            tend = pd.to_datetime(t)
            self.micro_x[t] = 0
            self.micro_x.loc[(self.micro_x['time']<=tend) & (self.micro_x['time']>tstart),t] = 1                     
        self.micro_x = self.micro_x.drop(columns=['time'])
           
        
    def microFold(self, byI=False, byI_allnamedby=None):
        """
        to seperate micro_x's individual features dataframes, dummies are retained for each feature;
        to avoid troubles, do not let the number of individual features greater than 10, especially without effects;
        byI for seperating by individuals, byI_allnamedby for setting the same name within a feature for individuals' features
        
        """
        self.isfold = True
        micro_x0 = []
        if not byI:
            if self.IE:
                for i in list(range(self.microlen)):
                    micro_x0.append(self.micro_x.iloc[:,i])
                    micro_x0[i] = pd.concat([micro_x0[i], self.micro_x.loc[:,[col for col in self.micro_x.columns.str.startswith('group_')]]],axis=1)
                    if self.TE:
                        micro_x0[i] = pd.concat([micro_x0[i], self.micro_x[self.tgroup[1:]]],axis=1)
            if self.TE and not self.IE:
                for i in list(range(self.microlen)):
                    micro_x0.append(self.micro_x.iloc[:,:-(len(self.tgroup)-1)].loc[:,[col for col in self.micro_x.columns[:len(self.name)*self.microlen].str.endswith(f'{i}')]])
                    micro_x0[i] = pd.concat([micro_x0[i], self.micro_x[self.tgroup[1:]]],axis=1)
            if not self.IE and not self.TE:
                for i in list(range(self.microlen)):
                    micro_x0.append(self.micro_x.loc[:,[col for col in self.micro_x.columns.str.endswith(f'{i}')]])            
        else:          
            if self.IE or self.TE:
                raise RuntimeError('not applicable with effects')
            for i in self.micro_x0[0].columns[1:]:
                mic_df = pd.DataFrame()
                for k, j in enumerate(self.micro_x0):
                    if byI_allnamedby != None:
                        mic_df[f'{byI_allnamedby}_{k}'] = j[i]
                    else:
                        mic_df[f'{i}_{k}'] = j[i]
                micro_x0.append(mic_df)
        self.micro_x = micro_x0.copy()          
            
            
    def get_state(self,show):
        if show == 'lag':           
            return self.lag
        if show == 'time':           
            return self.time_start, self.time_end
        if show == 'IE':
            return self.IE
        if show == 'TE':
            return self.TE
        if show == 'isfold':
            return self.isfold
        
        
class MacroX():
    """
    macro_x for variables do not change within cross-section;
    the first column is 'time'
    e.g. macro_x's columns ['time','GDP','exchange_rate',...]
    
    """
    def __init__(self, macro_x:pd.DataFrame):       
        self.timeall = pd.DataFrame(macro_x['time'])
        self.time = pd.DataFrame(macro_x['time'])
        self.macro_x = macro_x.copy()
        self.macro_x = self.macro_x.drop(columns=['time'])
        
        self.lag = 0
        self.timelag = pd.DataFrame(macro_x['time'])
        self.IE, self.TE = False, False
        self.time_start, self.time_end = macro_x.iloc[0]['time'], macro_x.iloc[-1]['time']    
    
   
    def set_lag(self,lag):
        """
        periods lag behind, non-negative integer;
        the data indices should be enough to lag.
        
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
        
        self.macro_x['time'] = self.timelag['time']
        self.macro_x = self.macro_x.loc[(self.macro_x['time'] >= self.time_startlag) & (self.macro_x['time'] <= self.time_endlag)].reset_index(drop='True')
        self.macro_x = self.macro_x.drop(columns=['time'])
        
    def set_Ieffect(self,times): 
        """
        times follows individual number in your regression, which is not seen in this class.
        
        """
        self.IE = True
        state_col = self.macro_x.columns
        self.macro_x = np.vstack([self.macro_x] * times)
        self.macro_x = pd.DataFrame(self.macro_x,columns=state_col)
        
        
    def set_Teffect(self,tgroup): 
        """
        add time effect by a list with time periods written in their last dates
        e.g. tgroup = ['2010-12-01', '2018-12-01']
        
        """
        if self.IE:
            raise RuntimeError('set_Teffect should be done before set_Ieffect, set again')
        self.TE = True
        self.tgroup = tgroup
        self.macro_x['time'] = self.time['time']
        for t in tgroup[1:]:
            tstart = tgroup[tgroup.index(t)-1]
            tstart = pd.to_datetime(tstart)
            tend = pd.to_datetime(t)
            self.macro_x[t] = 0
            self.macro_x.loc[(self.macro_x['time']<=tend) & (self.macro_x['time']>tstart),t] = 1                     
        self.macro_x = self.macro_x.drop(columns=['time'])
           
                       
    def get_state(self,show):
        if show == 'lag':           
            return self.lag
        if show == 'time':           
            return self.time_start, self.time_end
        if show == 'IE':
            return self.IE
        if show == 'TE':
            return self.TE
        if show == 'isfold':
            return 'not applicable'
        
        

class YforI():
    """
    y for dependent variables of individuals;
    the first column is 'time'
    e.g. y's columns ['time','bank_A','bank_B',...]
    
    """
    def __init__(self, y:pd.DataFrame):       
        self.name = y.columns[1:]
        self.timeall = pd.DataFrame(y['time'])
        self.time = pd.DataFrame(y['time'])
        self.y = y.copy()
        self.y = self.y.drop(columns=['time'])

        self.IE = False
        self.time_start, self.time_end = y.iloc[0]['time'], y.iloc[-1]['time']    
       
    
    def set_time(self,time_start,time_end): 
        """
        e.g. time_start = '2010-01-01'
        
        """
        self.time_start = pd.to_datetime(time_start)
        self.time_end = pd.to_datetime(time_end)
        self.time = pd.DataFrame(self.timeall[(self.timeall['time']>=time_start) & (self.timeall['time']<=time_end)])
        
        self.y['time'] = self.time['time']
        self.y = self.y.loc[(self.y['time'] >= self.time_start) & (self.y['time'] <= self.time_end)].reset_index(drop='True')
        self.y = self.y.drop(columns=['time'])

    def set_Ieffect(self,name_withgroup): 
        """
        group individuals by a nested list with self.name. 
        e.g. name_withgroup = [[a,b],[c,d,e]]
        
        """
        self.name_withgroup = name_withgroup
        self.group = [str(i) for i in range(len(self.name_withgroup))]
        self.IE = True
        self.renamed_columns = {}
        for group_idx, group_columns in enumerate(self.name_withgroup):
            prefix = str(group_idx)
            for column in group_columns:
                self.renamed_columns[column] = prefix + column
        self.grouped_n = list(self.renamed_columns.values())
    
        self.y.rename(columns=self.renamed_columns, inplace=True)
        self.y['time'] = self.time['time']
        self.y = self.y.set_index('time')
        self.y = self.y.T.stack()
        self.y = pd.DataFrame(self.y,columns=['y'])   
        self.yIE_id1 = self.y.index.get_level_values(0)
        self.yIE_id2 = self.y.index.get_level_values(1)
        self.yIE_id = self.y.index
        self.y = self.y.reset_index(drop=True)
        
            
    def get_state(self,show):
        if show == 'lag':           
            return 'not applicable'
        if show == 'time':           
            return self.time_start, self.time_end
        if show == 'IE':
            return self.IE
        if show == 'TE':
            return 'not applicable'
        if show == 'isfold':
            return 'not applicable'
        
        
class YforSys():
    """
    y_sys for dependent variables of a system;
    the first column is 'time'
    e.g. y_sys's columns ['time','system_index']
    
    """
    def __init__(self, y_sys:pd.DataFrame):       
        self.timeall = pd.DataFrame(y_sys['time'])
        self.time = pd.DataFrame(y_sys['time'])
        self.y_sys = y_sys.copy()
        self.y_sys = self.y_sys.drop(columns=['time'])

        self.IE = False
        self.time_start, self.time_end = y_sys.iloc[0]['time'], y_sys.iloc[-1]['time']    
        
    
    def set_time(self,time_start,time_end): 
        """
        e.g. time_start = '2010-01-01'
        
        """
        self.time_start = pd.to_datetime(time_start)
        self.time_end = pd.to_datetime(time_end)
        self.time = pd.DataFrame(self.timeall[(self.timeall['time']>=time_start) & (self.timeall['time']<=time_end)])
        
        self.y_sys['time'] = self.time['time']
        self.y_sys = self.y_sys.loc[(self.y_sys['time'] >= self.time_start) & (self.y_sys['time'] <= self.time_end)].reset_index(drop='True')
        self.y_sys = self.y_sys.drop(columns=['time'])

    def set_Ieffect(self,times): 
        """
        times follows individual number in your regression, which is not seen in this class.
       
        """
        self.IE = True
        state_col = self.y_sys.columns
        self.y_sys = np.vstack([self.y_sys] * times)
        self.y_sys = pd.DataFrame(self.y_sys,columns=state_col)
        
            
    def get_state(self,show):
        if show == 'lag':           
            return 'not applicable'
        if show == 'time':           
            return self.time_start, self.time_end
        if show == 'IE':
            return self.IE
        if show == 'TE':
            return 'not applicable'
        if show == 'isfold':
            return 'not applicable'
    
        
