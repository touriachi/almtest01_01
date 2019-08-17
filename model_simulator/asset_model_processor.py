import json
import logging
import logging.config
import os
import jsonschema
import copy
import time
import numpy as np
import pandas as pd
import math
import timeit
import copy

import sklearn.preprocessing
from   scipy  import  array, linalg, dot

from pandas import ExcelWriter
from pandas import ExcelFile
from tools import *

from model_simulator.tools import is_positive_semi_definite, nearPD, get_normd_matrix, get_previous_colunmn


def update_years_values_step1(group,invCor,years):

   t2 = np.array(group['temp'])
   mm = np.array(group[years])
   result=np.exp((((invCor.dot(mm)).transpose()) * t2).transpose()) - 1
   i=0
   for fy in years :
      group[fy] =result[:,i]
      i=i+1
   return group


def update_years_values_step2(group, years):
   n_scaler = sklearn.preprocessing.StandardScaler()
   scale_years= n_scaler.fit_transform(np.array(group[years]))
   temp = np.array(group['temp'])
   result=(scale_years.transpose() * temp).transpose()
   i = 0
   for fy in years :
      group[fy] =result[:,i]
      i = i + 1
   return group


def scale_values_step2(group, years):
   n_scaler = sklearn.preprocessing.StandardScaler()
   scale_years= n_scaler.fit_transform(np.array(group[years]))
   group.loc[:, 'Year1':'Year25'] =scale_years
   return group


def step31(group,vecMne,fy):
    mask = (group["MT"] == "Prb") & (group.loc[group['MN'].isin(vecMne)].empty == False)
    valid_data = group[mask]

    if valid_data.empty== True :
        return

    quant =np.quantile(valid_data[fy],0.5)
    sup_quant = valid_data[valid_data[fy] > quant]
    group.loc[sup_quant.index, "RT"] = 1

    quant = np.quantile(valid_data[fy], 0.5)
    sup_quant = valid_data[valid_data[fy] <= quant]
    group.loc[sup_quant.index, "RT"] = 0


    return group


class AssetModelProcessor():

    def __init__(self, contract,logger):
        # self.asset_model = contract.inputs.asset_model
        self.asset_model =copy.deepcopy(contract.inputs.asset_model)
        self.request = contract.request
        self.logger=logger

    def error_simu(self,assets,  selected_assets, corMat, nHorizon, nSimu):

        #Check if matrix is semi-definite positive , if not, we create the nearest
        if not is_positive_semi_definite(corMat,self.logger):
            corMat = nearPD(corMat)

        if not is_positive_semi_definite(corMat, self.logger):
            self.logger.debug("Following eign values approach, the nearest correlation matrix is not positive semi-definite")

        # take the inverse of the correlation matrix
        invCor = np.linalg.cholesky(corMat)

        #Declare all of the rows necessary for the simulation

        probabilistic_data = assets.loc[assets.index.isin(selected_assets)]


        #duplicate rows  and add  required variables  ( simuID , years ) for all of the simulations
        years = []
        for i in range(1, nHorizon + 1):
            years.append('Year' + str(i))

        probabilistic_data = probabilistic_data.loc[probabilistic_data.index.repeat(nSimu)]
        probabilistic_data['simuID'] = probabilistic_data.groupby(['MN', 'MT']).cumcount() + 1


        probabilistic_data =  pd.concat([ probabilistic_data, pd.DataFrame(columns=years)], sort=False)
        #TODO   bug in bug Pandas  concat  function  remove index name.
        probabilistic_data.index.name = 'MN'




        #Generate the simulated error terms that are correlated (multivariate process)
        matrix = np.zeros([nSimu, nHorizon])
        probabilistic_data = probabilistic_data.reset_index()
        probabilistic_data= probabilistic_data.groupby('MN').apply(get_normd_matrix, matrix, years)
        probabilistic_data = probabilistic_data.groupby('MN').apply(scale_values_step2,years)








        #Calculate Error  matrix
        probabilistic_data.loc[probabilistic_data['MT'].isin(['Ran', 'Aut', 'Prb']), 'temp'] = probabilistic_data['SD']
        probabilistic_data.loc[probabilistic_data['MT'].isin(['Ran', 'Aut', 'Prb']) == False, 'temp'] = probabilistic_data['SPSD'] / 10000

        probabilistic_data = probabilistic_data.groupby('simuID').apply(update_years_values_step1,invCor,years)
        # With scaling of error term
        probabilistic_data =  probabilistic_data.groupby('MN').apply(update_years_values_step2,years)

        probabilistic_data.drop (['temp'],axis=1, inplace=True)
        probabilistic_data['simuID'] = probabilistic_data['simuID'].astype('int64')

        probabilistic_data = probabilistic_data.sort_values(by=['simuID'])



        return  probabilistic_data



    @classmethod
    def process(cls,data,logger):

        logger.debug('--Start Asset Simulation')

        processor = cls(data,logger)
        errorDT=processor.step02()
        # errorDT = get_workspace('errorDT')
        # childDB=processor.asset_model.assets_child
        # processor.step03(errorDT,childDB)





    def step02(self):

        self.logger.debug("Step 2 started : Creation correlation matrix error")

        # select specif models :"Aut","Ran","Prb", "Inf", "Mat"
        select_assets = self.asset_model.assets.loc[self.asset_model.assets['MT'].isin(['Aut','Ran','Prb', 'Inf', 'Mat'])==False].index.values

        # dropping passed columns
        for mn in select_assets:
            self.asset_model.correlation_matrix= self.asset_model.correlation_matrix.drop(mn,axis=0,errors="ignore")
            self.asset_model.correlation_matrix = self.asset_model.correlation_matrix.drop(mn,axis=1, errors="ignore")


        selected_assets=self.asset_model.correlation_matrix.index.values
        corMat=self.asset_model.correlation_matrix.values

        nHorizon = self.asset_model.general_config["YR"]
        nSimu = self.asset_model.general_config["SIM"]

        errorDT =self.error_simu(self.asset_model.assets, selected_assets,corMat, nHorizon, nSimu)

        self.logger.debug("Step 2 finished : Creation correlation matrix error")

        return errorDT

    #---------------------------------------------------STEP 03----------------------------



    # STEP 3 - given the error terms, generate a table with all of the correlated
    # returns
    # attributes required: errorDT (all necessary elements are in this object),
    # childDB (have the dependencies for each Parent class)

    def step03(self,retDT,childDB):

        vecMne = (retDT.loc[retDT['MN'] != "PD",'MN']).unique()
        years_cols = [col for col in retDT.columns if 'Year' in col]
        for fy in years_cols :

            previous_col = get_previous_colunmn(retDT.columns, fy)

            if fy >years_cols[0] :   # Year2 ... Year25
                mask = (retDT.MT == "Aut") & (np.isnan(retDT["AC"]) == False)                       \
                       & (np.isnan(retDT[previous_col]) == False)                                   \
                       & (retDT.loc[retDT['MN'].isin(vecMne)].empty == False)

                valid_data = retDT[mask]
                retDT.loc[mask, fy] = valid_data["AC"]*valid_data[previous_col] +(1-valid_data['AC'])*valid_data['AM'] +valid_data[fy]

            else : # Year1
                mask = (retDT.MT=="Aut") & (np.isnan(retDT["AC"]) == False) & (np.isnan(retDT["SV"]) == False) \
                       & (np.isnan(retDT['AM'])==False)                                                        \
                       & (retDT.loc[retDT['MN'].isin(vecMne)].empty ==False)

                valid_data = retDT[mask]
                retDT.loc[mask,fy]= valid_data["AC"] *valid_data["SV"] +(1-valid_data['AC'])*valid_data['AM'] +valid_data[fy]



            # set a minimum return for AR processes
            mask = (retDT["MT"] == "Aut") & (retDT[fy] < retDT["MIN"]) & (retDT.loc[retDT['MN'].isin(vecMne)].empty == False)
            valid_data = retDT[mask]
            retDT.loc[mask, fy] = valid_data["MIN"]

            # run the Random process returns
            mask = (retDT["MT"] == "Ran") & (np.isnan(retDT["AM"]) == False) & (retDT.loc[retDT['MN'].isin(vecMne)].empty == False)
            valid_data = retDT[mask]
            retDT.loc[mask, fy] = valid_data["AM"]+valid_data[fy]

            mask = ((retDT["MT"]=="Inf") | (retDT["MT"]=="Mat"))  & (np.isnan(retDT["SP"]) == False) \
                   & (retDT.loc[retDT['MN'].isin(vecMne)].empty == False)
            valid_data = retDT[mask]
            retDT.loc[mask, fy] = valid_data["SP"]/1000 + valid_data[fy]

            # using the correlated errors for the Prb processes, simulate the regime
            # switching returns
            retDT = retDT.groupby('MN').apply(step31,vecMne,fy)

            temp1 = retDT.loc[ np.isnan(retDT["RT"]) == False ,["MN", "simuID", "RT"]]
            #temp2=




