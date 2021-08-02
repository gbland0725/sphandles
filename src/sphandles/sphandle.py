import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import scipy as sp
import math
import numpy as np
from sklearn.decomposition import NMF

plt.style.use('seaborn-notebook')
class sphandle:
    @staticmethod
    def parse_filename(fname):
        pattern = r".*__(\w+).csv"
        m = re.match(pattern, fname)
        return m.group(1)
    
    @staticmethod
    #function to parse old and new data format
    def tofread_csv(rawdata):
        #read csv file.
        data = pd.read_csv(rawdata, error_bad_lines=False)
        #rennaming 1st column as ID
        data = data.rename(index=str, columns={"Unnamed: 0": "ID"})
        #drop columns that have volume and size
        data = data[data.columns.drop(list(data.filter(regex='vol')))]
        data = data[data.columns.drop(list(data.filter(regex='size')))]
        data.columns = data.columns.str.replace('_mass_g','')
        return data
    
    @staticmethod
    def read_csv(fname,dropcols=None, dropzero = None):
        dropcols = dropcols if dropcols else []
        label = sphandle.parse_filename(fname)
        df = pd.read_csv('./{}'.format(fname),
                        index_col=0)
        df = df[[c for c in df.columns if c.endswith('mass_g') and c not in dropcols]].fillna(0.0)
        df.columns = df.columns.str.replace('_mass_g','')
        if dropzero != None:
            df = df[abs(df).T.sum() > 0].reset_index(drop=True)
            df[df < 0] = 0
            df['label'] = label
            df = df.set_index('label')
        return df, label

    @staticmethod
    def nmf_analysis(df, label):
        nmf = NMF(components=2)
        X = nmf.fit_transform(df)
        x1 = X[:, 0]
        x2 = X[:, 1]
        fig, ax = plt.subplots()
        ax.plot(x1, x2, '.')
        ax.set(
            xlabel="First component",
            ylabel="Second component",
            title="%s: projected onto first two components" % label,
        )
        error = nmf.reconstruction_err_ / float(len(df))
        components = pd.DataFrame(nmf.components_.T, index=df.columns)


    #Type: Data Frame. From a selected isotope and its particle events, select all other isotopes associated and drop others not associated with it
    @staticmethod
    def isotope_particle(data, isotope):
        obs = data[data[isotope] > 0.0]
        return obs

    #Merging particle splits based on analyte
    @staticmethod
    def merge_particles(df,dfc):
        index_selector = []
        for i in dfc.columns:
            index_selector1 = dfc[dfc[i].notnull()][i].cumsum()
            index_selector.append(index_selector1)
        dfcorrected = df.groupby(index_selector[13]).agg('sum').reset_index(drop=True).drop(columns = 'ID')
        dfcorrected = dfcorrected.replace(0, np.nan)
        return dfcorrected

    #displays R-Sqr value for each dissolved calibration
    @staticmethod
    def r_sqr_analyzer(data, R_value):
        c = data[data['r_sqr_counts'] < R_value]
        return c

    #user function: loads a list of variable path names and assigns a label 
    @staticmethod
    def load_and_label(pathname, newlabel, DROPCOLS):
        data1 = pd.DataFrame()
        for i in glob.glob(pathname):
            data, soil_label = read_csv(i,DROPCOLS)
            data1 = pd.concat([data,data1], axis = 0, sort=False)
        data1['newlabel'] = newlabel
        data1 = data1.set_index('newlabel')
        return data1


    #return df_with_prob
    @staticmethod
    def just_data(data):
        new = data.drop(columns = ['newlabel','Engineered','Natural','prob_obs_group'])
        return new

    #marginal_probabilities returns out of the total particle events, what is the probability of an individual isotope occurring
    @staticmethod
    def marginal_probabilities(data):
        return (abs(data) > 0.0).mean().sort_values(ascending=False)

    #Counts total number of peaks for each isotope
    @staticmethod
    def marginal_particle(data):
        return data[data > 0.0].count().sort_values(ascending=False)

    @staticmethod
    def non_zero_data(data):
        non_zero_rows = data.abs().sum(axis=1) > 0.0
        non_zero_data = data[non_zero_rows]
        non_zero_columns = non_zero_data.abs().sum(axis=0) > 0.0
        non_zero_data = non_zero_data.loc[: , non_zero_columns]
        return non_zero_data

    #counts the total particles associated to an isotope
    #conditional_probabilities returns out of the total selected-isotope particle events, what is the probability it is associated with the list of other isotopes 
    @staticmethod
    def conditional_probabilities(data, isotope):
        obs = data[abs(data[isotope]) > 0.0]
        partners = (abs(obs) > 0.0).astype(np.float64).mean()
        return partners[abs(partners) > 0.0].sort_values(ascending=False)

    @staticmethod
    #counts the total particles associated to an isotope
    def conditional_particle(data, isotope):
        obs = data[abs(data[isotope]) > 0.0].count()
        return obs.sort_values(ascending=False)

    #isotope_pure returns a data frame of the selected isotope particle event and its impurities
    @staticmethod
    def isotope_pure(data, isotope):
        obs = data[data[isotope] > 0.0]
        others = obs.drop(columns=isotope)
        pure = others.sum(axis=1) == 0.0
        others = obs[pure]
        return others

    #probability_pure returns out of the total selected-isotope particle events, what is the probability that it is not associated with any other isotope
    @staticmethod
    def probability_pure(data, isotope):
        obs = data[data[isotope] > 0.0]
        others = obs.drop(columns=isotope)
        pure = (others.sum(axis=1) == 0.0).mean()
        return pure

    #returns the same number of particles to the smallest dataframe
    @staticmethod
    def sameamount(dfA, dfB):
        if len(dfA) > len(dfB):
            dfC = pd.concat([dfA[:len(dfB)], dfB], sort=False).fillna(0.0).clip(lower=0.0)
        else:
            dfC = pd.concat([dfA, dfB[:len(dfA)]], sort=False).fillna(0.0).clip(lower=0.0)

        labelsA = dfC.index.get_level_values(level='newlabel')
        return dfC, labelsA

    @staticmethod
    def category_split(labeldf, label, upperbound, lowerbound):
            correct = labeldf[labeldf[label] > upperbound]
            uncertain = labeldf[labeldf[label] < upperbound]
            uncertain = uncertain[uncertain[label] > lowerbound]
            misclassified = labeldf[labeldf[label] < lowerbound]
            return [correct, uncertain, misclassified]

    #isotope_notisotope returns a data frame of the selected isotope particle event not related to another selected isotope
    @staticmethod
    def isotope_notisotope(data, isotope1, isotope2):
        obs = data[data[isotope1] > 0.0]
        obs = obs[obs[isotope2] == 0.0]
        return obs
    
    @staticmethod
    #wrapping multiple dataframes into one
    def wrapper(datapath, DROPCOLS):
        empty = []
        label = []
        for i in glob.glob(datapath + '*'):
            print(i)
            empty.append(read_csv(i, DROPCOLS)[0])
            label.append(read_csv(i, DROPCOLS)[1])
        return empty, label

    @staticmethod
    def categories(sample, analyte, perc1, perc2):
        sampleperc1 = len(sample[sample ['p_engineered'] >= perc1])/len(sample)
        sampleperc2 = len(sample[sample ['p_engineered'] >= perc2])/len(sample)
        #sampleperc1 = sampleperc1 - sampleperc2
        return sampleperc1, sampleperc2

    @staticmethod
    #Particle concentration with the necessary inputs
    def Particleconc(dataframe, analyte, Flowrate, nebefficiency, timeacq, numberruns,  volume = None, Soilmass = None, dilution = 1):
        Count = dataframe[dataframe[analyte].notnull()].count()[analyte]
        Particleconc = Count*timeacq/(numberruns*nebefficiency*Flowrate)
        print('{:0.2e}'.format(Particleconc*dilution * volume/Soilmass))
        try:
            return Particleconc*dilution * volume/Soilmass
        except TypeError:
            return Particleconc

    @staticmethod
    def categories(sample, analyte, perc1, perc2):
        sampleperc1 = len(sample[sample ['p_engineered'] >= perc1])/len(sample)
        sampleperc2 = len(sample[sample ['p_engineered'] >= perc2])/len(sample)
        return sampleperc1, sampleperc2