import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import math
import matplotlib.ticker as mtick
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import NMF, PCA
from sklearn.pipeline import make_pipeline, make_union, FeatureUnion
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sphandles.sphandle import sphandle

plt.style.use('seaborn-notebook')

class mltrain:
    
        #bootstrapping method
    # parsing DF for test dataset
    @staticmethod
    def holdoutdata(df, perc):
        #making hold out data
        dfnat = df[df.index == 'Natural']
        dfeng = df[df.index == 'Engineered']

        msk = np.random.rand(len(dfnat)) < perc
        mske = np.random.rand(len(dfeng)) < perc
        holdoutdatanat = dfnat[~msk]
        holdoutdataeng = dfeng[~mske]
        trainingdatanat = dfnat[msk]
        trainingdataeng = dfeng[mske]

        #combining holdout data together and labels
        holdoutdata = pd.concat([holdoutdatanat, holdoutdataeng], axis=0)
        holdoutlabels = holdoutdata.index.get_level_values(level='newlabel')
        return holdoutdata, holdoutlabels, trainingdatanat, trainingdataeng
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    @staticmethod
    def evaluate_model(clf, df, labels):
        model_name = '->'.join(next(zip(*clf.steps)))
        print("Model: %s" % model_name)
        scores = cross_val_score(clf, df, labels, cv=3)
        mean_score = scores.mean()
        std_score = scores.std()
        print("Accuracy: [%0.4f, %0.4f] (%0.4f +/- %0.4f)"
              % (mean_score - std_score, mean_score + std_score, mean_score, std_score))
        y_pred = cross_val_predict(clf, df, labels, cv=3)
        y_prob = cross_val_predict(clf, df, labels, cv=3, method='predict_proba')
        mltrain.plot_confusion_matrix(labels, y_pred, labels.unique())
        return y_pred, y_prob, labels, df

    @staticmethod
    def evaluate_modelrfe(clf, df, labels):
        model_name = '->'.join(next(zip(*clf.steps)))
        print("Model: %s" % model_name)
        scores = cross_val_score(clf, df, labels, cv=3)
        mean_score = scores.mean()
        std_score = scores.std()
        print("Accuracy: [%0.4f, %0.4f] (%0.4f +/- %0.4f)"
              % (mean_score - std_score, mean_score + std_score, mean_score, std_score))
        y_pred = cross_val_predict(clf, df, labels, cv=3)
        y_prob = cross_val_predict(clf, df, labels, cv=3, method='predict')
        return y_pred, y_prob, labels, df

    
    @staticmethod
    def graphmaker(df, label, upperbound, name, natdf, savefigs, opencircles = False):
        toppercent = sphandle.conditional_probabilities(sphandle.just_data(df[df[label] > upperbound]), '48Ti')
        bottompercent = sphandle.conditional_probabilities(sphandle.just_data(df[df[label] < upperbound]), '48Ti')
        percentdf = pd.DataFrame([toppercent, bottompercent]).T
        percentdf.columns = ['toppercent', 'bottompercent']
        percentdf.reset_index(inplace = True)
        percentdf = pd.melt(percentdf[1:13], id_vars = ['index'], value_vars = ['toppercent', 'bottompercent'])

        fig, axs = plt.subplots(1, 2, figsize=(16,4), dpi=300, gridspec_kw={'width_ratios': [2.5, 1]})
        ax1 = sns.barplot('index', 'value', 'variable', data = percentdf, ax = axs[0], palette = ['green', 'orange', 'red'])
        ax1.set_ylabel('Isotope Frequency', fontsize = 16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel('')
        ax1.set_ylim([10**-3, 1])
        ax1.legend_.remove()
        ax1.set_yscale('log')

        #if you want unassociated Ti as a separate category
        if opencircles == True:
            #make the df pure ti df
            Pure = sphandle.isotope_pure(sphandle.just_data(df).drop(columns = '46Ti'), '48Ti')
            percentdfpure = df.loc[Pure.index]
            df.drop(percentdfpure.index, inplace = True)

        #parse particles by confidence probability to the correct category
        top = df[df[label] > upperbound]
        puretop = percentdfpure[percentdfpure[label] > upperbound]
        bot = df[df[label] < upperbound]
        purebot = percentdfpure[percentdfpure[label] < upperbound]
    
        axs[1].scatter(bot['48Ti'], bot[label], linewidths = 0.75, edgecolors = 'white', facecolors = 'orange')
        axs[1].scatter(top['48Ti'], top[label], linewidths = 0.75, edgecolors = 'white', facecolors = 'green')
        axs[1].scatter(purebot['48Ti'], purebot[label], linewidths = 0.75, edgecolors = 'orange', facecolors = 'white')
        axs[1].scatter(puretop['48Ti'], puretop[label], linewidths = 0.75, edgecolors = 'green', facecolors = 'white')
        axs[1].set_xscale('log')
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[1].set_xlim([min(natdf['48Ti']), 10**-13])
        axs[1].set_xlabel('Ti Mass (g)')
        axs[1].set_ylabel('Prediction Probability of ' + str(label) + ' Category')
        axs[1].set_ylim(0.0)
        if savefigs == None:
            plt.savefig('figures/fingerprints' + str(label) + str(name) + '.png', dpi = 300)
     
    @staticmethod
    def graphmakerm(df, label, upperbound, lowerbound, name, natdf, savefigs, opencircles = False):
        toppercent = sphandle.conditional_probabilities(sphandle.just_data(df[df[label] > upperbound]), '48Ti')
        middlepercent = df[df[label] < upperbound]
        middlepercent = sphandle.conditional_probabilities(sphandle.just_data(middlepercent[middlepercent[label] > lowerbound]), '48Ti')
        bottompercent = sphandle.conditional_probabilities(sphandle.just_data(df[df[label] < lowerbound]), '48Ti')
        percentdf = pd.DataFrame([toppercent, middlepercent, bottompercent]).T
        percentdf.columns = ['toppercent', 'middlepercent', 'bottompercent']
        percentdf.reset_index(inplace = True)
        percentdf = pd.melt(percentdf[1:13], id_vars = ['index'], value_vars = ['toppercent', 'middlepercent', 'bottompercent'])

        fig, axs = plt.subplots(1, 2, figsize=(18,4), dpi=300, gridspec_kw={'width_ratios': [3, 1]})
        ax1 = sns.barplot('index', 'value', 'variable', data = percentdf, palette = 'Blues', ax = axs[0])
        ax1.set_ylabel('Isotope Frequency')
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.set_xlabel('')
        ax1.legend_.remove()
        ax1.set_yscale('log')

        #if opencircles == True:
            #Pure = isotope_pure(just_data(df).drop(columns = '46Ti'), label)

        top = df[df[label] > upperbound]
        top['category'] = 'Top ' + str(100-upperbound*100) + '%'
        middle = df[df[label] < upperbound]
        middle = middle[middle[label] > lowerbound]
        middle['category'] = 'Middle' + str((upperbound - lowerbound)*100) + '%'
        bot = df[df[label] < lowerbound]
        bot['category'] = 'Bottom ' + str(upperbound*100) + '%'


        df = pd.concat([bot, middle, top], axis = 0)

        ax2 = sns.scatterplot('48Ti', label, 'category', data = df, palette = 'Blues_d', ax = axs[1])
        Pure = sphandle.isotope_pure(sphandle.just_data(df).drop(columns = '46Ti'), '48Ti')
        percentdfpure = df.loc[Pure.index]
        ax2 = sns.scatterplot('48Ti', label, 'category', data = percentdfpure, palette = 'Blues_d', ax = axs[1], markers= '+')
        ax2.set_xscale('log')
        ax2.set_xlim([min(natdf['48Ti']), 10**-13])
        ax2.set_xlabel('Ti Mass (g)')
        ax2.legend_.remove()
        ax2.set_ylabel('Prediction Probability of ' + str(label) + ' Category')
        if savefigs == None:
            plt.savefig('figures/fingerprints' + str(label) + str(name) + '.png', dpi = 300)

    @staticmethod
    #This is to get the specific Ti column in the sample data frame. Please specify which index the column lies in. For this specific set of columns, it is 6. 
    def get_ti(x):
        return x[:, [6]]

    @staticmethod
    def MLsimulation(data, keys, labels, iterations, tolerance, naturalkeys, engineeredkeys, flag, savefigs, returns = None):
    
        print(keys, pd.DataFrame(np.ones_like(labels), columns=['count'], index=labels).groupby('newlabel').sum())
        #hold out some data for testing
        holdoutdata1, holdoutlabels, trainingdatanat, trainingdataeng = mltrain.holdoutdata(data, 0.8)
        #combine the left out training dataset and heldout labels. 
        trainingdata = pd.concat([trainingdatanat, trainingdataeng], axis=0)
        traininglabels = trainingdata.index.get_level_values(level='newlabel')

        #Logistic Regression with NMF and StandardScalar
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            FeatureUnion([('nmf', NMF(n_components = 10)), 
                          ('functiontransformer', FunctionTransformer(mltrain.get_ti))
                         ]),
            LogisticRegressionCV(tol=tolerance, cv=5, max_iter=iterations)
        )
        y_pred, y_prob, y_true, df = mltrain.evaluate_model(clf, trainingdata, traininglabels)
        if savefigs == None:
            plt.savefig('figures/confusionmatrix' + str(keys) + '.png', dpi =300)

        #fitting the model given the trained dataset
        clf = clf.fit(df, traininglabels)
        #predict the probability of trained datapoint
        y_prob = clf.predict_proba(df)
        #get the test_score of the test dataset
        test_score = clf.score(holdoutdata1, holdoutlabels)
        #print('test score: ' + str(test_score * 100))

        #Analyze trained components
        compcoeff = clf.named_steps['logisticregressioncv'].coef_   
        nmf = clf.named_steps['featureunion'].transformer_list[0][1]
        comps = pd.DataFrame(nmf.components_.T, index=data.columns)

        Ticolumn = np.zeros((clf.named_steps['featureunion'].transformer_list[0][1].components_[0].shape))
        Ticolumn[5] = 1
        Ticolumndf = pd.DataFrame(Ticolumn.T, index=data.columns)
        comps = pd.concat([comps, Ticolumndf], axis = 1)
        comps.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Ti']
        
        #multiply coefficient by componenets
        multipliedcomps = (compcoeff[0] * comps)


         #Loading dataset with prob_obs_group
        label_prob = pd.DataFrame(y_prob, columns=sorted(labels.unique()))
        df_with_prob = df.reset_index().join(label_prob)
        df_with_prob['prob_obs_group'] = df_with_prob.Natural.apply(lambda p: float("{:0.1f}".format(p))) 
        df_with_prob.append(df_with_prob)

        #Multiply the occurence of the element with the Ti particles by the importance
        multipliedcompscor =multipliedcomps.apply(lambda x: x *
                                                  sphandle.conditional_probabilities(
                                                      sphandle.just_data(df_with_prob), '48Ti')).reindex(multipliedcomps.index).sum(axis=1)

        mc = sphandle.non_zero_data(pd.DataFrame(multipliedcompscor))
        mc['label'] = ['Eng' if mc.loc[i,0] < 0 else 'Nat' for i in mc.index]
        mc[0] = abs(mc[0])

        color = []
        for i in mc.sort_values(by = 0, ascending = False)['label']:
            if i == 'Nat':
                color.append('r')
            else:
                color.append('b')


        fig, ax = plt.subplots(1,1, figsize = [7,3], dpi = 300)
        ax.barh(mc.sort_values(by = 0, ascending = False)[:10].index, mc.sort_values(by = 0, ascending = False)[:10][0], color = color)
        ax.text(0.9, 0.9, keys, horizontalalignment = 'right', verticalalignment = 'center', transform = ax.transAxes)
        ax.set_xlabel('Magnitude')

        plt.tight_layout()
        if savefigs == None:
            plt.savefig('figures/elementalimportance' + str(keys) + '.png', dpi = 300) 


        #Finding the composition of Misclassification
        Nat_df_with_prob = df_with_prob[df_with_prob['newlabel'] == 'Natural']
        Nat_df_with_prob_mis = Nat_df_with_prob[Nat_df_with_prob['Natural'] < 0.5]
        Eng_df_with_prob = df_with_prob[df_with_prob['newlabel'] == 'Engineered']
        Eng_df_with_prob_mis = Eng_df_with_prob[Eng_df_with_prob['Engineered'] < 0.5]

        #Printing the size distribution of natural and engineered
        print(naturalkeys, (((( Nat_df_with_prob['48Ti']*(48+32)/48)/ 4.23 * 10 **21)*6/math.pi)**(1/3)).mean(), 
             ((((Nat_df_with_prob['48Ti']*(48+32)/48)/ 4.23 * 10 **21)*6/math.pi)**(1/3)).std())
        print(engineeredkeys, (((( Eng_df_with_prob['48Ti']*(48+32)/48)/ 4.23 * 10 **21)*6/math.pi)**(1/3)).mean(), 
             ((((Eng_df_with_prob['48Ti']*(48+32)/48)/ 4.23 * 10 **21)*6/math.pi)**(1/3)).std())

        #Printing purity of TiO2 of natural and engineered
        print(naturalkeys, sphandle.probability_pure(sphandle.just_data(Nat_df_with_prob).drop(columns = '46Ti'), '48Ti'))
        print(engineeredkeys, sphandle.probability_pure(sphandle.just_data(Eng_df_with_prob).drop(columns = '46Ti'), '48Ti'))


        #Finding the composition of Misclassification
        Nat_df_with_prob = df_with_prob[df_with_prob['newlabel'] == 'Natural']
        Eng_df_with_prob = df_with_prob[df_with_prob['newlabel'] == 'Engineered']
        Nat = sphandle.category_split(Nat_df_with_prob, 'Natural', 0.85, 0.15)
        Eng = sphandle.category_split(Eng_df_with_prob, 'Engineered', 0.85, 0.15)
        total_correct = (len(Nat[0]) + len(Eng[0]))/ len(df_with_prob)
        total_engcorrect = (len(Eng[0]))/ len(Eng_df_with_prob)
        total_uncertain = (len(Nat[1]) + len(Eng[1]))/ len(df_with_prob)
        total_mis = (len(Nat[2]) + len(Eng[2]))/ len(df_with_prob)
        print('Total correct = ' + str(total_correct))
        print('Total Engineered Correct =' + str(total_engcorrect))
        print('Total uncertain = ' + str(total_uncertain))
        print('Total misclassified = ' + str(total_mis))
        piechart_data = [total_correct, total_uncertain, total_mis]
        #piechart_data_labels = 'Classified', 'Uncertain', 'Misclassified'

        #get the top and bottom percent of Nat and Eng
        #parse particles by confidence probability to the correct category
        Nat_df_pure = Nat_df_with_prob.loc[sphandle.isotope_pure(sphandle.just_data(Nat_df_with_prob).drop(columns = '46Ti'), '48Ti').index]
        Eng_df_pure = Eng_df_with_prob.loc[sphandle.isotope_pure(sphandle.just_data(Eng_df_with_prob).drop(columns = '46Ti'), '48Ti').index]

        topnat = Nat_df_with_prob[Nat_df_with_prob['Natural'] > 0.85]
        puretopnat = Nat_df_pure[Nat_df_pure['Natural'] > 0.85]
        botnat = Nat_df_with_prob[Nat_df_with_prob['Natural'] < 0.85]
        purebotnat = Nat_df_pure[Nat_df_pure['Natural'] < 0.85]

        topeng = Eng_df_with_prob[Eng_df_with_prob['Engineered'] > 0.85]
        puretopeng = Eng_df_pure[Eng_df_pure['Engineered'] > 0.85]
        boteng = Eng_df_with_prob[Eng_df_with_prob['Engineered'] < 0.85]
        pureboteng = Eng_df_pure[Eng_df_pure['Engineered'] < 0.85]

        #Plot Ti
        xy_line = (0.15, 0.15)
        yz_line = (0.85, 0.85)
        fig, axs = plt.subplots(1, 3, figsize=(16,5), dpi=300)
        #axs[0].scatter(Nat_df_with_prob.loc[:,'48Ti'], Nat_df_with_prob.loc[:,'Natural'], color = 'brown')
        axs[0].scatter(topnat['48Ti'], topnat['Natural'], linewidths = 0.75, edgecolors = 'white', facecolors = 'green')
        axs[0].scatter(puretopnat['48Ti'], puretopnat['Natural'], linewidths = 0.75, edgecolors = 'green', facecolors = 'white')
        axs[0].scatter(botnat['48Ti'], botnat['Natural'], linewidths = 0.75, edgecolors = 'white', facecolors = 'orange')
        axs[0].scatter(purebotnat['48Ti'], purebotnat['Natural'], linewidths = 0.75, edgecolors = 'orange', facecolors = 'white')
        axs[0].set_xlim(min(Nat_df_with_prob['48Ti']), 1E-13)
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Ti Mass (g)', fontsize = 16)
        axs[0].tick_params(axis = 'both', which = 'major', labelsize = 18)
        #axs[0].set_ylabel('Prediction probability of Natural category', fontsize = 16)
        axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[0].set_title(naturalkeys)
        axs[0].plot(yz_line, 'r--', color = 'black')
        axs[0].set_ylim(0.0, 1.1)
        plt.legend(labels, loc='best')
        #axs[1].scatter(Eng_df_with_prob.loc[:,'48Ti'], Eng_df_with_prob.loc[:,'Engineered'], color = 'blue')
        axs[1].scatter(topeng['48Ti'], topeng['Engineered'], linewidths = 0.75, edgecolors = 'white', facecolors = 'green')
        axs[1].scatter(puretopeng['48Ti'], puretopeng['Engineered'], linewidths = 0.75, edgecolors = 'green', facecolors = 'white')
        axs[1].scatter(boteng['48Ti'], boteng['Engineered'], linewidths = 0.75, edgecolors = 'white', facecolors = 'orange')
        axs[1].scatter(pureboteng['48Ti'], pureboteng['Engineered'], linewidths = 0.75, edgecolors = 'orange', facecolors = 'white')
        axs[1].set_xlim(min(Nat_df_with_prob['48Ti']), 1E-13)
        axs[1].set_xscale('log')
        axs[1].set_xlabel('Ti Mass (g)', fontsize = 16)
        axs[1].tick_params(axis = 'both', which = 'major', labelsize = 18)
        #axs[1].set_ylabel('Prediction probability of Engineered category', fontsize = 16)
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[1].set_title(engineeredkeys)
        axs[1].plot(yz_line, 'r--', color = 'black')
        axs[1].set_ylim(0.0, 1.1)
        fig.tight_layout()

        axs[2].pie(piechart_data, colors = ['green', 'orange', 'red'])
        axs[2].axis('equal')
        fig.set_facecolor('white')
        if savefigs == None:
            plt.savefig('figures/Timass' + str(keys) + '.png', dpi =500)

        topnat= sphandle.conditional_probabilities(sphandle.just_data(Nat_df_with_prob), '48Ti')[2:12]
        topeng= sphandle.conditional_probabilities(sphandle.just_data(Eng_df_with_prob), '48Ti')[2:12]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), dpi=300)
        ax1.bar(topnat.index, topnat)
        ax1.set_ylabel('Isotope Frequency')
        ax1.tick_params(axis='both', which='major', labelsize=12)

        ax2.bar(topeng.index, topeng)
        ax2.set_ylabel('Isotope Frequency')
        ax2.tick_params(axis='both', which='major', labelsize=12)

        fig.tight_layout
        if savefigs == None:
            plt.savefig('figures/isotopes' + str(keys) + '.png', dpi = 300)

        if flag == 1:
            mltrain.graphmaker(Nat_df_with_prob, 'Natural', 0.85, keys, Nat_df_with_prob, savefigs, opencircles = True)
            mltrain.graphmaker(Eng_df_with_prob, 'Engineered', 0.85, keys, Nat_df_with_prob, savefigs, opencircles = True)
        else:
            mltrain.graphmakerm(Nat_df_with_prob, 'Natural', 0.85, 0.5, keys, Nat_df_with_prob, savefigs, opencircles = True)
            mltrain.graphmakerm(Eng_df_with_prob, 'Engineered', 0.85, 0.5, keys, Nat_df_with_prob, savefigs, opencircles = True)


        if returns != None:
            return clf, df_with_prob, multipliedcomps, df, labels


    #bootstrapping process. Holdout some data, predict on it and do that N amount of times. Repeated this procedure M amount of times
    @staticmethod
    def bootstrap(df, N, m):
        #repeat this process M amount of times
        for _ in range(m):
            #holdout 20% of data. 
            holdoutdata, holdoutlabels, trainingdatanat, trainingdataeng = mltrain.holdoutdata(df, 0.2)
            #create a NP array from 0.05 to 1.05 with 0.05 step increment
            p = np.arange(0.05, 1.05, 0.05)
            #create an empty dataframe
            df1 = pd.DataFrame()
            #for each 0.05 step from p
            for i in p:
                #create an empty list
                percent = []
                #repeat this N amount of times
                for _ in range(N):
                    #sample an i fraction of the training data
                    snat = trainingdatanat.sample(frac = i, replace = True)
                    seng = trainingdataeng.sample(frac = i, replace = True)

                    #combine training data and get traininglabels

                    trainingdata = pd.concat([snat, seng], axis=0)
                    traininglabels = trainingdata.index.get_level_values(level='newlabel')

                    #create the logistic Regression model with 5-fold cross validation with NMF of 10 componenets

                    clf = make_pipeline(
                        StandardScaler(with_mean=False),
                        NMF(n_components=10),
                        LogisticRegressionCV(cv=5, max_iter=300)
                    )
                    #fit with fraction of training set
                    clf = clf.fit(trainingdata, traininglabels)
                    #score with the heldout test set
                    y_score = clf.score(holdoutdata, holdoutlabels)
                    #append the score to percent
                    percent.append(y_score)
                    #put it in a dataframe format
                    pcdf = pd.DataFrame(percent)
                df1 = pd.concat([df1, pcdf], axis = 1)

            df1.columns = np.arange(0.05, 1.05, 0.05)

            average = []
            stdev = []
            for i in df1.columns:
                average.append(df1[i].mean())
                stdev.append(df1[i].std())
            plt.errorbar(np.arange(0.05, 1.05, 0.05), average, yerr = stdev)
            plt.xlabel('% of training data')
            plt.ylabel('Accuracy of model (%)')
            
    @staticmethod
    def pca_analysis(df, label):
        pca = PCA()
        X = pca.fit_transform(df)
        x1 = X[:, 0]
        x2 = X[:, 1]

        fig, ax = plt.subplots()
        ax.plot(x1, x2, '.')
        ax.set(
            xlabel="First component",
            ylabel="Second component",
            title="%s: projected onto first two components" % label,
        )

        components = pd.DataFrame(pca.components_.T, index=df.columns)
        print_until = 0.9999
        cumulative_explained_variance = 0
        i = 0
        while cumulative_explained_variance < print_until:
            explained = pca.explained_variance_ratio_[i]
            print("=> COMPONENT %d explains %0.2f%% of the variance" % (i, 100*explained))
            top_components = components[i].sort_values(ascending=False)
            top_components = top_components[abs(top_components) > 0.0]
            print(top_components)
            i += 1
            cumulative_explained_variance += explained
        return components