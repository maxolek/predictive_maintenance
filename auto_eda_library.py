import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import pearsonr, skew, kurtosis
from scipy.stats import boxcox_normplot, shapiro, normaltest, gaussian_kde, probplot, boxcox
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

###################################################################################################################
## Anything that involves preprocessing (e.g. transformations, scaling, modeling) !!! must not use !!! the values #
## Generated from those preprocessing results as this will result in !!! data leakage !!!                         #
##                                                                                                                #
## The same transformations can be applied they just must be fit/generated from the training data only           #
## Then the now defined transformation is applied to the test set without refitting                              #
##################################################################################################################


####################
## Pre-processing ##
####################
def loadData(data_path):
    # idea: adding *args and **kwargs for functions and parameters to pass/apply in data loading
    df = pd.read_csv(data_path)
    print(df.shape)
    display(df.head())
    print(" ---------- Data Types ----------")
    display(df.dtypes)
    print(" ---------- Nulls ----------")
    display(df.isna().sum())
    print()
    print(r"% of data")
    print()
    display(df.isna().sum()/len(df))
    print(" ---------- Descriptive Statistics ----------")
    display(df.describe())
    return df

def numericMapping(df,features,targets,one_hot=False):
    temp_df = df.copy()[features+targets]
    non_numeric_features = [col for col in temp_df[features].columns if not is_numeric_dtype(temp_df[col])]
    non_numeric_targets = [col for col in temp_df[targets].columns if not is_numeric_dtype(temp_df[col])]

    if one_hot:
        # feature transformation
        temp_df = pd.get_dummies(temp_df,columns=non_numeric_features)
        # target transformation
        le = LabelEncoder()
        for target in non_numeric_targets:
            temp_df[target] = le.fit_transform(temp_df[target])
            # print the mapping so it can be recorded
            print(f" ----- {target} -----")
            print(dict(zip(le.classes_,le.transform(le.classes_))))
    else:
        for col in [non_numeric_features + non_numeric_targets][0]:
            le = LabelEncoder()
            temp_df[col] = le.fit_transform(temp_df[col])
            print(f" ----- {col} -----")
            print(dict(zip(le.classes_,le.transform(le.classes_))))

    return temp_df

###############
## Target(s) ##
###############
def plotTargets(df,targets):
    fig, ax = plt.subplots(nrows=len(targets)//2,ncols=2,figsize=(16.5,2.5+2*(len(targets)//2)))
    total_points = len(df)

    for i in range(len(targets)//2):
        for j in range(2):
            if len(targets)//2 > 1:
                plot = df[targets[2*i+j]].value_counts().plot(kind='bar',ax=ax[i][j])
                ax[i][j].set_title(targets[2*i+j])
            else:
                plot = df[targets[2*i+j]].value_counts().plot(kind='bar',ax=ax[j])
                ax[j].set_title(targets[2*i+j])

            for p in plot.patches:
                width = p.get_width()
                height = p.get_height()
                x,y = p.get_xy()
                plot.annotate(f"{height}\n{np.round((height/total_points)*100,2)}%",(x+width/2,y+height*1.02),ha='center')

    fig.tight_layout()
    plt.show()
    return

# correlation heatmap for features and target variables          
def corrWithTargets(df,features,targets):
    fig, ax = plt.subplots(nrows=len(targets),ncols=1,figsize=(12.5,4.5+2*len(targets)))

    i = 0
    if len(targets) > 1:
        for i in range(len(targets)):
            sns.heatmap(pd.DataFrame(df[features].corrwith(df[targets[i]]),columns=[targets[i]]),ax=ax[i],annot=True,cmap='vlag',fmt='.2f')
    else:
        sns.heatmap(pd.DataFrame(df[features].corrwith(df[targets[i]]),columns=[targets[i]]),ax=ax,annot=True,cmap='vlag',fmt='.2f')
    
    fig.tight_layout()
    plt.show()
    return

# warning: large sample sizes can lead to statistically significant correlations when they may not actually be 'significant'
def printSignificantCorrsWithTargets(df,features,targets,alpha=.05):
    for feat in features:
        for tgt in targets:
            c = pearsonr(df[feat],df[tgt])
            if c[1] < alpha:
                print(f"Feature: {feat}\nTarget: {tgt}\nCorrelation: {c[0]}\np-value: {c[1]}")
                print('------------------------')

    return

##############
## Features ##
##############
def generateFeaturesNormalAttributes(df,features):
    kurt_diff = {}
    skews = {}
    comb = {}
    for feat in features:
        kurt_diff[feat] = kurtosis(df[feat])
        skews[feat] = skew(df[feat])
        comb[feat] = kurt_diff[feat]**2 + skews[feat]**2

    #output_df = pd.DataFrame(data={'Kurtosis Deviation':kurt_diff,'Skew':skew,'Kurtosis_Deviation**2 + Skew**2':comb},index=)
    output_df = None
    return kurt_diff, skew, comb, output_df 

# correlation heatmap between features
def featureCorrs(df,features):
    return sns.heatmap(df[features].corr(),annot=True,cmap='vlag',fmt='.2f')

# warning: large sample sizes can lead to statistically significant correlations when they may not actually be 'significant'
def printSignificantCorrsFeatures(df,features,alpha=.05):
    for feat in range(len(features)):
        for feat2 in range(feat+1,len(features)):
            c = pearsonr(df[features[feat]],df[features[feat2]])
            if c[1] < alpha:
                print(f"Feature 1: {features[feat]}\nFeature 2: {features[feat2]}\nCorrelation: {c[0]}\np-value: {c[1]}")
                print('------------------------')

    return

# plotting the distributions of the features
# target = target column name (e.g. 'Target', 'Failure Type')
def featureDistributionPlots(df,features,categorical_feautres = [],kde=False,target=None):
    if not kde and target != None: print("It is recommended to use kde when splitting by target if the target is not evenly distributed.")
    fig, ax = plt.subplots(nrows=len(features)//2+1,ncols=2,figsize=(16.5,2.5+2*(len(features)//2)))

    for i in range(len(features)//2+1):
        for j in range(2):
            if 2*i+j == len(features):
                return
            if target is not None:
                if is_numeric_dtype(df[features[2*i+j]]) and features[2*i+j] not in categorical_feautres:
                    if kde:
                        for target_val in pd.unique(df[target]):
                            data = df[df[target]==target_val][features[2*i+j]]
                            x = np.linspace(min(data),max(data),1000)
                            kde = gaussian_kde(data.sample(min(10000,len(data))))
                            ax[i][j].plot(x,kde(x),label=target_val)
                    else: 
                        for target_val in pd.unique(df[target]):
                            ax[i][j].hist(df[df[target]==target_val][features[2*i+j]],label=target_val)
                    ax[i][j].set_title(features[2*i+j])      
                    ax[i][j].legend()         
                else:
                    plot = sns.countplot(df,x=features[2*i+j],hue=target,ax=ax[i][j])

                    for p in plot.patches:
                        width = p.get_width()
                        height = p.get_height()
                        x,y = p.get_xy()
                        plot.annotate(f"{height}\n{np.round((height/len(df))*100,2)}%",(x+width/2,y+height*1.02),ha='center')
            else:
                if is_numeric_dtype(df[features[2*i+j]]) and features[2*i+j] not in categorical_feautres:
                    if kde:
                        data = df[features[2*i+j]]
                        x = np.linspace(min(data),max(data),1000)
                        kde = gaussian_kde(data.sample(min(10000,len(data))))
                        ax[i][j].plot(x,kde(x))
                        ax[i][j].set_title(features[2*i+j])
                    else:
                        ax[i][j].hist(df[features[2*i+j]])
                        ax[i][j].set_title(features[2*i+j])
                else:
                    plot = sns.countplot(df,x=features[2*i+j],ax=ax[i][j])

                    for p in plot.patches:
                        width = p.get_width()
                        height = p.get_height()
                        x,y = p.get_xy()
                        plot.annotate(f"{height}\n{np.round((height/len(df))*100,2)}%",(x+width/2,y+height*.9),ha='center')

                    ax[i][j].set_title(features[2*i+j])

    fig.tight_layout()
    plt.show()
    return

# boxplots for features with ability to have target splitting (and which target if multiple)
def featureBoxPlots(df,features,categorical_features=[],target=None):
    fig, ax = plt.subplots(nrows=len(features)//4+1,ncols=4,figsize=(20.5,2+2.5*(len(features)//4)))
    non_cat_feats = [_ for _ in features if _ not in categorical_features]

    for i in range(len(features)//4+1):
        for j in range(4):
            if 4*i+j == len(features):
                return
            if target is not None:
                sns.boxplot(data=df,y=non_cat_feats[4*i+j],x=target,ax=ax[i][j],orient='v')
                ax[i][j].set_title(non_cat_feats[4*i+j])
            else:
                sns.boxplot(data=df,y=non_cat_feats[4*i+j],orient='v',ax=ax[i][j])
                ax[i][j].set_title(non_cat_feats[4*i+j])
        
    plt.show()
    fig.tight_layout()
    return 

# performs different normal tests and provides a normality plot for each feature
def normalityTesting(df,features,alpha=.05):
    print(" --------------------------- Normal Probability Plot --------------------------- ")
    fig, ax = plt.subplots(nrows=len(features)//4+1,ncols=4,figsize=(20.5,2.5+2*(len(features)//4)))
    for i in range(len(features)//4+1):
        for j in range(4):
            if i*4+j == len(features):
                break
            probplot(df[features[i*4 + j]],dist='norm',plot=ax[i][j])
            ax[i][j].set_title(features[i*4 + j])
        if i*4+j == len(features):
            break
    fig.tight_layout()
    plt.show()

    print(" --------------------------- Normal Test --------------------------- ")
    for feat in features:
        norm = normaltest(df[feat])
        if norm[1] < alpha:
            print(f" ----------- {feat} ----------- ")
            print(norm)
            plt.hist(df[feat])
            plt.title(feat)
            plt.show()    

    print(" --------------------------- Shapiro --------------------------- ")
    for feat in features:
        shpro = shapiro(df[feat])
        if shpro[1] < alpha:
            print(f" ----------- {feat} ----------- ")
            print(shpro)
            plt.hist(df[feat])
            plt.title(feat)
            plt.show()


        # interesting concept (but will not be implemented here) is a neural network
        # that takes a feature as input and outputs a transformation 
        # that leads to a better prediction (likely just aim for normal dist)
        # ... all meaning behind the transformed feature would be lost
        # but predictions could be better
        # and would autonamize the transformation stage in a project
        # ... not good for understanding the model but good for prediction (theoretically)
        # ... take n (=50?) sample from feature and input into neural network, n input nodes
        # have neural network apply transformation to make it normal (loss = linear combination of normal tests?)
        # ... test by perfect easy transforms (inverse, log, boxcox, etc)
        # and others that not perfect or easy transforms (get good enough type transforms)
        # and evaluate how much improvement (anyway to verify if this is good improvement? can set up a set that have been done the hard way to get best approx which is used to test against)

    return 

# transforms each feature under different transformation types and reports if any provided better kurtosis/skew metric
# transformation results and suggestions are based purely on normality, not association with target(s)
def transformationPlots(data,features,targets,p1=True,plots=True):  #p1 adds the minimum to the feature in order to turn it positive to prevent complex results of log transform
    # boxcox and log don't allow for 0 values
    df = data.copy() # need dt and temp_df for plotting (original distribution is first col of plots)
    if targets: df.drop(columns=targets,inplace=True,errors='ignore')
    df.replace({0:None},inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    output = pd.DataFrame(columns=['Original','Box-Cox','Yeo-Johnson','Log','Exponential','Box-Cox lambda','Yeo-Johnson lambda','Log base','Log constant','Exponential lambda'],index=features)

    _k, _s, feature_combination, _df = generateFeaturesNormalAttributes(data,features)

    for feat in features:       
        single_feat = df[[feat]].dropna()

        if plots:                                
            fig, ax = plt.subplots(nrows=4,ncols=3,figsize=(22.5,10.5))
            print(f"--------------------------- {feat} ---------------------------")              
            print(f"Original k**2 + s**2: {feature_combination[feat]}")              

        if p1: 
            const = -single_feat[feat].min() + .001 
            if const - .001 > 0:
                single_feat[feat] = single_feat[feat] + const 
            else:
                const = 0
            output.loc[feat,'Log constant'] = const
        else:
            const = 0
                                                                 # f(x) = 1/lambda * (x^lambda - 1)       if p1 then f(x) = 1/lambda * ((x+1)^lambda - 1) 
        #############                                            # lambda = 1/2 --> f(x) = 2*(x^(1/2) - 1) <--> sqrt
        ## Box-Cox ##                                            # lambda = 0 --> f(x) = ln(x) <--> (natural) log
        #############                                            # lambda = -1 --> f(x) = -1*(x^-1 -1) <--> inverse 

        normplot_result = boxcox_normplot(single_feat[feat],-10,10)                                         
        la = normplot_result[0][list(normplot_result[1]).index(np.max(normplot_result[1]))]   
    
        transformed = boxcox(single_feat[feat],la)

        if plots:
            print(f"Box-Cox: {kurtosis(transformed)**2 + skew(transformed)**2}")
            ax[0][0].hist(data[feat])
            ax[0][0].set_title('Original')
            ax[0][1].hist(transformed)
            ax[0][1].set_title(f'Transformed:\nlambda = {la}')
            ax[0][2].plot(normplot_result[0], normplot_result[1])
            ax[0][2].set_title("Box-Cox")

        output.loc[feat,'Original'] = feature_combination[feat]
        output.loc[feat,'Box-Cox'] = kurtosis(transformed)**2 + skew(transformed)**2
        output.loc[feat,'Box-Cox lambda'] = la

        #################
        ## Yeo-Johnson ##
        #################
    
        transformer = PowerTransformer()
        try: # can run into error where dataset has large values that overflow float dtype after transformation                                                                           
            transformed = transformer.fit_transform(np.array(single_feat[feat]).reshape(-1,1))   
            output.loc[feat,'Yeo-Johnson'] = kurtosis(transformed)**2 + skew(transformed)**2
            output.loc[feat,'Yeo-Johnson lambda'] = transformer.lambdas_[0]

            if plots:
                print(f"Yeo-Johnson: {kurtosis(transformed)**2 + skew(transformed)**2}")
                ax[1][0].hist(data[feat])
                ax[1][0].set_title('Original')
                ax[1][1].hist(transformed)
                ax[1][1].set_title(f'Transformed:\nlambda = {transformer.lambdas_[0]}')
                ax[1][2].set_title('Yeo-Johnson')
        except:        
            # problem can be solved by scaling beforehand, but how to translate this outside of the function so dataset is scaled in generation ???
            output.loc[feat,'Yeo-Johnson'] = np.nan
            output.loc[feat,'Yeo-Johnson lambda'] = np.nan  


        #########
        ## Log ##
        ######### 

        comb = []                                               # f(x) = log_{base}(x) where base is iterated over
        bases = np.linspace(1.5,10,100)

        for base in bases:
            tmp = np.emath.logn(base,single_feat[feat]+const)
            comb.append(kurtosis(tmp)**2+skew(tmp)**2)

        best_base_idx = comb.index(np.min(comb))
        best_base = bases[best_base_idx]

        if plots:
            print(f"Log: {comb[best_base_idx]}")
            ax[2][0].hist(data[feat])
            ax[2][0].set_title('Original')
            ax[2][1].hist(np.emath.logn(best_base,single_feat[feat]))
            ax[2][1].set_title(f'Transformed:\nbase = {best_base}')
            ax[2][2].plot(bases, comb)
            ax[2][2].set_title("Log")
            ax[2][2].set_ylabel("kurtosis**2 + skew**2")

        output.loc[feat,'Log'] = comb[best_base_idx]
        output.loc[feat,'Log base'] = best_base

        #################
        ## Exponential ##
        #################

        comb = []                                    # f(x) = lambda*e^(-lambda*x)
        lambdas = np.linspace(.01,5,1000)

        for la in lambdas:
            tmp = la*np.e**(-la*single_feat[feat])
            if np.isnan(kurtosis(tmp)):
                continue
            comb.append(kurtosis(tmp)**2+skew(tmp)**2)

        if len(comb) > 0:
            best_la_idx = comb.index(np.min(comb))
            best_la = lambdas[best_la_idx]
        else:
            comb = [np.nan]
            best_la_idx = 0
            best_la = np.nan 

        if plots:
            print(f"Exponential: {comb[best_la_idx]}")
            ax[3][0].hist(data[feat])
            ax[3][0].set_title('Original')
            ax[3][1].hist(best_la*np.e**(-best_la*df[feat]))
            ax[3][1].set_title(f'Transformed:\nlambda = {best_la}')
            ax[3][2].plot(lambdas[:len(comb)], comb)
            ax[3][2].set_title("Exponential")
            ax[3][2].set_ylabel("kurtosis**2 + skew**2")


        output.loc[feat,'Exponential'] = comb[best_la_idx]
        output.loc[feat,'Exponential lambda'] = best_la

        if plots:
            fig.tight_layout()
            plt.show()   

    return output