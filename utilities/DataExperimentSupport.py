import numpy as np
import pandas as pd
import decimal
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import random
import math
from tqdm.notebook import tqdm
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
import lime
from lime import lime_tabular
from matplotlib.pyplot import xticks
from wordcloud import WordCloud
from IPython.display import HTML
from keras import backend as K 



def createModel(data,
                uniqueColumn,
                targetColumn,
                untrained_model,
                experiment_method):
    tDf = data.copy()

    # Get Y value from dataframe
    Y = np.array(tDf[targetColumn])

    # Drop unneeded columns for model training
    tDf.drop([uniqueColumn, targetColumn], axis=1, inplace=True)

    # Get X Value from rest of dataframe
    X = tDf.to_numpy()

    # fit model on training data
    model = copy.deepcopy(untrained_model)

    if experiment_method == 'supervised':
        _ = model.fit(X, Y)

    elif experiment_method == 'unsupervised':
        _ = model.fit(X)
    else:
        print("Invalid experiment_method detected")

    return model

def createTensorModel(data,
                      uniqueColumn,
                      targetColumn,
                      untrained_model,
                      tensor_parms):
    
    tDf = data.copy()

    # Get Y value from dataframe
    Y = np.array(tDf[targetColumn])

    # Drop unneeded columns for model training
    tDf.drop([uniqueColumn, targetColumn], axis=1, inplace=True)

    # Get X Value from rest of dataframe
    X = tDf.to_numpy()

    # fit model on training data
    #model = copy.deepcopy(untrained_model)
    model = untrained_model

    # print(f'tensor_parms.loss={tensor_parms.loss}')
    # print(f'tensor_parms.optimizer={tensor_parms.optimizer}')
    # print(f'tensor_parms.metrics={tensor_parms.metrics}')

    #K.clear_session()
    model.compile(loss=tensor_parms.loss,
                  optimizer=tensor_parms.optimizer,
                  metrics=tensor_parms.metrics)
    
    history = model.fit(X, Y, 
                        validation_split=tensor_parms.validation_split,
                        epochs=tensor_parms.epochs,
                        batch_size=tensor_parms.batch_size,
                        shuffle=tensor_parms.shuffle,
                        verbose=tensor_parms.verbose)

    return model, history

def predictModel(model,
                 data,
                 uniqueColumn,
                 targetColumn,
                 colActual='y_test',
                 colPredict='y_pred',
                 use_argmax=False):
    tDf = data.copy()

    Y = np.array(tDf[targetColumn])

    # Drop unneeded columns for model testing
    tDf.drop([uniqueColumn, targetColumn], axis=1, inplace=True)

    # Get X Value from rest of dataframe
    X = tDf.to_numpy()

    
    if use_argmax: # Tensor style prediction
        predictions = model.predict(X)
        y_pred = predictions.argmax(axis=1)
    else: 
        y_pred = model.predict(X)

    # make a dataframe for the results
    tDf = pd.DataFrame(data=Y, columns=[colActual])
    tDf[colPredict] = y_pred.tolist()

    return tDf, colActual, colPredict


def predictProba(model,
                 data,
                 uniqueColumn,
                 targetColumn,
                 colActual='y_test',
                 colPredict='y_pred'):
    tDf = data.copy()

    Y = np.array(tDf[targetColumn])

    # Drop unneeded columns for model testing
    tDf.drop([uniqueColumn, targetColumn], axis=1, inplace=True)

    # Get X Value from rest of dataframe
    X = tDf.to_numpy()

    y_prob = model.predict_proba(X)
    #print(f'predictProba: returning {y_prob}')
    return y_prob






# Creates a dataframe with two columns:
# feature_idx: index of features
# importance: importance value of the relevent feature
def getModelFeatureImportance(model,
                              featureLabel='feature_idx',
                              valueLabel='importance',
                              ):
    if hasattr(model, 'feature_importances_'):
        # Create a dataframe with feature importances
        impDf = pd.DataFrame(data=model.feature_importances_, columns=[valueLabel])
        impDf.reset_index(inplace=True)
        impDf.rename(columns={'index': featureLabel}, inplace=True)
    elif hasattr(model, 'coef_'):
        w = model.coef_[0]
        impDf = pd.DataFrame(pow(math.e, w), columns=[valueLabel])
        # feature_importance['importance'] = w
        impDf.reset_index(inplace=True)
        impDf.rename(columns={'index': featureLabel}, inplace=True)

    else:
        raise "no feature_importances_ or coef_ found in model"

    return impDf, featureLabel, valueLabel


# plots importance of features in given model
def analyzeModelFeatureImportance(data,
                                  valueLabel='importance',
                                  startValue=0.0001,
                                  increment=0.0001,
                                  upperValue=0.01,
                                  returnAbove=0.002,
                                  showSummary=True,
                                  showPlot=True):
    tqdm.pandas()
    xAxisLabel = 'Feature Importance'
    recCountLabel = 'Number of Features'
    dx = startValue

    # calc the number of decimals to round the value
    d = decimal.Decimal(str(increment))
    roundValue = d.as_tuple().exponent * -1

    # Create the list with initial value of 0
    xAxisVal = [startValue]

    while dx <= upperValue:
        # add to the list of xAxisValues
        xAxisVal.append(dx)

        dx += increment
        # round value included due to errors in FP addition
        dx = round(dx, roundValue)

    # turn list into dataframe
    tDf = pd.DataFrame(xAxisVal, columns=[xAxisLabel])

    # Add in column for number of features >= that value
    tDf[recCountLabel] = tDf.progress_apply(lambda x:
                                            len(data.loc[data[valueLabel] >= x[xAxisLabel]]),
                                            axis=1
                                            )

    # return a list of features to be used in an optimized model
    # it's by feature index
    tDf2 = data.loc[data[valueLabel] >= returnAbove].copy()
    tDf2.reset_index(drop=True, inplace=True)

    if showSummary:
        # Give some sort of summary
        indent = '---> '
        print('Feature Importance Summary:')
        print(f'{indent}Original feature count: {len(data)}')
        print(f'{indent}Returned feature count: {len(tDf2)}')
        print(f'{indent}Removed feature count: {len(data) - len(tDf2)}')
        print(f'{indent}Return items above (including): {returnAbove}')

    if showPlot:
        # Plot it after the summary
        tDf.plot(x=xAxisLabel,
                 y=recCountLabel,
                 ylabel='Number of Features',
                 title='Total Features >= Importance Level')

    return tDf2


def showAllModelFeatureImportance(data,
                                  featureLabel,
                                  valueLabel,
                                  xlim=None):
    # if len(data) > 100:
    #    recLimit = 25
    # else:
    #    recLimit = max(len(data),25)

    recLimit = 25

    if xlim is None:
        xlim = .03

    newFeatLabel = featureLabel + '_s'
    tDf = data.sort_values(by=valueLabel, ascending=False).head(recLimit).copy()
    tDf[newFeatLabel] = tDf.apply(lambda x:
                                  'feature_' + str(x[featureLabel]),
                                  axis=1
                                  )

    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 10))

    # Plot the todtal crashes
    # sns.set_color_codes("pastel")
    sns.barplot(x=valueLabel,
                y=newFeatLabel,
                data=tDf,
                label="Total",
                palette="crest").set(title=f'Model Feature Importance (top {recLimit})')

    plt.tick_params(axis='y', labelsize=10)

    # ax.legend(ncol=1, loc="lower right", frameon=True)
    ax.set(xlim=(0, xlim),
           ylabel="",
           xlabel="Feature Importance")
    # sns.despine(left=True, bottom=True)
    sns.despine()
    _ = plt.show()
    plt.clf()


def showConfusionMatrix(data,
                        colNameActual,
                        colNamePredict,
                        axis_labels,
                        titleSuffix,
                        cmap='mako',
                        plotsize=2):
    plt.clf()

    cm = confusion_matrix(np.array(data[colNameActual]).reshape(-1, 1),
                          np.array(data[colNamePredict]).reshape(-1, 1)
                          )

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap=cmap,
                xticklabels=axis_labels,
                yticklabels=axis_labels
                )

    if plotsize == 5:
        sns.set(rc={'figure.figsize': (20, 8)})
    elif plotsize == 4:
        sns.set(rc={'figure.figsize': (15, 8)})
    elif plotsize == 3:
        sns.set(rc={'figure.figsize': (10, 8)})
    elif plotsize == 2:
        sns.set(rc={'figure.figsize': (8, 8)})
    elif plotsize == 1:
        sns.set(rc={'figure.figsize': (4, 8)})
    else:  # Should be size 1
        # should only be one but catch it and default to size 1
        sns.set(rc={'figure.figsize': (4, 4)})

    # title with fontsize 20
    plt.title(f'Confusion Matrix: {titleSuffix}', fontsize=20)
    # x-axis label with fontsize 15
    plt.xlabel('Predicted', fontsize=15)
    # y-axis label with fontsize 15
    plt.ylabel('Actual', fontsize=15)
    _ = plt.show()
    plt.clf()


def showReport(data, colNameActual, colNamePredict, axis_labels, titleSuffix):
    # results = metrics.classification_report(pd.to_numeric(data[colNameActual]).to_list(),
    #                                        data[colNamePredict].to_list(),
    #                                        zero_division=0)

    results = metrics.classification_report(data[colNameActual].to_list(),
                                            data[colNamePredict].to_list(),
                                            zero_division=0)

    print(results)

    showConfusionMatrix(data=data,
                        colNameActual=colNameActual,
                        colNamePredict=colNamePredict,
                        axis_labels=axis_labels,
                        titleSuffix=titleSuffix
                        )


def showROCAUC(dataTrain,
               dataTest,
               classifier,
               axis_labels,
               colNameActual,
               features):
    model = classifier
    visualizer = ROCAUC(model, classes=axis_labels)

    # Fit the training data to the visualizer
    visualizer.fit(dataTrain[features],
                   dataTrain[colNameActual]
                   )
    # Evaluate model
    visualizer.score(dataTest[features],
                     dataTest[colNameActual]
                     )

    visualizer.show()

    return visualizer


def showFeatureImportance(model,
                          XTrain,
                          YTrain,
                          topn=5,
                          useLasso=False):
    if useLasso:
        viz = FeatureImportances(Lasso(), relative=False, topn=topn)
    else:
        viz = FeatureImportances(model, topn=topn)

    plt.close()
    viz.fit(XTrain, YTrain)
    viz.show()
    plt.clf()


def create_learning_curve(estimator,
                          X,
                          y,
                          cv=None,
                          n_jobs=None,
                          train_sizes=None,
                          verbose=4):
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.5, 1.0]

    if cv is None:
        cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        verbose=verbose
    )
    return train_sizes, train_scores, test_scores, fit_times


def plot_learning_curve(train_sizes,
                        train_scores,
                        test_scores,
                        fit_times,
                        title,
                        axes=None,
                        ylim=None
                        ):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    """
    plt.clf()

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    _ = plt.show()
    plt.clf()

def getModelAccuracy(data, colActual, colPredict):
    accuracy = accuracy_score(data[colActual],
                              data[colPredict])

    return accuracy

def getModelAccuracyBalanced(data, colActual, colPredict):
    accuracy = balanced_accuracy_score(data[colActual],
                                      data[colPredict])

    return accuracy


def getModelPrecision(data, colActual, colPredict, average='weighted'):
    precision = precision_score(data[colActual],
                                data[colPredict],
                                average=average)

    return precision


def getModelRecall(data, colActual, colPredict, average='weighted'):
    recall = recall_score(data[colActual],
                          data[colPredict],
                          average=average)

    return recall

def getAUC(data, colActual, predictProba, average='macro'):
    auc = roc_auc_score(data[colActual],
                        predictProba,
                        multi_class='ovr',
                        average=average)
    print(f'getAUC: returning {auc}')
    return auc

def getModelF1(data, colActual, colPredict, average='weighted'):
    f1 = f1_score(data[colActual],
                  data[colPredict],
                  average=average)

    return f1


def getModelCohenKappa(data, colActual, colPredict):
    ck = cohen_kappa_score(data[colActual],
                           data[colPredict])

    return ck


def printMetrics(data, colActual, colPredict):
    print(f'Accuracy: {round(getModelAccuracy(data=data, colActual=colActual, colPredict=colPredict), 2)}')
    print(f'Precision: {round(getModelPrecision(data=data, colActual=colActual, colPredict=colPredict), 2)}')
    print(f'Recall: {round(getModelRecall(data=data, colActual=colActual, colPredict=colPredict), 2)}')
    print(f'F1: {round(getModelF1(data=data, colActual=colActual, colPredict=colPredict), 2)}')
    print(f'CohenKappa: {round(getModelCohenKappa(data=data, colActual=colActual, colPredict=colPredict), 2)}')


def showPrecisionRecallCurve(model,
                             XTrain,
                             YTrain,
                             XTest,
                             YTest):
    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(model, per_class=False)
    viz.fit(XTrain, YTrain)
    viz.score(XTest, YTest)
    viz.show()
    plt.clf()


def getLimeExplainer(XTrain,
                     features,
                     mode='classification'):
    XTrain = XTrain.to_numpy()
    explainer = lime_tabular.LimeTabularExplainer(XTrain,
                                                  mode=mode,
                                                  # class_names=[0,1],
                                                  feature_names=np.array(features)
                                                  )

    print(explainer)
    return explainer


def showLimeGlobalImportance(XTrain,
                             YTrain,
                             features):
    lr = LogisticRegression()
    scaler = MinMaxScaler()
    XTrain_scale = scaler.fit_transform(XTrain)

    lr.fit(XTrain_scale, YTrain)

    with plt.style.context("ggplot"):
        _ = plt.figure(figsize=(10, 15))
        plt.barh(range(len(lr.coef_[0])), lr.coef_[0], color=["red" if coef < 0 else "green" for coef in lr.coef_[0]])
        plt.yticks(range(len(lr.coef_[0])), features)
        plt.title("Global Importance: Weights")

    plt.show()
    plt.clf()


def showLimeLocalImportance(XTrain,
                            YTrain,
                            XTest,
                            YTest,
                            features,
                            mode):
    # XTrain = XTrain.to_numpy()
    # YTrain = YTrain.to_numpy()
    XTest = XTest.to_numpy()
    YTest = YTest.to_numpy()

    explainer = getLimeExplainer(XTrain=XTrain,
                                 features=features,
                                 mode=mode)

    lr = LogisticRegression()
    lr.fit(XTrain, YTrain)

    preds = lr.predict(XTest)

    false_preds = np.argwhere((preds != YTest)).flatten()

    idx = random.choice(false_preds)

    print(f'predicted {lr.predict(XTest[idx].reshape(1, -1))}')
    print(f'actual {YTest[idx]}')

    explanation = explainer.explain_instance(XTest[idx], lr.predict_proba)

    explanation.show_in_notebook()


#    from IPython.display import HTML
#    html_data = explanation.as_html()
#    HTML(data=html_data)


def plot_history(history):
    plt.style.use('ggplot')
    print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    auc = history.history['auc']
    val_auc = history.history['val_auc']
    mse = history.history['mse']
    val_mse = history.history['val_mse']

    precision = history.history['precision']
    val_precision = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']

    x = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(x, mse, 'b', label='Training mse')
    plt.plot(x, val_mse, 'r', label='Validation mse')
    plt.title('Training and validation MSE')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(x, auc, 'b', label='Training AUC')
    plt.plot(x, val_auc, 'r', label='Validation AUC')
    plt.title('Training and validation AUC')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(x, precision, 'b', label='Training precision')
    plt.plot(x, val_precision, 'r', label='Validation precision')
    plt.title('Training and validation precision')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(x, recall, 'b', label='Training recall')
    plt.plot(x, val_recall, 'r', label='Validation recall')
    plt.title('Training and validation recall')
    plt.legend()

    _ = plt.show()
    plt.close()


def show_model_summary(data_frame,
                       id_var,
                       value_vars,
                       individual=False,
                       title='Experiment comparison'):
    value_name = 'value_name'
    var_name = 'Metric'

    dataframe_rows = len(data_frame)

    tDF = pd.melt(data_frame,
                  id_vars=id_var,
                  value_vars=value_vars,
                  var_name=var_name,
                  value_name=value_name
                  )

    plt.close()

    # plt.figure(figsize=(10, 10) )
    # fig, ax = plt.subplots(figsize=(10,10))
    aspect_ratio = 0.4 + dataframe_rows*0.2
    gfg = sns.catplot(x=id_var,
                      y=value_name,
                      hue=var_name,
                      data=tDF,
                      kind='bar',
                      height=8,
                      aspect=aspect_ratio,
                      palette='coolwarm_r')

    gfg.set(ylim=(0, 1))
    if individual:
        gfg.set(xlabel=None,
                xticklabels=[],
                ylabel="Metric Value",
                title=title)
    else:
        gfg.set(xlabel='Experiment', ylabel="Metric Value", title=title)
    plt.show()

    records_to_show = len(data_frame)
    display(HTML(data_frame.to_html(index=False)))
    #display(data_frame.style.hide_index().head(records_to_show))



def getWordCloud(text):
    if len(text) == 0:
        genText = "[No_Text]"
    else:
        genText = text

    return WordCloud().generate(str(genText))

# Show comparison for unsupervised model with a target column
# Used for determining how to map predicted in unsupervised to
# an actual target columnn
def show_cluster_mapping_cloud(srcDF,
                               textCol,
                               colNameActual,
                               colNamePredict):
    
    # Find out how many unique values we have to deal with
    a = srcDF[colNameActual].unique().tolist()
    b = srcDF[colNamePredict].unique().tolist()
    uniqueVals = list(set(a + b))
    uniqueVals.sort()

    plt.close()
    fig, axs = plt.subplots(len(uniqueVals), 2, figsize=(20, 10))
    plt.tight_layout()

    for x in uniqueVals:
        # get Text for actual
        tmpDFActual = srcDF[srcDF[colNameActual] == x].copy()
        textActual = tmpDFActual[textCol].values
        
        # get Text for predicted
        tmpDFPred = srcDF[srcDF[colNamePredict] == x]
        textPred = tmpDFPred[textCol].values

        # add the charts
        axs[x, 0].set_title(f'Actual: {x}', fontsize=15, loc='left')
        axs[x, 0].axis('off')
        _ = axs[x, 0].imshow(getWordCloud(textActual))

        axs[x, 1].set_title(f'Cluster: {x}', fontsize=15, loc='left')
        axs[x, 1].axis('off')
        _ = axs[x, 1].imshow(getWordCloud(textPred))

    plt.tight_layout()
    plt.show()

# unsupervised results to a target column for accuracy,etc
def get_unsupervised_mapping(srcDF,
                             colNameActual,
                             colNamePredict,
                             showNumResults=5):
    # Find out how many unique values we have to deal with
    unique_act = srcDF[colNameActual].unique().tolist()
    unique_pred = srcDF[colNamePredict].unique().tolist()
    uniqueVals = list(set(unique_act + unique_pred))
    uniqueVals.sort()

    print(f'')
    print(f'Current results of comparison between clusters and actuals')
    sumDF = srcDF[[colNameActual, colNamePredict]].copy()
    sumDF = sumDF.groupby([colNameActual, colNamePredict]).size().to_frame('record_count')
    sumDF.reset_index(inplace=True)
    sumDF = sumDF.sort_values(by='record_count', ascending=False, inplace=False)
    sumDF.reset_index(drop=True, inplace=True)
    print(sumDF.head(showNumResults))
    print(f'Unique values in {colNameActual}: {len(unique_act)}')
    print(f'Unique values in {colNamePredict}: {len(unique_pred)}')
    print(f'Unique values in {colNameActual} and {colNamePredict}: {len(uniqueVals)}')
    print(f'Number of rows/combinations: {len(sumDF)}')
    print(f'')


    map_act_to_pred = dict()
    map_pred_to_act = dict()

    # Loop through table to find matching
    for ind in sumDF.index:
        ind_actual = sumDF[colNameActual][ind]
        ind_pred = sumDF[colNamePredict][ind]

        #print(f'Actual: {ind_actual}, Predicted: {ind_pred}')

        # Neither the actual nor the predicted should exist
        if ind_actual not in map_act_to_pred and ind_pred not in map_pred_to_act:
            #print(f'ind_actual and ind_pred do not exist. Adding to dict')
            map_act_to_pred[ind_actual] = ind_pred
            map_pred_to_act[ind_pred] = ind_actual

    #print(f'map_act_to_pred: {map_act_to_pred}')
    #print(f'map_pred_to_act: {map_pred_to_act}')

    # Find out if any mappings have not been made

    # Remove any unique_act that appear in map_act_to_pred
    #print(f'unique_act: {unique_act}')
    for x in map_act_to_pred:
        #print(f'Removing {x} from unique_act')
        unique_act.remove(x)

    #print(f'unique_act: {unique_act}')
    for y in map_pred_to_act:
        #print(f'Removing {y} from unique_pred')
        unique_pred.remove(y)

    #print(f'length of unique_act: {len(unique_act)}')
    #print(f'length of unique_pred: {len(unique_pred)}')

    if len(unique_act)==0 and len(unique_pred)==0:
        # everything accounted for. all good
        print(f'All mappings appear in predictions. All good')
    elif len(unique_act)==1 and len(unique_pred)==1:
        # only one mapping not completed. Add
        print(f'One mapping not found in predictions. Adding to map')
        map_act_to_pred[unique_act[0]] = unique_pred[0]
        map_pred_to_act[unique_pred[0]] = unique_act[0]
    else:
        print('Multiple mappings missing. Unable to automap.')
        print(f'Actuals without mapping: {unique_act}')
        print(f'Predictions without mapping: {unique_pred}')
        return None


    return map_pred_to_act






