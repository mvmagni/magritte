import DataPackage as dp
import DataExperimentSupport as des
import ShapSupport as sSupp
import copy
import pickle
import pandas as pd


# Note to self: self, you need to add alot more comments
class DataExperiment:

    def __init__(self,
                 projectName,
                 experimentName,
                 untrained_model,
                 dataPackage):
        self.projectName = projectName
        self.experimentName = experimentName
        self.__setDataPackage(dataPackage=dataPackage)
        self.__setUntrainedModel(untrained_model)

        # Should really consider putting these into a function
        # Following are default values on init for stuff set later
        self.isModelLoaded = False
        self.Model = None

        self.isModelPredicted = False
        self.modelPrediction = None
        self.modelAccuracy = None
        self.modelPrecision = None
        self.modelRecall = None
        self.modelF1 = None
        self.modelCohenKappa = None

        self.isLearningCurveCreated = False
        self.model_train_sizes = None
        self.model_train_scores = None
        self.model_test_scores = None
        self.model_fit_times = None
        self.modelROCAUC = None

        self.isROCAUCCalculated = False
        self.modelROCAUC = None

        self.isProcessed = False

        self.hasSHAPValues = False
        self.shap_values = None

        # ===============================================

        self.display()

    def display(self):
        indent = '---> '
        print(f'DataExperiment summary:')
        print(f'{indent}projectName: {self.projectName}')
        print(f'{indent}experimentName: {self.experimentName}')
        print(f'{indent}isDataPackageLoaded: {self.isDataPackageLoaded}')

        print(f'{indent}isProcessed: {self.isProcessed}')
        print(f'{indent}isModelLoaded: {self.isModelLoaded}')
        print(f'{indent}isModelPredicted: {self.isModelPredicted}')
        print(f'{indent}isModelLearningCurveCreated: {self.isLearningCurveCreated}')

        print(f'{indent}isUntrainedModelLoaded: {self.isUntrainedModelLoaded}')
        print(self.getUntrainedModel())
        print('')

    def getUntrainedModel(self):
        return copy.deepcopy(self.untrained_model)

    def __setUntrainedModel(self, untrained_model):
        self.untrained_model = untrained_model
        self.isUntrainedModelLoaded = True

    def __setDataPackage(self,
                         dataPackage):

        self.dataPackage = dataPackage
        self.isDataPackageLoaded = True

    def createModel(self):
        print(f'Training model for {self.experimentName}')
        model = des.createModel(data=self.dataPackage.getTrainData(),
                                uniqueColumn=self.dataPackage.uniqueColumn,
                                targetColumn=self.dataPackage.targetColumn,
                                untrained_model=self.getUntrainedModel())

        self.__setModel(model)
        self.predictModel()

    def __setModel(self, model):
        self.model = model
        self.isModelLoaded = True

        # when you set model invalidate some things
        self.isModelPredicted = False
        self.modelPrediction = None

    def getModel(self):
        return self.model

    def __setModelPrediction(self,
                             predictionData,
                             colActual,
                             colPredict,
                             average='weighted',
                             sigDigs=2):
        self.modelPrediction = predictionData
        self.isModelPredicted = True
        self.modelPredictionColActual = colActual
        self.modelPredictionColPredict = colPredict

        self.modelAccuracy = round(des.getModelAccuracy(data=predictionData,
                                                        colActual=colActual,
                                                        colPredict=colPredict), sigDigs)

        self.modelPrecision = round(des.getModelPrecision(data=predictionData,
                                                          colActual=colActual,
                                                          colPredict=colPredict,
                                                          average=average), sigDigs)

        self.modelRecall = round(des.getModelRecall(data=predictionData,
                                                    colActual=colActual,
                                                    colPredict=colPredict,
                                                    average=average), sigDigs)

        self.modelF1 = round(des.getModelF1(data=predictionData,
                                            colActual=colActual,
                                            colPredict=colPredict,
                                            average=average), sigDigs)

        self.modelCohenKappa = round(des.getModelCohenKappa(data=predictionData,
                                                            colActual=colActual,
                                                            colPredict=colPredict), sigDigs)

    def showModelStats(self):
        print(f'Model Stats:')
        #print(f'Accuracy: {self.modelAccuracy}')
        #print(f'Precision: {self.modelPrecision}')
        #print(f'Recalll: {self.modelRecall}')
        #print(f'F1 Score: {self.modelF1}')
        #print(f'Cohen kappa:: {self.modelCohenKappa}')

        id_var, value_vars, df = self.getModelStats_Frame(exp_label=self.experimentName)

        des.show_model_summary(data_frame=df,
                               id_var=id_var,
                               value_vars=value_vars)
        print(f'')

    def getModelStats_Frame(self,
                            exp_label):
        # list of strings
        id_var = 'Experiment'
        id_vars = [id_var]
        value_vars = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cohen kappa']

        col_names = []
        col_names.extend(id_vars)
        col_names.extend(value_vars)

        # list of int
        vals = [exp_label,
                self.modelAccuracy,
                self.modelPrecision,
                self.modelRecall,
                self.modelF1,
                self.modelCohenKappa]

        # both lists, with columns specified
        df = pd.DataFrame([vals], columns=col_names)
        return id_var, value_vars, df.copy()

    def getModelPrediction(self):
        if self.isModelPredicted:
            return self.modelPrediction
        else:
            print(f'No model predictions calculated.')
            return None

    def predictModel(self, average='weighted'):
        print(f'Predicting model for {self.experimentName}')
        if self.isModelPredicted:
            display("Model already predicted. Displaying results:")
            self.showModelStats()
            return

        tDf, colActual, colPredict = des.predictModel(model=self.getModel(),
                                                      data=self.dataPackage.getTestData(),
                                                      uniqueColumn=self.dataPackage.uniqueColumn,
                                                      targetColumn=self.dataPackage.targetColumn)

        self.__setModelPrediction(predictionData=tDf,
                                  colActual=colActual,
                                  colPredict=colPredict,
                                  average=average)

        self.showModelStats()

    def analyzeModelFeatureImportance(self,
                                      returnAbove=0.002,
                                      startValue=0.0001,
                                      increment=0.0001,
                                      upperValue=0.01,
                                      showSummary=True,
                                      showPlot=True):

        df, featureLabel, valueLabel = des.getModelFeatureImportance(self.getModel())

        retDf = des.analyzeModelFeatureImportance(data=df,
                                                  valueLabel=valueLabel,
                                                  startValue=startValue,
                                                  increment=increment,
                                                  upperValue=upperValue,
                                                  returnAbove=returnAbove,
                                                  showSummary=showSummary,
                                                  showPlot=showPlot)
        return retDf

    def showModelFeatureImportance(self,
                                   startValue=0.0001,
                                   increment=0.0001,
                                   upperValue=0.01,
                                   useLasso=False,
                                   topn=5):

        df, featureLabel, valueLabel = des.getModelFeatureImportance(self.getModel())

        des.analyzeModelFeatureImportance(data=df,
                                          startValue=startValue,
                                          increment=increment,
                                          upperValue=upperValue,
                                          showSummary=False)

        des.showAllModelFeatureImportance(data=df,
                                          featureLabel=featureLabel,
                                          valueLabel=valueLabel
                                          )

        des.showFeatureImportance(model=self.getModel(),
                                  XTrain=self.dataPackage.getXTrainData(),
                                  YTrain=self.dataPackage.getYTrainData(),
                                  topn=topn,
                                  useLasso=useLasso)

    def process(self,
                axis_labels,
                n_jobs=-1):
        self.createModel()
        self.createModelLearningCurve(n_jobs=n_jobs)
        #self.showModelReport(axis_labels)
        self.isProcessed = True

    def showModelReport(self,
                        axisLabels,
                        startValue=0.0001,
                        increment=0.0001,
                        upperValue=0.01,
                        useLasso=False,
                        topn=5):
                            
        self.showModelStats()

        des.showReport(data=self.getModelPrediction(),
                       colNameActual=self.modelPredictionColActual,
                       colNamePredict=self.modelPredictionColPredict,
                       axisLabels=axisLabels,
                       titleSuffix=self.experimentName)

        self.showPrecisionRecallCurve()
        self.showModelLearningCurve()
        self.showModelROCAUC(axisLabels=axisLabels)
        self.showModelFeatureImportance(startValue=startValue,
                                            increment=increment,
                                            upperValue=upperValue,
                                            useLasso=useLasso,
                                            topn=topn)
        self.showLimeGlobalImportance()
        self.showLimeLocalImportance()

    def showModelROCAUC(self, axisLabels, useStored=False):
        if useStored and self.isModelROCAUCCalculated:
            print('Model ROCAUC already calculated. Displaying stored results')
            tViz = self.__getModelROCAUC()
            tViz.show()
        else:
            print('Model ROCAUC not calculated. Starting now')
            viz = des.showROCAUC(dataTrain=self.dataPackage.getTrainData(),
                                 dataTest=self.dataPackage.getTestData(),
                                 classifier=self.getUntrainedModel(),
                                 axisLabels=axisLabels,
                                 colNameActual=self.dataPackage.targetColumn,
                                 features=self.getFeatures())
            self.__setModelROCAUC(visualizer=viz)
            viz.show()

    def __setModelROCAUC(self,
                             visualizer):
        self.isModelROCAUCCalculated = True
        self.modelROCAUC = pickle.dumps(visualizer)

    def __getModelROCAUC(self):
        return pickle.loads(self.modelROCAUC)
        
    def showPrecisionRecallCurve(self):
        des.showPrecisionRecallCurve(model=self.getModel(),
                                     XTrain=self.dataPackage.getXTrainData(),
                                     YTrain=self.dataPackage.getYTrainData(),
                                     XTest=self.dataPackage.getXTestData(),
                                     YTest=self.dataPackage.getYTestData()
                                     )

    # Do features include the target and unique? DOn't think so but can't recall
    def getFeatures(self):
        return self.dataPackage.dataFeatures

    def createModelLearningCurve(self,
                                 cv=None,
                                 n_jobs=None,
                                 train_sizes=None,
                                 verbose=4):
        # If it is already predicted just show it
        if self.isLearningCurveCreated:
            print('Model learning curve already calculated. Displaying results:')
            #self.showLearningCurve()
        else:
            df = self.dataPackage.getTrainData()
            train_sizes, train_scores, test_scores, fit_times = des.create_learning_curve(
                estimator=self.getUntrainedModel(),
                X=df[self.dataPackage.dataFeatures],
                y=df[self.dataPackage.targetColumn],
                cv=cv,
                n_jobs=n_jobs,
                train_sizes=train_sizes,
                verbose=verbose)

            self.__setModelLearningData(train_sizes=train_sizes,
                                        train_scores=train_scores,
                                        test_scores=test_scores,
                                        fit_times=fit_times)


    def __setModelLearningData(self,
                                   train_sizes,
                                   train_scores,
                                   test_scores,
                                   fit_times):
        self.isModelLearningCurveCreated = True
        self.model_train_sizes = train_sizes
        self.model_train_scores = train_scores
        self.model_test_scores = test_scores
        self.model_fit_times = fit_times

    def showModelLearningCurve(self,
                               axes=None,
                               ylim=(0.0, 1.01)
                               ):
        if self.isModelLearningCurveCreated:

            des.plot_learning_curve(train_sizes=self.model_train_sizes,
                                    train_scores=self.model_train_scores,
                                    test_scores=self.model_test_scores,
                                    fit_times=self.model_fit_times,
                                    title=self.experimentName,
                                    axes=axes,
                                    ylim=ylim
                                    )
        else:
            display('Model Learning curve has not yet been calculated')

    def showLimeGlobalImportance(self):
        des.showLimeGlobalImportance(XTrain=self.dataPackage.getXTrainData(),
                                     YTrain=self.dataPackage.getYTrainData(),
                                     features=self.dataPackage.dataFeatures
                                     )

    def showLimeLocalImportance(self):
        des.showLimeLocalImportance(XTrain=self.dataPackage.getXTrainData(),
                                    YTrain=self.dataPackage.getYTrainData(),
                                    XTest=self.dataPackage.getXTestData(),
                                    YTest=self.dataPackage.getYTestData(),
                                    features=self.dataPackage.dataFeatures,
                                    mode='classification')

    def show_shap_summary(self):
        if not self.hasSHAPValues:
            print(f'SHAP values not calculated. Generating now')
            self.calc_shap_value()

        sSupp.show_shap_summary(experiment_name=self.experimentName,
                                shap_values=self.shap_values,
                                xData=self.dataPackage.getXTrainData())

    def show_shap_waterfall(self,
                            value_index=0):
        if not self.hasSHAPValues:
            print(f'SHAP values not calculated. Generating now')
            self.calc_shap_value()

        sSupp.show_shap_waterfall(experiment_name=self.experimentName,
                                  shap_values=self.shap_values,
                                  value_index=value_index)

    def calc_shap_value(self,
                        GPU=False,
                        debug=False):
        if self.hasSHAPValues:
            print(f'{self.experimentName} already has SHAP values. Not recalculating')
            return

        shap_values = sSupp.calc_shap_value(experiment_name=self.experimentName,
                                            model=self.getModel(),
                                            xData=self.dataPackage.getXTrainData(),
                                            GPU=GPU,
                                            debug=debug)
        self.shap_values = shap_values
        self.hasSHAPValues = True

