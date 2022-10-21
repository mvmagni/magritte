import DataPackage as dp
import DataExperimentSupport as des
import ShapSupport as sSupp
import copy
import pickle
import pandas as pd
from performance_utils import PerformanceStore
from keras import backend as K 

# Note to self: self, you need to add alot more comments, this has grown
# too big, too fast.
# Note to self: self, you need to change this to a dataclass
# Note to self: desperately needs a refactor!!!!
class DataExperiment:

    def __init__(self,
                 projectName,
                 experimentName,
                 experiment_method,
                 untrained_model,
                 dataPackage):
        self.projectName = projectName
        #print(f'Settings experimentName to: {experimentName}')
        self.experimentName = experimentName
        self.experiment_method = experiment_method # [supervised | unsupervised]
        self.has_been_automapped = False # If unsupervised and Target available predictions can be mapped

        self.__setDataPackage(dataPackage=dataPackage)
        self.__setUntrainedModel(untrained_model)

        # Really need to re-architect this. Grown too big for current architecture
        # Following are default values on init for stuff set later
        self.isModelLoaded = False
        self.Model = None

        self.isModelPredicted = False
        self.modelPrediction = None
        self.modelPredictProba = None
        self.unmappedPrediction = None # stores model prediction if unsupervised with target and mapped
        self.modelAccuracy = None
        self.modelAccuracyBal = None
        self.modelPrecisionMicro = None
        self.modelRecallMicro = None
        self.modelF1Micro = None
        self.modelF1Weighted = None
        self.modelCohenKappa = None
        self.modelAUC = None
        
        self.isLearningCurveCreated = False
        self.model_train_sizes = None
        self.model_train_scores = None
        self.model_test_scores = None
        self.model_fit_times = None
        
        self.isROCAUCCalculated = False
        self.modelROCAUC = None

        self.isProcessed = False

        self.hasSHAPValues = False
        self.shap_values = None

        # parameters for tensor model training
        self.tensor_parms = None
        self.model_history = None # Model history for after training completed
        
        # ===============================================
        self.check_input()
        self.display()

    def check_input(self):
        # Check that methods are valid
        valid_methods = ['supervised', 'unsupervised']
        isOk = False
        e_message = ''

        for x in valid_methods:
            if x == self.experiment_method:
                isOk = True

            if not isOk:
                e_message = f'Invalid method submitted.'

        try:
            assert (isOk is True)
        except AssertionError as e:
            e.args += (e_message, f'Use: {", ".join(valid_methods)}')
            raise

    def isKerasModel(self):
        strDes = str(type(self.untrained_model))
        print(strDes)
        if strDes.find('keras.engine') == -1:
            return False
        else:
            return True

    def add_tensor_parms(self, tensor_parms):
        self.tensor_parms = tensor_parms

    def display(self):
        indent = '---> '
        print(f'')
        print(f'DataExperiment summary:')
        print(f'{indent}projectName: {self.projectName}')
        print(f'{indent}experimentName: {self.experimentName}')
        print(f'{indent}experimentMethod: {self.experiment_method}')
        if self.is_unsupervised_with_target():
            print(f'{indent}Unsupervised experiment with target column {self.dataPackage.targetColumn}')
            print(f'{indent}{indent}prediction has been mapped: {self.has_been_automapped}')
        print(f'{indent}isDataPackageLoaded: {self.isDataPackageLoaded}')

        print(f'{indent}isProcessed: {self.isProcessed}')
        print(f'{indent}isModelLoaded: {self.isModelLoaded}')
        print(f'{indent}isModelPredicted: {self.isModelPredicted}')
        print(f'{indent}isLearningCurveCreated: {self.isLearningCurveCreated}')

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
        monitor = PerformanceStore()
        print(f'Training model for {self.experimentName}. ', end='')
        if self.isKerasModel():
            print(f'Keras model detected. compiling and training')
            model, history = des.createTensorModel(data=self.dataPackage.getTrainData(),
                                                   uniqueColumn=self.dataPackage.uniqueColumn,
                                                   targetColumn=self.dataPackage.targetColumn,
                                                   untrained_model=self.getUntrainedModel(),
                                                   tensor_parms=self.tensor_parms)
        
            # Save the training history
            self.model_history = history
            
        else: 
        
            model = des.createModel(data=self.dataPackage.getTrainData(),
                                    uniqueColumn=self.dataPackage.uniqueColumn,
                                    targetColumn=self.dataPackage.targetColumn,
                                    untrained_model=self.getUntrainedModel(),
                                    experiment_method=self.experiment_method)

        self.__setModel(model)

        print(f'Completed. {monitor.end_timer()}')
        self.predictModel()
        K.clear_session()


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
                             predict_proba=None,
                             sigDigs=2):
        self.modelPrediction = predictionData
        self.isModelPredicted = True
        self.modelPredictionColActual = colActual
        self.modelPredictionColPredict = colPredict
        if predict_proba is None:
            self.modelPredictProba = 0
        else: 
            self.modelPredictProba = predict_proba


        self.modelAccuracy = round(des.getModelAccuracy(data=predictionData,
                                                        colActual=colActual,
                                                        colPredict=colPredict), sigDigs)
        
        self.modelAccuracyBal = round(des.getModelAccuracyBalanced(data=predictionData,
                                                                   colActual=colActual,
                                                                   colPredict=colPredict), sigDigs)
        
        self.modelPrecisionMicro = round(des.getModelPrecision(data=predictionData,
                                                          colActual=colActual,
                                                          colPredict=colPredict,
                                                          average='micro'), sigDigs)

        self.modelRecallMicro = round(des.getModelRecall(data=predictionData,
                                                    colActual=colActual,
                                                    colPredict=colPredict,
                                                    average='micro'), sigDigs)

        self.modelF1Micro = round(des.getModelF1(data=predictionData,
                                            colActual=colActual,
                                            colPredict=colPredict,
                                            average='micro'), sigDigs)

        self.modelF1Weighted = round(des.getModelF1(data=predictionData,
                                            colActual=colActual,
                                            colPredict=colPredict,
                                            average='weighted'), sigDigs)

        self.modelCohenKappa = round(des.getModelCohenKappa(data=predictionData,
                                                            colActual=colActual,
                                                            colPredict=colPredict), sigDigs)
        
        if predict_proba is None:
            self.modelAUC = 0
        else:
            self.modelAUC = round(des.getAUC(data=predictionData,
                                             colActual=colActual,
                                             colPredict=colPredict,
                                             average='macro'), sigDigs)

    def showModelStats(self):
        print(f'')
        print(f'Model Stats:')

        id_var, value_vars, df = self.getModelStats_Frame(exp_label=self.experimentName)

        des.show_model_summary(data_frame=df,
                               id_var=id_var,
                               value_vars=value_vars,
                               individual=True,
                               title=f'Model summary for {self.experimentName}')
        print(f'')

    def getModelStats_Frame(self,
                            exp_label):
        # list of strings
        id_var = 'Experiment'
        id_vars = [id_var]
        value_vars = ['Acc.',
                      'Acc. (bal)',
                      'Prec (micro)',
                      'Recall (micro)',
                      'F1 (micro)',
                      'F1 (weight)',
                      'Cohen Kappa']
        
        # Removed AUC as it was not calculating for all
        # Add back later
        #              'AUC (macro)']

        col_names = []
        col_names.extend(id_vars)
        col_names.extend(value_vars)

        # list of int
        vals = [exp_label,
                self.modelAccuracy,
                self.modelAccuracyBal,
                self.modelPrecisionMicro,
                self.modelRecallMicro,
                self.modelF1Micro,
                self.modelF1Weighted,
                self.modelCohenKappa]
        
        # Removed AUC as it was not calculating for all
        # Add back later
        #    self.modelAUC]

        # both lists, with columns specified
        df = pd.DataFrame([vals], columns=col_names)
        return id_var, value_vars, df.copy()

    def getModelPrediction(self):
        if self.isModelPredicted:
            return self.modelPrediction
        else:
            print(f'No model predictions calculated.')
            return None

    def predictModel(self):
        monitor = PerformanceStore()
        print(f'Predicting model for {self.experimentName}. ', end='')
        if self.isModelPredicted:
            print("Model already predicted. Displaying results:")
            self.showModelStats()
            return

        tDf, colActual, colPredict = des.predictModel(model=self.getModel(),
                                                      data=self.dataPackage.getTestData(),
                                                      uniqueColumn=self.dataPackage.uniqueColumn,
                                                      targetColumn=self.dataPackage.targetColumn,
                                                      use_argmax=self.isKerasModel()) # if kerasModel=True use argmax


        if self.experiment_method == 'supervised' and not self.isKerasModel():
            print(f'Supervised model being processed. Calling predictProba')
            predict_proba = des.predictProba(model=self.getModel(),
                                            data=self.dataPackage.getTestData(),
                                            uniqueColumn=self.dataPackage.uniqueColumn,
                                            targetColumn=self.dataPackage.targetColumn
                                            )

            self.modelPredictProba = predict_proba
        else: 
            print(f'Unsupervised model being processed. no predict proba')

        self.__setModelPrediction(predictionData=tDf,
                                  colActual=colActual,
                                  colPredict=colPredict
                                  )

        print(f'Completed. {monitor.end_timer()}')
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

        # des.showAllModelFeatureImportance(data=df,
        #                                   featureLabel=featureLabel,
        #                                   valueLabel=valueLabel
        #                                   )

        des.showFeatureImportance(model=self.getModel(),
                                  XTrain=self.dataPackage.getXTrainData(),
                                  YTrain=self.dataPackage.getYTrainData(),
                                  topn=topn,
                                  useLasso=useLasso)

    def process(self):
        self.createModel()
        self.isProcessed = True

    def showFullModelReport(self,
                           startValue=0.0001,
                           increment=0.0001,
                           upperValue=0.01,
                           useLasso=False,
                           topn=5):

        axis_labels = list(self.dataPackage.labelEncoder.classes_) 
        self.showModelStats()

        des.showReport(data=self.getModelPrediction(),
                       colNameActual=self.modelPredictionColActual,
                       colNamePredict=self.modelPredictionColPredict,
                       axis_labels=axis_labels,
                       titleSuffix=self.experimentName)

        self.showPrecisionRecallCurve()
        self.showModelLearningCurve()
        self.showModelROCAUC(axis_labels=axis_labels)
        self.showModelFeatureImportance(startValue=startValue,
                                        increment=increment,
                                        upperValue=upperValue,
                                        useLasso=useLasso,
                                        topn=topn)
        self.showLimeGlobalImportance()
        self.showLimeLocalImportance()

    def showModelReport(self,
                        axis_labels):
        des.showReport(data=self.getModelPrediction(),
                       colNameActual=self.modelPredictionColActual,
                       colNamePredict=self.modelPredictionColPredict,
                       axis_labels=axis_labels,
                       titleSuffix=self.experimentName)

    def showModelROCAUC(self, axis_labels, useStored=False):
        if useStored and self.isModelROCAUCCalculated:
            print('Model ROCAUC already calculated. Displaying stored results')
            tViz = self.__getModelROCAUC()
            tViz.show()
        else:
            print('Model ROCAUC not calculated. Calculating now')
            viz = des.showROCAUC(dataTrain=self.dataPackage.getTrainData(),
                                 dataTest=self.dataPackage.getTestData(),
                                 classifier=self.getUntrainedModel(),
                                 axis_labels=axis_labels,
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
        des.showPrecisionRecallCurve(model=self.getUntrainedModel(),
                                     XTrain=self.dataPackage.getXTrainData(),
                                     YTrain=self.dataPackage.getYTrainData(),
                                     XTest=self.dataPackage.getXTestData(),
                                     YTest=self.dataPackage.getYTestData()
                                     )

    # Do features include the target and unique? Don't think so but can't recall
    def getFeatures(self):
        return self.dataPackage.dataFeatures

    def createModelLearningCurve(self,
                                 cv=None,
                                 n_jobs=None,
                                 train_sizes=None,
                                 verbose=0):
        # If it is already predicted just show it
        if self.isLearningCurveCreated:
            print('Model learning curve already calculated. Displaying results:')
            # self.showLearningCurve()
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
        self.isLearningCurveCreated = True
        self.model_train_sizes = train_sizes
        self.model_train_scores = train_scores
        self.model_test_scores = test_scores
        self.model_fit_times = fit_times

    def showModelLearningCurve(self,
                               axes=None,
                               ylim=(0.0, 1.01)
                               ):
        if not self.isLearningCurveCreated:
            print('Model Learning curve has not yet been calculated. Calculating now')
            self.createModelLearningCurve()

        des.plot_learning_curve(train_sizes=self.model_train_sizes,
                                train_scores=self.model_train_scores,
                                test_scores=self.model_test_scores,
                                fit_times=self.model_fit_times,
                                title=self.experimentName,
                                axes=axes,
                                ylim=ylim
                                )

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

    # Cluster can be auto mapped to actual target column
    def is_unsupervised_with_target(self):
        if self.experiment_method == 'unsupervised' and self.dataPackage.targetColumn is not None:
            return True
        else:
            return False

    def __get_mapping_data(self):

        # Get clean data
        #cleanDF = self.dataPackage.getCleanData()
        cleanDF = self.dataPackage.getOrigData()

        # Get the uuid applicable for the test data
        testDF = self.dataPackage.getTestData()
        testDF = testDF[[self.dataPackage.uniqueColumn]].copy()

        # Get prediction info
        predDF = self.getModelPrediction()


        testDF.reset_index(drop=True, inplace=True)
        predDF.reset_index(drop=True, inplace=True)

        # Assemble frame for reporting
        # concat the predictions with the test data (subset of full data)
        result = pd.concat([testDF, predDF], axis=1)
        result = pd.merge(result, cleanDF, on=self.dataPackage.uniqueColumn, how='inner')

        return result


    def show_cluster_mapping_cloud(self):
        
        # Requirements:
        # Unsupervised with target column present
        if self.is_unsupervised_with_target():
            
            srcDF = self.__get_mapping_data()
            

            # pass frame to function for processing
            des.show_cluster_mapping_cloud(srcDF=self.__get_mapping_data(),
                                           textCol=self.dataPackage.dataColumn,
                                           colNameActual=self.modelPredictionColActual,
                                           colNamePredict=self.modelPredictionColPredict)
        else:
            print('Experiment needs to be unsupervised with a target column')
            print(f'Experiment name: {self.experimentName}')
            print(f'Experiment method: {self.experiment_method}')
            print(f'Target column: {self.dataPackage.targetColumn}')

    def get_unsupervised_mapping(self,
                                 showNumResults=5,
                                 auto_mapping=False):
        # Requirements:
        # Unsupervised with target column present
        if self.is_unsupervised_with_target():


            map_pred_to_actual = des.get_unsupervised_mapping(srcDF=self.__get_mapping_data(),
                                                              colNameActual=self.modelPredictionColActual,
                                                              colNamePredict=self.modelPredictionColPredict,
                                                              showNumResults=showNumResults)
            # Do we need to automap the predictions
            if auto_mapping and map_pred_to_actual is not None:
                print(f'Auto mapping the results')
                print(f'Mapping predicted to actual: {map_pred_to_actual}')
                self.set_prediction_mapping(map_pred_to_actual=map_pred_to_actual)
            else:
                print(f'Display automapped prediction to actual results (no data changed)')
                print(map_pred_to_actual)

            return map_pred_to_actual

        else:
            print('Experiment needs to be unsupervised with a target column')
            print(f'Experiment name: {self.experimentName}')
            print(f'Experiment method: {self.experiment_method}')
            print(f'Target column: {self.dataPackage.targetColumn}')


    def set_prediction_mapping(self,
                               map_pred_to_actual):
        # map_pred_to_act: dict with prediction to actual mappings

        # get prediction
        current_prediction = self.getModelPrediction().copy()

        # copy predict column
        orig_predict_col = f'{self.modelPredictionColPredict}_orig'
        current_prediction[orig_predict_col] = current_prediction[self.modelPredictionColPredict]

        # update predict column with mapping for all values
        for key in map_pred_to_actual:
            current_prediction.loc[current_prediction[orig_predict_col]==key,
                                   self.modelPredictionColPredict] = map_pred_to_actual[key]

        current_prediction = current_prediction.drop(columns=[orig_predict_col])

        # Save old modelPrediction
        self.unmappedPrediction = self.getModelPrediction().copy()

        # Set prediction to mapped prediction and set to automapped
        self.has_been_automapped = True
        self.__setModelPrediction(predictionData=current_prediction,
                                  colActual=self.modelPredictionColActual,
                                  colPredict=self.modelPredictionColPredict)