from DataExperiment import DataExperiment
import DataExperimentSupport as des
import pickle

# Note to self: self, you need to add alot more comments
class ExperimentManager:

    def __init__(self,
                 project_name,
                 experiment_name,
                 classifier,
                 data_package):

        self.isDataPackageLoaded = False
        self.data_package = None

        self.project_name = project_name
        self.experiments = []

        self.add_data_package(data_package=data_package)

        self.add_experiment(experiment_name=experiment_name,
                            classifier=classifier)

    def add_data_package(self,
                         data_package):
        self.data_package = data_package
        self.isDataPackageLoaded = True
        self.data_package.display()

    def add_experiment(self,
                       experiment_name,
                       classifier):

        de = DataExperiment(projectName=self.project_name,
                            experimentName=experiment_name,
                            dataPackage=self.data_package,
                            classifier=classifier)
        self.experiments.append(de)

    def list_experiments(self):
        print(f'idx Processed Experiment name')
        for count, exp in enumerate(self.experiments):
            idx = '{0: >3}'.format(count)
            status = '{0: >9}'.format(str(exp.isProcessed))
            print(f'{idx} {status} {exp.experimentName}')

    # Remove model from list using model index from "list_models"
    def remove_experiment(self, experiment_index):
        if experiment_index < len(self.experiments):
            del (self.experiments[experiment_index])
            print(f'Experiment with index {experiment_index} removed')
            print(f'New experiment list:')
            self.list_experiments()
        else:
            print(f'Experiment index does not exist.')
            print(f'Experiment list:')
            self.list_models()

    def run_experiment(self,
                       axis_labels,
                       n_jobs=-1,
                       index=None):
        openProc = f'-------------------------------------------------------'
        closeProc = f'======================================================='
        if self.data_package.isProcessed is False:
            print(f'Data package has not been processed. Processing now.')
            print(openProc)
            self.data_package.processDataPackage()
            print(closeProc)
            print('')
        if index is None:
            # process all experiments
            for idx, exp in enumerate(self.experiments):
                print(f'')
                print(openProc)
                print(f'Processing experiment: [{idx}] {exp.experimentName}')
                exp.process(axis_labels=axis_labels,
                            n_jobs=n_jobs)
                print(closeProc)
                print(f'')
        else:
            print(f'Processing experiment: {self.experiements[index].experimentName}')
            self.experiments[index].process(axis_labels=axis_labels,
                                            n_jobs=n_jobs)

        print(f'')
        print(f'Processing experiments complete.')
        self.show_model_comparison()

    def display_experiment_summary(self,
                                   axisLabels,
                                   index):
        print(f'Displaying summary for experiment: {self.experiments[index].experimentName}')
        self.experiments[index].showModelReport(axisLabels=axisLabels)

    def display(self):
        self.data_package.display()
        for exp in self.experiments:
            exp.display()

    def show_model_comparison(self):
        results = None
        id_var = None
        value_vars = None
        # get info from each experiment
        for idx, exp in enumerate(self.experiments):
            #print(f'Collecting info from experiment: [{idx}] {exp.experimentName}')
            #exp_label = f'Experiment {idx}'
            exp_label = exp.experimentName
            if results is None:
                id_var, value_vars, results = exp.getModelStats_Frame(exp_label=exp_label)
            else:
                id_var, value_vars, temp_results = exp.getModelStats_Frame(exp_label=exp_label)
                results = results.append(temp_results, ignore_index=True)

        des.show_model_summary(data_frame=results,
                               id_var=id_var,
                               value_vars=value_vars)

        return id_var, value_vars, results

    def show_shap_waterfall(self,
                            model_index,
                            value_index=0):
        self.experiments[model_index].show_shap_waterfall(value_index)

    def show_shap_summary(self,
                          model_index):

        self.experiments[model_index].show_shap_summary()

    def save(self,
             filename):
        if filename is None:
            print(f'No filename provided. Please provide full path')

        with open(filename, 'wb') as f:
            print(f'Saving file as {filename}')
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)



