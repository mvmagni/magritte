import shap
import matplotlib.pyplot as plt

# SHAP summary plot
def show_shap_summary(experiment_name,
                      shap_values,
                      xData,
                      plot_type='bar'):
    
    plt.clf()

    plt.title(f'Model: {experiment_name}')
    shap.summary_plot(shap_values,
                      xData,
                      plot_type=plot_type)  
    plt.show()
    plt.clf()

def show_shap_bar(experiment_name,
                  shap_values):

    plt.clf()
    plt.title(f'Model: {experiment_name}')
    shap.plots.bar(shap_values)
    plt.show()
    plt.clf()

def show_shap_beeswarm(experiment_name,
                       shap_values):
    plt.clf()
    plt.title(f'SHAP beeswarm: {experiment_name}')
    shap.plots.beeswarm(shap_values)
    plt.show()
    plt.clf()

def show_shap_waterfall(experiment_name,
                        shap_values,
                        value_index=0):
    plt.clf()
    plt.title(f'SHAP Waterfall [index:{value_index}]: {experiment_name}')
    shap.plots.waterfall(shap_values[value_index])
    plt.show()
    plt.clf()

def calc_shap_value(experiment_name,
                    model,
                    xData,
                    GPU=False,
                    debug=False):
    print (f'Calculating shap_values for {experiment_name}')
    if GPU:
        #explainer = shap.explainers.GPUTree(modelStore.model, xData)
        #shap_values = explainer(xData)
        #explainer = shap.Explainer(modelStore.model, xData)
        #shap_values = explainer(xData)
        print(f'STOP: Do not use GPU=True yet')
    else:
        if debug:
            print(f'DEBUG: non-gpu path')
        explainer = shap.Explainer(model)
        shap_values = explainer(xData)

    if debug:
        print(f'DEBUG: shap_value type: {type(shap_values)}')
        print(f'DEBUG: explainer type: {type(explainer)}')
        print(f'DEBUG: modelStore.model:')

    return shap_values
        
