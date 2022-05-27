import lime
from lime import lime_tabular
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def showLimeGlobalImportance(XTrain,
                             YTrain):

    lr = LogisticRegression()
    scaler = MinMaxScaler()
    XTrain_scale = scaler.fit_transform(XTrain)
    
    lr.fit(XTrain_scale, YTrain)

    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(10, 15))
        plt.barh(range(len(lr.coef_[0])), lr.coef_[0], color=["red" if coef < 0 else "green" for coef in lr.coef_[0]])
        plt.yticks(range(len(lr.coef_[0])), XTrain.columns.values.tolist());
        plt.title("Global Importance: Weights")

    plt.show()
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

def showLimeLocalImportance(XTrain,
                            YTrain,
                            XTest,
                            YTest,
                            mode):


    #XTrain = XTrain.to_numpy()
    #YTrain = YTrain.to_numpy()
    XTest = XTest.to_numpy()
    YTest = YTest.to_numpy()

    explainer = getLimeExplainer(XTrain=XTrain,
                                 features=XTrain.columns.values.tolist(),
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
