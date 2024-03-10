import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

db = pd.read_csv("diabetes_prediction_dataset.csv")
db = db.drop_duplicates()

def binData(category):
    """
    Seperates the category by 1 = Having or 0 = Not Having.
    Plots them into a piechart
    """    
    hasCount = 0
    notCount = 0
    for bin in db[category]:
        if bin == 1:
            hasCount += 1
        elif bin == 0:
            notCount += 1
    total = hasCount + notCount
    size = np.array([hasCount/total, notCount/total])
    labels = ['HAS', 'Does NOT have']
    myexplode = [0.2, 0]

    plt.title('Distribution of ' + category + ' out of ' + str(total) + ' people')
    plt.pie(size,labels=labels, explode = myexplode, shadow = True, autopct='%1.1f%%')
    plt.show()

def smokingHist():
    """
    Seperates the smoking history by 3 categories then plots them into a piechart.
    """    
    noHist = 0
    hasHist = 0
    noInfo = 0
    for result in db['smoking_history']:
        if result == 'never':
            noHist += 1
        elif result == 'current' or result == 'former' or result == 'ever' or result =='not current':
            hasHist += 1
        elif result == 'No Info':
            noInfo += 1
    total = noInfo + noHist + hasHist
    size = np.array([hasHist/total, noHist/total, noInfo/total])
    labels = ['History of Smoking', 'No History', 'No Info']

    plt.title('Distribution of General Smoking History with total: ' + str(total))
    plt.pie(size,labels=labels, explode = [0.2, 0, 0], shadow = True, autopct='%1.1f%%')
    plt.show()

def pdfPlot(category):
    """
    Plots the probability density function of the category.
    """    
    sns.histplot(db[category], kde=True, stat="density", bins=30)
    plt.title('Probability Density of ' + str(category))
    plt.xlabel(str(category))
    plt.ylabel('Density')
    plt.show()


def hist_hasORnot_diab(category):
    """
    Plots a histogram of the input category with diabetes and without diabetes.
    """    
    plt.figure(figsize=(12, 6))

    sns.histplot(db[db['diabetes'] == 0][str(category)], color="blue", label='No Diabetes', kde=False, bins=20)
    sns.histplot(db[db['diabetes'] == 1][str(category)], color="red", label='Has Diabetes', kde=False, bins=20)

    plt.title('Histogram of ' + str(category) + ' for Individuals with and without Diabetes')
    plt.xlabel(str(category))
    plt.ylabel('Amount from Dataset')
    plt.legend()
    plt.show()


def cdcBMI():
    """
    Plots the BMI seperated by CDC categorization.
    """    
    under = 0
    healthy = 0
    over = 0
    obese = 0
    for bmi in db['bmi']:
        if bmi < 18.5:
            under += 1
        elif bmi >= 18.5 and bmi < 25:
            healthy += 1
        elif bmi >= 25 and bmi < 30:
            over += 1
        elif bmi >= 30:
            obese += 1

    size = np.array([under, healthy, over, obese])
    labels = [
        'Underweight\n(<18.5)',
        'Healthy Weight\n(18.5 - <25)',
        'Overweight\n(25 - <30)',
        'Obesity\n(≥30)'
    ]
    plt.bar(labels, size)
    plt.title('BMI Seperated by CDC Interperation')
    plt.xlabel('Weight Status')
    plt.ylabel('Amount in Dataset Per')
    plt.show()

def cdcH1():
    """
    Plots the HbA1c_level seperated by CDC categorization.
    """    
    h1diab = 0
    h1pre = 0
    h1norm = 0

    for h1 in db['HbA1c_level']:
        if h1 < 5.7:
            h1norm += 1
        elif h1 >= 5.7 and h1 < 6.5:
            h1pre += 1
        elif h1 >= 6.5:
            h1diab += 1

    labelsh1 = [
        'Normal\n(<5.7)',
        'Prediabetes\n(5.7 - <6.5)',
        'Diabetes\n(≥6.5)'
    ]
    h1size = np.array([h1norm, h1pre, h1diab])

    plt.bar(labelsh1, h1size)
    plt.title('HbA1c Level Seperated by CDC Interperation')
    plt.xlabel('Result Status')
    plt.ylabel('Amount in Dataset Per')
    plt.show()

def cdcGluc():
    """
    Plots the blood_glucose_level seperated by CDC categorization.
    """   
    glucD = 0
    glucP = 0
    glucN = 0

    for gluc in db['blood_glucose_level']:
        if gluc < 140:
            glucN += 1
        elif gluc >= 140 and gluc < 200:
            glucP += 1
        elif gluc >= 200:
            glucD += 1


    labelsgl = [
        'Normal\n(<140)',
        'Prediabetes\n(140 - <200)',
        'Diabetes\n(≥200)'
    ]
    glsize = np.array([glucN, glucP, glucD])


    plt.bar(labelsgl, glsize)
    plt.title('Blood Glucose Level Seperated by CDC Interperation')
    plt.xlabel('Result Status')
    plt.ylabel('Amount in Dataset Per')
    plt.show()

def genderSplit():
    """
    Seperates the gender and plots it on a piechart.
    """   
    female = 0
    male = 0
    for gen in db['gender']:
        if gen == "Female":
            female += 1
        elif gen == 'Male':
            male += 1
    total = female + male
    size = np.array([female/total, male/total])
    labels = ['Female', 'Male']

    plt.title('Distribution of Gender out of ' + str(total) + ' samples')
    plt.pie(size,labels=labels, shadow = True, autopct='%1.1f%%')
    plt.show()


def genderSplitbyDiab(gender):
    """
    Splits the gender into a piechart with and without Diabetes
    """   
    diabetes_count = len(db[(db['gender'] == str(gender)) & (db['diabetes'] == 1)])
    no_diabetes = len(db[(db['gender'] == str(gender)) & (db['diabetes'] == 0)])
    total = diabetes_count + no_diabetes
    labels = ['No Diabetes', 'Has Diabetes']

    plt.figure(figsize=(12, 6))
    sizes = [no_diabetes / total, diabetes_count / total]
    plt.title(str(gender) + ': With and Without Diabetes')
    plt.pie(sizes, labels=labels, explode = [0, 0.2], shadow=True, autopct='%1.1f%%')

