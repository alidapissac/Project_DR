from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.utils import Bunch

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from skimage.io import imread
from skimage.transform import resize


#Load images in structured directory like it's sklearn sample dataset
def load_image_files(container_path, dimension=(250, 250, 3)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data, target=target, target_names=categories, images=images, DESCR=descr)


image_dataset = load_image_files("dataset_identify/")

#Split data
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.15,random_state=109)

X = X_train
y = y_train


clf1 = SVC(kernel='linear', C=1)
clf2 = RandomForestClassifier(random_state=1)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=lr)



label = ['SVM', 'Random Forest', 'Stacking Classifier']
clf_list = [clf1, clf2, sclf]
    

grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
for clf, label, grd in zip(clf_list, label, grid):
        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())
        
    clf.fit(X, y)		

#plot classifier accuracy    
(_, caps, _) = plt.errorbar(range(3), clf_cv_mean, yerr=clf_cv_std, c='blue', fmt='-o', capsize=5)
for cap in caps:
    cap.set_markeredgewidth(1)                                                                                                                                
plt.xticks(range(3), ['SVM', 'RF', 'Stacking'])        
plt.ylabel('Accuracy')
plt.xlabel('Classifier')
plt.title('Stacking Ensemble')
plt.show()

import pickle
pickle.dump(sclf, open('stack_model_identify.sav', 'wb'))

pred = sclf.predict(X_test)
print(pred)

scores = sclf.score(X_test, y_test)  
print("Test score: {0:.2f} %".format(100 * scores)) 
