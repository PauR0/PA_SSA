
import os
import numpy as np
import pyvista as pv


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

import vascular_encoding_framework as vef
from vef_scripts.vef_cohort import load_cohort_object

from configuration import (cohort_dir,
                           exclude,
                           p_data)



def main(suffix='_aligned', n_components=.99):

    print("Loading Cohort Encodings")
    cohort = load_cohort_object(cohort_dir, which='encoding', exclude=exclude, keys_from_dirs=os.path.basename, suffix=suffix)
    print("......Done!")

    #Prepare data set in correct format
    X = np.array([cohort[i].to_feature_vector(mode='full', add_metadata=False) for i in cohort])
    pca = PCA(n_components=n_components, whiten=True).fit(X=X)
    print(f"To cover {n_components} of variance {pca.n_components_} are required")
    X_tr = pca.transform(X=X)


    #Prepare response variable in the correct format
    p = p_data.loc[list(cohort.keys())].to_numpy()

    #Linear Regression
    #regr = LinearRegression()
    #regr = SVR()
    regr = Lasso()
    regr.fit(X_tr, y=p)
    p_pred = regr.predict(X_tr)

    mae = mean_absolute_error(p, p_pred)
    mse = mean_squared_error(p, p_pred)
    r2 = r2_score(p, p_pred)
    print(f"Regression ressults: \n\t MAE: {mae}\n\t MSE: {mse}\n\t r2 : {r2}")



    fv_metadata = next(iter(cohort.values())).get_metadata()
    mean = vef.VascularEncoding.from_feature_vector(pca.mean_, md=fv_metadata)
    mean.to_multiblock().plot()

    print(regr.coef_, regr.intercept_)
    return
    mode_0 = vef.VascularEncoding.from_feature_vector(pca.mean_+2*np.sqrt(pca.explained_variance_[0])*pca.components_[0],  md=fv_metadata)
    mode_0_ = vef.VascularEncoding.from_feature_vector(pca.mean_-2*np.sqrt(pca.explained_variance_[0])*pca.components_[0], md=fv_metadata)

    p=pv.Plotter()
    p.add_mesh(mean.to_multiblock(), color='w', opacity=0.5)
    p.add_mesh(mode_0.to_multiblock(), color='r', opacity=0.5)
    p.add_mesh(mode_0_.to_multiblock(), color='b', opacity=0.5)
    p.show()


if __name__ == '__main__':
    main()