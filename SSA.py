
import os

import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

import vascular_encoding_framework as vef
from vef_scripts.vef_cohort import load_cohort_object

from configuration import (cohort_dir,
                           exclude,
                           p_data)


def pca_transform_standardized(X, pca):
    """
    Transform using the PCA but normalize the directions with the standard deviation.
    """
    assert len(X.shape) == 2, f"Wrong shape, expected (N, {pca.mean_.shape[0]})"
    assert X.shape[1] == pca.mean_.shape[0], f"The number of features expected is ({pca.mean_.shape[0]})"

    X_tr = X - pca.mean_
    return pca.components_.dot(X_tr.T).T / np.sqrt(pca.explained_variance_)
#

def pca_inverse_transform_standardized(X, pca, add_mean=False):
    """
    Apply the inverse transform of an standardized PCA.
    """
    assert len(X.shape) == 2, f"Wrong shape, expected (N, {pca.n_components_}) and given is {X.shape}"
    assert X.shape[1] == pca.n_components_, f"The number of features expected is ({pca.n_components_}) and given is {X.shape[1]}"

    X_inv = (X * np.sqrt(pca.explained_variance_)).dot(pca.components_)
    if add_mean:
        return pca.mean_ + X_inv

    return X_inv
#

def get_regressor(regressor_kind):
    """
    Get the appropriate regressor based on regressor_kind
    """
    regressors = {
        'linear' : LinearRegression(),
        'lasso'  : Lasso(),
    }
    return regressors[regressor_kind]
#

def make_cohort_convergence_study(X, y, regressor_kind, test_bounds=None, test_step=0.1):
    """
    Display the results of the regression results.
    """

    if test_bounds is None:
        test_bounds = (0.1, 0.5)

    results = []
    for test_size in np.linspace(test_bounds[0], test_bounds[1], round((test_bounds[1]-test_bounds[0])/test_step)):
        ress_train = {}
        ress_test  = {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True)

        regr = get_regressor(regressor_kind)
        regr.fit(X_train, y=y_train)

        y_pred = regr.predict(X_train)
        mae, mse = np.abs(y_train-y_pred), np.power(y_train-y_pred, 2)
        for ae, se in zip(mae, mse):
            results.append(
                {'MAE': ae, 'MSE': se, 'type': 'train', 'Test size': test_size}
            )
        ress_train['MAE'] = mean_absolute_error(y_train, p_pred)
        ress_train['MSE'] = mean_squared_error(y_train, p_pred)
        ress_train['r2']  = r2_score(y_train, p_pred)
        ress_train['type'] = 'train'
        ress_train['Test size'] = test_size
        results.append(ress_train)

        p_pred = regr.predict(X_test)
        ress_test['MAE'] = np.abs(y_test - p_pred)#mean_absolute_error(y_test, p_pred)
        ress_test['MSE'] = np.power(y_test - p_pred, 2)#mean_squared_error(y_test, p_pred)
        ress_test['r2']  = r2_score(y_test, p_pred)
        ress_test['type'] = 'test'
        ress_test['Test size'] = test_size
        results.append(ress_test)

    data = pd.DataFrame(results)


    f, ax = plt.subplots()
    sns.lineplot(data, x='Test size', y='MSE', ax=ax, hue='type', estimator='sd')
    plt.show()

    return data
#

def make_correlation_study(X, y, regressor_kind, plot_r2=False):
    """
    Asses the regression and, if linear, plot the correlation between response and predicted.
    """

    reg = get_regressor(regressor_kind=regressor_kind)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y_true=y, y_pred=y_pred)
    print(f"\tResubstitution r2: {r2}")

    if plot_r2:
        f, ax = plt.subplots()
        sns.regplot(x=y_pred, y=y, color='b', ci=None, scatter_kws={'color':'b'}, line_kws={'color':'r'})
        ax.set_ylabel('mPAP (mmHg)')
        ax.set_xlabel('Regression value (mmHg)')
        ax.set_title(f'r2 = {r2:.3f}')
        plt.show()

    return reg
#

def make_extreme_shapes(X, pca, metadata, linreg):
    """
    Use the regression line as a deformation mode to show the importance of the underlying
    morfology related with the pressure increase.


    Arguments
    ---------

        X : np.ndarray (N, M)
            The dataset.

        mean : np.ndarray (M,)
            The mean of the cohort in case X has been previously centered (as sklearn's PCA does).

        metadata : np.ndarray
            The encoding metadata to reconstruct using the same parameters.

        coeffs : np.ndarray (M,)
            The regression coefficients.
    """

    #Let us compute the extreme values along the coeffs axis using the coordinates of the projections.
    axis = linreg.coef_/np.linalg.norm(linreg.coef_)
    coords = (axis.dot(X.T).T)
    exts = np.array([-2, 2])*np.std(coords) #np.quantiles(coords, q=[0.25, 0.75])#np.array([coords.min(), coords.max()])

    print("\tCoefficients:", linreg.coef_)
    print("\tIntercept:", linreg.intercept_)
    print("\tExtreme coordinates:", exts)

    axis_ = pca_inverse_transform_standardized(X=axis.reshape(1, -1), pca=pca).ravel()
    #axis_ = pca.inverse_transform(axis, ).ravel()

    p = pv.Plotter(shape=(1, 3))
    p.subplot(0, 0)
    pos_ext = vef.VascularEncoding.from_feature_vector(pca.mean_ + exts[0]*axis_, md=metadata)
    p.add_mesh(pos_ext.make_surface_mesh(tau_resolution=100, theta_resolution=150, RPA={'tau_resolution':40}), color='orange')
    p.add_text("Negative Extreme", position='upper_left', font_size=18, color='orange')

    p.subplot(0, 1)
    mean = vef.VascularEncoding.from_feature_vector(pca.mean_, md=metadata)
    p.add_mesh(mean.make_surface_mesh(tau_resolution=100, theta_resolution=150, RPA={'tau_resolution':40}), color='white')
    p.add_text("Mean shape", position='upper_left', font_size=18, color='k')

    p.subplot(0, 2)
    neg_ext = vef.VascularEncoding.from_feature_vector(pca.mean_ + exts[1]*axis_, md=metadata)
    p.add_mesh(neg_ext.make_surface_mesh(tau_resolution=100, theta_resolution=150, RPA={'tau_resolution':40}), color='purple')
    p.add_text("Positive Extreme", position='upper_left', font_size=18, color='purple')

    p.add_axes()
    p.link_views()
    p.show()
#

def ssm_pressure_relation(encodings, n_components, regressor_kind, plot_r2=True, plot_extreme_shapes=True):
    """
    """

    #Prepare data set in correct format
    X = np.array([encodings[i].to_feature_vector(mode='full', add_metadata=False) for i in encodings])
    pca = PCA(n_components=n_components, whiten=False).fit(X=X)
    if isinstance(n_components, int):
        print(f"\tWith {n_components} components the amount of variance explained is: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        print(f"\tTo explain {n_components} of variance, {pca.n_components_} components are needed")
    X_pca = pca_transform_standardized(X=X, pca=pca) #pca.transform(X=X)

    #Prepare response variable in the correct format
    y = p_data.loc[list(encodings.keys())].to_numpy()

    reg = make_correlation_study(X=X_pca, y=y, regressor_kind=regressor_kind, plot_r2=plot_r2)

    if plot_extreme_shapes:
        md = next(iter(encodings.values())).get_metadata()
        make_extreme_shapes(X=X_pca, pca=pca, metadata=md, linreg=reg)


    # make_cohort_convergence_study(X=X_pca, y=y, regressor_kind=regressor_kind, test_bounds=[0.1, 0.5], test_step=0.1)
    return
#

def main(suffix, regressor_kind='linear', plot_r2=True, plot_extreme_shapes=True):

    alignmnt = "GPA" if suffix == '_aligned' else "junction"
    print(f"Loading Cohort Encodings aligned with {alignmnt}")
    cohort = load_cohort_object(cohort_dir, which='encoding', exclude=exclude, keys_from_dirs=os.path.basename, suffix=suffix)
    print("......Done!")

    print("-"*50)
    print(f"Computing the pressure regression study on 8 components with suffix {suffix}")
    ssm_pressure_relation(encodings=cohort, n_components=8, regressor_kind=regressor_kind, plot_r2=plot_r2, plot_extreme_shapes=plot_extreme_shapes)
    print("-"*50)

    print("-"*50)
    print(f"Computing the pressure regression study on 10 components with suffix {suffix}")
    ssm_pressure_relation(encodings=cohort, n_components=10, regressor_kind=regressor_kind, plot_r2=plot_r2, plot_extreme_shapes=plot_extreme_shapes)
    print("-"*50)

    return
#

if __name__ == '__main__':
    main(suffix='_aligned', regressor_kind='linear', plot_r2=False, plot_extreme_shapes=True)
    main(suffix='_aligned_junction', regressor_kind='linear', plot_r2=True, plot_extreme_shapes=True)