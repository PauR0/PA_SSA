
import os

import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import (mean_absolute_error, r2_score)

import vascular_encoding_framework as vef
from vef_scripts.vef_cohort import load_cohort_object

from configuration import (cohort_dir,
                           exclude,
                           p_data)


def show_pca_variance(pca, n_comp=None, show=True, thrs=None, ax=None):

    if n_comp is None:
        n_comp = pca.n_components_


    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
        fig = None

    if thrs is not None:
        if thrs < 1:
            thrs *= 100

    #plt.rcParams.update({'font.size': 22})

    ax1.set_facecolor(color='w')
    color = 'dimgrey'
    ax1.set_xlabel('principal components')
    ax1.set_ylabel('individual variance (%)', color=color)
    ax1.bar(np.arange(1, n_comp+1), pca.explained_variance_ratio_[:n_comp]*100, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(visible=True, color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'g'
    ax2.set_ylabel('cumulative variance(%)', color=color)  # we already handled the x-label with ax1
    ax2.grid(visible=True, color=color)
    var = [np.sum(pca.explained_variance_ratio_[:n+1])*100 for n in range(n_comp)]
    ax2.plot(np.arange(1, n_comp+1), var, '-o', color=color)#mec=color, mfc=color)
    if thrs:
        ax2.axhline(thrs, linestyle='-.', color='gray')
    ax2.tick_params(axis='y', labelcolor=color)

    if fig is not None:
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if show:
        plt.show()

    #plt.style.use('default')
#

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

def get_regressor(regressor_kind, n_components=1):
    """
    Get the appropriate regressor based on regressor_kind
    """
    regressors = {
        'linear' : LinearRegression(),
        'lasso'  : Lasso(),
        'pls'    : PLSRegression(n_components=n_components)
    }
    return regressors[regressor_kind]
#

def make_cohort_convergence_study(X, y, regressor_kind, test_sizes=None):
    """
    Display the results of the regression results.
    """

    if test_sizes is None:
        test_sizes = np.linspace(0.1, 0.5, 5)

    results = []
    for test_size in test_sizes:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True)

        regr = get_regressor(regressor_kind)
        regr.fit(X_train, y=y_train)

        y_pred = regr.predict(X_train)
        mae, mse = np.abs(y_train-y_pred), np.power(y_train-y_pred, 2)
        for ae, se in zip(mae, mse):
            results.append({'Absolute Error (mmHg)': ae[0], 'Squared Error': se[0], 'set': 'train', 'Test size': f"{test_size:.2f}"})

        y_pred = regr.predict(X_test)
        mae, mse = np.abs(y_test-y_pred), np.power(y_test-y_pred, 2)
        for ae, se in zip(mae, mse):
            results.append({'Absolute Error (mmHg)': ae[0], 'Squared Error': se[0], 'set': 'test', 'Test size': f"{test_size:.2f}"})
    results = pd.DataFrame(results)

    f, ax = plt.subplots(1, 2)
    sns.boxplot(results, x='Test size', y='Absolute Error (mmHg)', ax=ax[0], hue='set')
    sns.lineplot(results, x='Test size', y='Absolute Error (mmHg)', ax=ax[1], hue='set', errorbar='sd')
    plt.show()

    return results
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
    pos_mesh = pos_ext.make_surface_mesh(tau_resolution=100, theta_resolution=150, RPA={'tau_resolution':40})
    p.add_mesh(pos_mesh, color='orange')
    p.add_text("Negative Extreme", position='upper_left', font_size=18, color='orange')
    pos_ext.to_multiblock().save('meshes/positive_shape.vtm')

    p.subplot(0, 1)
    mean = vef.VascularEncoding.from_feature_vector(pca.mean_, md=metadata)
    mean_mesh = mean.make_surface_mesh(tau_resolution=100, theta_resolution=150, RPA={'tau_resolution':40})
    p.add_mesh(mean_mesh, color='white')
    p.add_text("Mean shape", position='upper_left', font_size=18, color='k')
    mean.to_multiblock().save('meshes/mean_shape.vtm')

    p.subplot(0, 2)
    neg_ext = vef.VascularEncoding.from_feature_vector(pca.mean_ + exts[1]*axis_, md=metadata)
    neg_mesh = neg_ext.make_surface_mesh(tau_resolution=100, theta_resolution=150, RPA={'tau_resolution':40})
    p.add_mesh(neg_mesh, color='purple')
    p.add_text("Positive Extreme", position='upper_left', font_size=18, color='purple')
    neg_ext.to_multiblock().save('meshes/negative_shape.vtm')

    p.add_axes()
    p.link_views()
    p.show()
#

def ssm_pressure_relation(encodings, n_components, regressor_kind='linear',
                          plot_pca_variance=True,
                          plot_r2=True,
                          plot_extreme_shapes=True,
                          plot_cohort_convergence=True):
    """
    """

    #Prepare data set in correct format
    X = np.array([encodings[i].to_feature_vector(mode='full', add_metadata=False) for i in encodings])
    pca = PCA(n_components=n_components, whiten=False).fit(X=X)
    if isinstance(n_components, int):
        print(f"\tWith {n_components} components the amount of variance explained is: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        print(f"\tTo explain {n_components} of variance, {pca.n_components_} components are needed")

    if plot_pca_variance:
        show_pca_variance(pca, n_comp=None, show=True)

    X_pca = pca_transform_standardized(X=X, pca=pca) #pca.transform(X=X)

    #Prepare response variable in the correct format
    y = p_data.loc[list(encodings.keys())].to_numpy()

    reg = make_correlation_study(X=X_pca, y=y, regressor_kind=regressor_kind, plot_r2=plot_r2)

    if plot_extreme_shapes:
        md = next(iter(encodings.values())).get_metadata()
        make_extreme_shapes(X=X_pca, pca=pca, metadata=md, linreg=reg)

    if plot_cohort_convergence:
        make_cohort_convergence_study(X=X_pca, y=y, regressor_kind=regressor_kind, test_sizes=None)
    return
#

def main(suffix, regressor_kind='linear', plot_r2=True, plot_extreme_shapes=True, plot_cohort_convergence=True):

    alignmnt = "GPA" if suffix == '_aligned' else "junction"
    print(f"Loading Cohort Encodings aligned with {alignmnt}")
    cohort = load_cohort_object(cohort_dir, which='encoding', exclude=exclude, keys_from_dirs=os.path.basename, suffix=suffix)
    print("......Done!")

    print("-"*50)
    print(f"Computing the pressure regression study on 8 components with suffix {suffix}")
    ssm_pressure_relation(encodings=cohort, n_components=8, regressor_kind=regressor_kind, plot_r2=plot_r2, plot_extreme_shapes=plot_extreme_shapes, plot_cohort_convergence=plot_cohort_convergence)
    print("-"*50)

    print("-"*50)
    print(f"Computing the pressure regression study on 10 components with suffix {suffix}")
    ssm_pressure_relation(encodings=cohort, n_components=10, regressor_kind=regressor_kind, plot_r2=plot_r2, plot_extreme_shapes=plot_extreme_shapes, plot_cohort_convergence=plot_cohort_convergence)
    print("-"*50)

    return
#

def r2_over_pca(X, y, regresor_kind, thrs, n_max=50):
    """
    """

    pca = PCA().fit(X=X)
    f, ax = plt.subplots(1, 2)
    show_pca_variance(pca=pca, thrs=thrs, n_comp=n_max, ax=ax[0], show=False)

    results = []
    for i in range(1, n_max):
        pca = PCA(n_components=i).fit(X=X)
        X_pca = pca_transform_standardized(X=X, pca=pca)
        reg = get_regressor(regressor_kind=regresor_kind).fit(X_pca, y)
        results.append({'number of components':i, 'r2':reg.score(X_pca, y)})
    results = pd.DataFrame(results)


    results.plot(x='number of components', y='r2', ax=ax[1], marker="o")
    ax[1].set_ylabel('r2')
    ax[1].grid(visible=True)
    f.tight_layout()
    plt.show()
#

def cross_validate_regression(X, y, regressor_kind, n_splits=3, n_components=.95):
    """
    Perform a K fold cross validation on regressor using r2 and mean absolute error.
    """

    kf = KFold(n_splits=n_splits)


    results = []
    for i, (train_id, test_id) in enumerate(kf.split(X)):

        pca = None
        if regressor_kind in ['lasso', 'linear']:
            pca = PCA(n_components=n_components).fit(X=X[train_id])
            X = pca_transform_standardized(X=X, pca=pca)

        reg = get_regressor(regressor_kind=regressor_kind, n_components=n_components).fit(X[train_id], y[train_id])
        y_test_pred = reg.predict(X[test_id])
        y_train_pred = reg.predict(X[train_id])
        results.append({'Fold': i,
                        'r2 (test)'   : r2_score(y[test_id], y_test_pred),
                        'MAE (test)'  : mean_absolute_error(y[test_id], y_test_pred),
                        'r2 (train)'  : r2_score(y[train_id], y_train_pred),
                        'MAE (train)' : mean_absolute_error(y[train_id], y_train_pred),
                        "n_modes"     : pca.n_components_ if regressor_kind in ['lasso', 'linear'] else reg.x_weights_.shape[1]})
    results = pd.DataFrame(results)

    var_msg = f" explaining/using {n_components} variance/modes" if regressor_kind in ['lasso', 'linear'] else ""
    print(f"Results of The {n_splits}-fold cross validation{var_msg} with {regressor_kind} regression:")
    print(results.to_markdown(floatfmt=".2f"))
    print(results.mean().T.to_markdown(floatfmt=".2f"))
#

def alternative(suffix='_aligned', exp_var=.95, regressor_kind='linear', plot_r2_over_pca=True, cv_assessment=True, n_splits=3, pls_assessment=True, pls_components=1):
    """
    TODO
    """

    alignmnt = "GPA" if suffix == '_aligned' else "junction"
    print(f"Loading Cohort Encodings aligned with {alignmnt}")
    cohort = load_cohort_object(cohort_dir, which='encoding', exclude=exclude, keys_from_dirs=os.path.basename, suffix=suffix)
    print("......Done!")

    #Prepare data set in correct format
    X = np.array([cohort[i].to_feature_vector(mode='full', add_metadata=False) for i in cohort])
    #Prepare response variable in the correct format
    y = p_data.loc[list(cohort.keys())].to_numpy()


    if plot_r2_over_pca:
        r2_over_pca(X=X, y=y, regresor_kind=regressor_kind, thrs=exp_var)

    if cv_assessment:
        cross_validate_regression(X=X, y=y, regressor_kind=regressor_kind, n_splits=n_splits, n_components=exp_var)

    if pls_assessment:
        cross_validate_regression(X=X, y=y, regressor_kind='pls', n_splits=n_splits, n_components=pls_components)
#




if __name__ == '__main__':

    main(suffix='_aligned', regressor_kind='linear', plot_r2=True, plot_extreme_shapes=True, plot_cohort_convergence=True)
    main(suffix='_aligned_junction', regressor_kind='linear', plot_r2=True, plot_extreme_shapes=True)

    alternative(suffix='_aligned', exp_var=.95, regressor_kind='linear', plot_r2_over_pca=True, cv_assessment=True, n_splits=3, pls_assessment=True, pls_components=1)
#
