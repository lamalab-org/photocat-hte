import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12


def poly(a, x):
    a = np.flip(a)
    y = a[0] * x**0

    for i in range(1, len(a)):
        y += a[i] * x**i

    return y

def residual_generic(p, x, y, function):
    y_fit = function(p, x)
    res = y - y_fit

    return res

def skewed_gaussian_model(p, x):
    return p[0] * skewnorm.pdf(x, p[3], loc=p[1], scale=p[2])

def combined_model(datapoints, functions, parameters, offset):

    result = np.zeros_like(datapoints[0])
    
    for data, function, parameter in zip(datapoints, functions, parameters):
        result += function(parameter, data)

    return result - offset

def residual_combined_model(offset, datapoints, functions, parameters, y):

    y_fit = combined_model(datapoints, functions, parameters, offset)
    residuals = y - y_fit

    return np.sum(residuals**2)

def optimize_offset(df, functions, parameters, columns = ['c([Ru(bpy(3]Cl2) [M]', 'c(Na2S2O8) [M]', 'pH [-]', 'mean_rate']):

    filtered_df = filter_data(df, [], [], drop_ZA = True)
    selected_df = filtered_df[columns]
    data = selected_df.to_numpy()

    datapoints = [data[:, i] for i in range(len(data[0]))]
    rates = data[:,-1]

    initial_guess = 0

    p = minimize(residual_combined_model, initial_guess, args = (datapoints, functions, parameters, rates))

    return p.x[0]
    
def import_excel(filename):
    df = pd.read_excel(filename, sheet_name='Tabelle1')
    return df

def import_data(excel_file, csv_file):

    df = import_excel(excel_file)
    df_csv = pd.read_csv(csv_file)

    df_csv['Experiment'] = df_csv['Filename'].apply(lambda x: x.split('_')[1])
    df_csv['Experiment'] = df_csv['Experiment'].apply(lambda x: x.split('.')[0])

    df_merged = pd.merge(df, df_csv[['Experiment', 'Rate']], on='Experiment', how='left')

    df_cleaned = df_merged.dropna(subset=['Rate'])
    df_cleaned = df_cleaned[df_cleaned['Rate'] != 0]

    return df_cleaned

def average_reproductions(df_cleaned):

    df_cleaned['Base Experiment'] = df_cleaned['Experiment'].apply(lambda x: x[:-2])

    grouped_df = df_cleaned.groupby('Base Experiment').agg(
        mean_rate=('Rate', 'mean'),
        min_rate=('Rate', 'min'),
        max_rate=('Rate', 'max')
    ).reset_index()

    base_experiment_df = df_cleaned[df_cleaned['Experiment'].str.endswith('-1')].merge(grouped_df, on='Base Experiment')

    base_experiment_df['Experiment'] = base_experiment_df['Base Experiment']
    base_experiment_df = base_experiment_df.drop(columns = ['Base Experiment', 'Rate', 'rate', 'annotations'])

    return base_experiment_df

def filter_data(df, column_names, column_values, drop_ZA = False):

    filtered_df = df

    for name, value in zip(column_names, column_values):
        filtered_df = filtered_df[filtered_df[name] == value]

    if drop_ZA:
        filtered_df = filtered_df[~filtered_df['Experiment'].isin(['MRG-059-ZA-6', 'MRG-059-ZA-7'])]
  
    return filtered_df

def plot_data(filtered_df, column_name, legend = True):

    fig, ax = plt.subplots(figsize = (10,6))
    fig.subplots_adjust(right = 0.8)

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=len(filtered_df) - 1)

    counter = 0

    for index, row in filtered_df.iterrows():
        x = row[column_name]
        y = row['mean_rate']
        yerr = [[row['mean_rate'] - row['min_rate']], [row['max_rate'] - row['mean_rate']]]
      
        color = cmap(norm(counter))

        plt.errorbar(x, y, yerr = yerr, fmt='o', ecolor=color, 
                 capsize=3, capthick=1, markersize=5,label = row['Experiment'], color = color)
        
        counter += 1
        
    ax.set_xlabel(column_name)
    ax.set_ylabel(r'Rate / $\mu mol(O_2) \, s^{-1}$')
    ax.set_title(f'{column_name} vs. rate')

    if legend:
        ax.legend(title='Base Experiment', bbox_to_anchor=(1.02, 1.01), loc='upper left')

    return fig, ax

def construct_model(df_averaged, column_names, column_values, column_of_interest, model, 
                    drop_ZA = False, plotting = False, save_fig = False):

    df_filtered = filter_data(df_averaged, column_names, column_values, drop_ZA = drop_ZA)

    x = df_filtered[column_of_interest]
    x_full = np.linspace(df_filtered[column_of_interest].min(), df_filtered[column_of_interest].max(), 100)
    y = df_filtered['mean_rate']

    if model == skewed_gaussian_model:
        popt = least_squares(fun=residual_generic, x0 = [1, np.mean(x), np.std(x), 1], args=(x, y, skewed_gaussian_model)).x
        y_fit = skewed_gaussian_model(popt, x_full)
    elif model == poly:
        popt = np.polyfit(x, y, 2)
        y_fit = poly(popt, x_full)
        
    else:
        raise Exception("Model not found")

    if plotting:
        fig, ax = plot_data(df_filtered, column_of_interest, legend = False)
        ax.plot(x_full, y_fit, color = 'black', label = 'Fit')
        ax.legend(bbox_to_anchor=(1.02, 1.01), loc='upper left')

        if save_fig:
            fig.savefig(f'/Users/jacob/Documents/Water_Splitting/HTE_Photocatalysis/{column_of_interest}_analyzed.pdf')

    return x, y, x_full, popt

class Model:
    def __init__(self, df_averaged, column_names, column_values, column_of_interest, model, 
                    drop_ZA = False, plotting = False, save_fig = False):
        
        self.column_of_interest = column_of_interest
        self.model = model
        self.column_names = column_names
        self.column_values = column_values
        self.df_averaged = df_averaged

        self.x, self.y, self.x_full, self.popt = construct_model(df_averaged, column_names, column_values, column_of_interest, model, 
                    drop_ZA = drop_ZA, plotting = plotting, save_fig = save_fig)

def plot_3D_model_fit(model_X_axis, model_Y_axis, other_models, other_models_values, drop_ZA = True):

    x = model_X_axis.x_full
    y = model_Y_axis.x_full

    X, Y = np.meshgrid(x, y)
    datapoints = [X, Y]

    functions = [model_X_axis.model, model_Y_axis.model]
    parameters = [model_X_axis.popt, model_Y_axis.popt]
    names = [model_X_axis.column_of_interest, model_Y_axis.column_of_interest]

    other_models_names = []

    for model, value in zip(other_models, other_models_values):
        W = np.ones_like(X) * value
        datapoints.append(W)
        functions.append(model.model)
        parameters.append(model.popt)
        other_models_names.append(model.column_of_interest)
        names.append(model.column_of_interest)
    
    names.append('mean_rate')
    
    offset = optimize_offset(model_X_axis.df_averaged, functions, parameters, columns = names)

    Z = combined_model(datapoints, functions, parameters, offset)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel(model_X_axis.column_of_interest)
    ax.set_ylabel(model_Y_axis.column_of_interest)
    ax.set_zlabel(r'Rate / $\mu mol(O_2) \, s^{-1}$')

    filtered_data = filter_data(model_X_axis.df_averaged, other_models_names, other_models_values, drop_ZA = drop_ZA)

    ax.scatter(
    filtered_data[model_X_axis.column_of_interest],
    filtered_data[model_Y_axis.column_of_interest],
    filtered_data['mean_rate'],
    c='b', marker='o')

def plot_2D_model_fit(model_X_axis, other_models, other_models_values, drop_ZA = True):

    x = model_X_axis.x_full

    datapoints = [x]
    functions = [model_X_axis.model]
    parameters = [model_X_axis.popt]
    names = [model_X_axis.column_of_interest]
    other_models_names = []

    for model, value in zip(other_models, other_models_values):
        y = np.ones_like(x) * value
        datapoints.append(y)
        functions.append(model.model)
        parameters.append(model.popt)
        names.append(model.column_of_interest)
        other_models_names.append(model.column_of_interest)

    names.append('mean_rate')
    
    offset = optimize_offset(model_X_axis.df_averaged, functions, parameters, columns = names)
    print(offset)

    Z = combined_model(datapoints, functions, parameters, offset)

    filtered_data = filter_data(model_X_axis.df_averaged, other_models_names, other_models_values, drop_ZA = drop_ZA)

    fig, ax = plot_data(filtered_data, model_X_axis.column_of_interest, legend = False)

    ax.plot(x, Z, color = 'green', label = 'Combined model fit')
    ax.legend(bbox_to_anchor=(1.02, 1.01), loc='upper left')


def main():
    
    df_cleaned = import_data('/Users/jacob/Documents/Water_Splitting/HTE_Photocatalysis/data/HTE-overview_240815.xlsx',
                                '/Users/jacob/Documents/Water_Splitting/HTE_Photocatalysis/data/analyzed_csv/output.csv')
    df_averaged = average_reproductions(df_cleaned)

    Ru_model = Model(df_averaged, ['c(Na2S2O8) [M]', 'pH [-]'], [0.06, 9.6], 'c([Ru(bpy(3]Cl2) [M]',
                     skewed_gaussian_model, plotting = False, save_fig=False)
    
    Ox_model = Model(df_averaged, ['c([Ru(bpy(3]Cl2) [M]', 'pH [-]'], [0.00001, 9.6], 'c(Na2S2O8) [M]', 
                    skewed_gaussian_model, plotting = False, save_fig=False)
    
    pH_model = Model(df_averaged, ['c([Ru(bpy(3]Cl2) [M]', 'c(Na2S2O8) [M]'], [0.00001, 0.06], 'pH [-]', 
                    poly, drop_ZA = True, plotting = False, save_fig=False)
    
    plot_3D_model_fit(Ru_model, Ox_model, [pH_model], [9.6])
    #plot_3D_model_fit(Ru_model, pH_model, [Ox_model], [0.06])
    #plot_3D_model_fit(pH_model, Ox_model, [Ru_model], [0.00001])

    #plot_2D_model_fit(Ru_model, [Ox_model, pH_model], [0.06, 9.6])
    #plot_2D_model_fit(pH_model, [Ru_model, Ox_model], [0.00001, 0.06])
    #plot_2D_model_fit(Ox_model, [Ru_model, pH_model], [0.00001, 9.6])

if __name__ == '__main__':
    main()
    plt.show()