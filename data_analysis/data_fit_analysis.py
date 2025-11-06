from fit import fit_data
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


def import_data(file_name):
    df = pd.read_csv(file_name, low_memory = False)
    return df

def main():

    with open('../data/analyzed_csv/output.csv', mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Rate'])

        for file in os.listdir('../data/csv'):
            #if file.endswith('Z-3-1.csv'):
            df = import_data(f'../data/csv/{file}')

            try:
                rate = fit_data(df, filename = f'../data/png/{file}_fit.png', plotting = True, plot_baseline = True)
            except Exception:
                rate = 0

            csvwriter.writerow([file, rate])
            print(f'{file} analyzed, rate: {rate}')





if __name__ == '__main__':
    main()