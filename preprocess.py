import pandas as pd
import glob
import os
import numpy as np
import argparse

def process(df_dataset, filename, malicious=True):
    columns_to_drop = [
        'Dst Port',
        'Timestamp',
        'Fwd PSH Flags',
        'Bwd PSH Flags',
        'Fwd URG Flags',
        'Bwd URG Flags',
        'Flow Byts/s',  # some np.inf values
        'Flow Pkts/s',  # some np.inf values
        'Dst IP',   #These fields are in some days but not others
        'Flow ID',
        'Src IP',
        'Src Port'
    ]
    # Step 2-3
    print(f'Labels: {df_dataset.Label.unique()}')
    df_dataset.drop(df_dataset.loc[df_dataset["Label"] == "Label"].index, inplace=True)
    if malicious:
        print('Test set with only malicous labels')
        df_dataset.replace({'Label': 'No Label'}, 'Malicious', inplace=True)
    else:
        mal_labels = set(df_dataset['Label'].unique())
        mal_labels.remove('Benign')
        df_dataset.replace({'Label': mal_labels}, 'Malicious', inplace=True)
        print(f'two labels: {df_dataset.Label.unique()}')

    #Step 4
    print(f'protocols: {df_dataset.Protocol.unique()}')
    df_dataset = df_dataset.astype({"Protocol": str})
    df_dataset = pd.get_dummies(df_dataset, columns=['Protocol'], drop_first=True)
    # making Label column the last column again
    df_dataset.insert(len(df_dataset.columns)-1, 'Label', df_dataset.pop('Label'))
    # Step 5-6
    # df_dataset.drop(columns=columns_to_drop, inplace=True)
    df_dataset.drop(df_dataset.filter(columns_to_drop), axis=1, inplace=True) # axis=1 for columns
    df_dataset.dropna(inplace=True)
    df_dataset.drop_duplicates(inplace=True)

    if not malicious:
        bsum = (df_dataset["Label"].value_counts()[['Benign']].sum())
        msum = (df_dataset["Label"].value_counts()[['Malicious']].sum())
        print(f'Benign percentage: {bsum / (bsum + msum)}')
        df_dataset.replace(to_replace="Benign", value=0, inplace=True)
    df_dataset.replace(to_replace="Malicious", value=1, inplace=True)
    print(df_dataset.info())
    df_dataset.to_csv(filename, index=False)
    return df_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default="Processed Traffic Data for ML Algorithms")
    parser.add_argument('-s', '--specific', default='all')
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('-m', '--malicious', action='store_true')
    args = parser.parse_args()

    # code for over all days, can add options to split by specific days
    #joined_files = os.path.join(args.data, "*.csv")
    part = "*.csv"
    if args.specific != 'all':
        part = args.specific
    joined_files = os.path.join(args.data, part + "*")
    print(joined_files)
    joined_list = glob.glob(joined_files)
    print('Files used:')
    print(joined_list)
    if args.out == None:
        out = part + ".csv"
    else:
        out = args.out
    print(f'Will output to: {out}')
    all_df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
    process(all_df, out, args.malicious)
