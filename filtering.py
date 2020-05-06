import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as spc


def get_final_features(reward_df, full_df):

    # Correlation Filtering
    columns = list(full_df.columns)
    columns.remove('t')
    columns.remove('anomalous')

    reward_filtered = full_df[columns]

    pearsoncorr = reward_filtered.corr(method='pearson')
    correlations = pearsoncorr.values

    pdist = spc.distance.pdist(correlations)
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, 0.30 * pdist.max(), 'distance')

    corr_filt_df = pd.DataFrame()
    corr_filt_df['cols'] = columns
    corr_filt_df['cluster'] = idx
    reward_df_new = reward_df.reset_index()
    corr_filt_df = corr_filt_df.merge(reward_df_new, left_on='cols', right_on='index')[['cols','cluster','Reward']]
    temp_df = corr_filt_df.groupby(by = 'cluster')['Reward'].max().reset_index()
    temp_df = corr_filt_df.merge(temp_df, left_on = ['Reward','cluster'], right_on = ['Reward','cluster'])
    temp_df = temp_df.drop_duplicates(subset = ['cluster'], ignore_index=1)
    temp_df = temp_df[['cols','Reward']]
    temp_df = temp_df.set_index(keys=['cols'])

    # Step Reward Filtering
    sorted_df = temp_df.sort_values(by=['Reward'], axis=0, ascending=False)
    upper = sorted_df['Reward'][0]
    change = []
    for event, rew in sorted_df.iterrows():
        lower = rew['Reward']
        rdiff = upper - lower
        change.append(rdiff)
        upper = lower
    sorted_df['Change'] = change
    rank = np.arange(len(sorted_df))
    sorted_df['Rank'] = rank
    max_row = np.argmax(sorted_df['Change'].values)
    result_df = sorted_df[(sorted_df['Rank'] < max_row)]

    return list(result_df.index)
