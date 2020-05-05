import pandas as pd
import numpy as np


class Rewards:

    def __init__(self, dataPath, anomaly_strt_end):

        # using the input start and end timestamps for anomalous periods
        self.anomaly_strt_end = anomaly_strt_end
        # using the input path of the file
        self.dataPath = dataPath
        # read csv into DataFrame
        self.df = pd.read_csv(dataPath)
        # dropping empty columns
        self.drop_empty_cols()
        # create class (anomalous/reference) column
        self.create_class_col()
        # get class entropy
        self.class_entropy = self.calc_class_entropy()

    def get_rewards_dict(self):
        """

        Call this method on the initialized object to get the sorted reward dictionary (no arguments required)

        Returns:
            A sorted dictionary containing rewards for each feature

        """
        final_dict = {}
        j = 0
        for colName, _ in self.df.iteritems():
            # j+=1
            # if j ==7:
            #     break
            if colName not in ['t', 'anomalous']:

                df_temp = self.df[[colName, 'anomalous']]
                map_dict = dict(df_temp.groupby(by=[df_temp[colName], df_temp['anomalous']]).count().reset_index()[
                                    colName].value_counts())
                df_temp['segment'] = df_temp.apply(lambda x: 2 if map_dict[x[colName]] > 1 else x['anomalous'], axis=1)

                df_temp = df_temp.sort_values(by=[colName])
                seg_entropy = self.calc_seg_entropy(df_temp)

                reward = self.class_entropy / seg_entropy
                final_dict.update({colName: reward})
        final_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
        return final_dict

    def calc_seg_entropy(self, df):
        """

        Method used to calculate the total segmentation entropy for a single feature

        Arguments:
            df (Pandas DataFrame): DataFrame with 3 columns - feature, anomalous, segment
        Returns:
            entropy (float): The segmentation entropy for the feature

        """
        x = list(df['segment'])
        count = 0
        seg = 1
        seg_dict = {}
        for i in range(len(x)):
            if i == 0:
                temp_val = x[i]
                count += 1
            else:
                if x[i] == temp_val:
                    count += 1
                    if i == len(x) - 1:
                        final_count = count
                        seg_dict.update({seg: (final_count, x[i])})
                else:
                    final_count = count
                    seg_dict.update({seg: (final_count, x[i - 1])})
                    if i == len(x) - 1:
                        seg += 1
                        seg_dict.update({seg: (1, x[i])})

                    count = 0
                    seg += 1

            temp_val = x[i]

        entropy = 0
        penalty = 0
        for val in seg_dict.values():
            if int(val[1]) == 2:
                p_i = int(val[0] / 2) / len(x)
                p_j = (val[0] - int(val[0] / 2)) / len(x)
                a = 0
                b = 0
                if p_i != 0:
                    a = (-1 * p_i * np.log(p_i))
                if p_j != 0:
                    b = (-1 * p_j * np.log(p_j))
                penalty += a + b
            c = 0
            p = val[0] / len(x)
            if p != 0:
                c = -1 * p * np.log(p)
            entropy += c
        entropy += penalty
        return entropy

    def drop_empty_cols(self):
        """

        Used to drop columns that have NA in all its rows
        Method is run when the object is created

        """
        drop_cols_list = []
        for colName, col in self.df.iteritems():
            if col.isna().all():
                drop_cols_list.append(colName)
        self.df = self.df.drop(columns=drop_cols_list)

    def create_class_col(self):
        """

        Used to create a column that identifies if a row belongs to the reference or the anomalous interval
        Method is run when the object is created

        """

        self.df['anomalous'] = self.df['t'].apply(lambda x: self.create_class_lambda(x))

    def create_class_lambda(self, col_val):
        """

        Method applied to each row of the DataFrame to get the 'anomalous' column

        """
        for limits in self.anomaly_strt_end:
            if col_val >= limits[0] and col_val <= limits[1]:
                return 1
        return 0

    def calc_class_entropy(self):
        """

        Used to calculate the class entropy
        Method is run when the object is created

        """

        p_i = self.df.anomalous.value_counts()[0] / len(self.df)
        p_j = self.df.anomalous.value_counts()[1] / len(self.df)
        return (-1 * p_i * np.log(p_i)) + (-1 * p_j * np.log(p_j))
