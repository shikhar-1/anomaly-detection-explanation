from load_data import DataLoader
from filtering import get_final_features

# path of the csv here
dataPath_1 = '../TrainData/preprocessed_benchmark_userclicks_14_19_10003_1000000_batch146_19_.csv'
dataPath_2 = '../TrainData/preprocessed_benchmark_userclicks_11_20_10003_1000000_batch146_17_.csv'
dataPath_3 = '../TrainData/preprocessed_benchmark_userclicks_1_18_10000_1000000_batch146_20_.csv'

# copy corresponding anomalous intervals from the Excel file here
anomaly_strt_end_1 = [(1529015043,1529016293),(1528999019,1528999419),(1529002179,1529003079),(1529007602,1529008102),
                    (1529030089,1529031589),(1529023063,1529023763)]
anomaly_strt_end_2 = [(1529015034,1529016293),(1528999019,1528999419),(1529002179,1529003079),(1529030089,1529031589),
                      (1529007602,1529008102),(1529023063,1529023763)]
anomaly_strt_end_3 = [(1529015034,1529016234),(1528999019,1528999419),(1529002179,1529003079),(1529007602,1529008102),
                      (1529030089,1529031589),(1529023063,1529023763)]

# json path here, if available
json_path = "../reward.json"

data = DataLoader(json_path = json_path, dataPath = dataPath_1, anomaly_strt_end = anomaly_strt_end_1, json_available = 1)

full_df = data.full_df
reward_df = data.rewards_df
# print(reward_df)
feature_list = get_final_features(reward_df, full_df)

print(feature_list)