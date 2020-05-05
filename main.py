from rewards import Rewards

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


# get the rewards dictionary
reward_dict = Rewards(dataPath_1, anomaly_strt_end_1).get_rewards_dict()

print(reward_dict)