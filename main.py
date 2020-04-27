from rewards import Rewards

dataPath = '../TrainData/preprocessed_benchmark_userclicks_14_19_10003_1000000_batch146_19_.csv'
anomaly_strt_end = [(1529015043,1529016293),(1528999019,1528999419),(1529002179,1529003079),(1529007602,1529008102),
                    (1529030089,1529031589),(1529023063,1529023763)]

reward_dict = Rewards(dataPath, anomaly_strt_end).get_rewards_dict()

print(reward_dict)