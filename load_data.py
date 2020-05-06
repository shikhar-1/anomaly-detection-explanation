import json
import pandas as pd
from rewards import Rewards

# Alternative 0: Get rewards by running on raw data
# Alternative 1: Import rewards using JSON file


class DataLoader:

    def __init__(self, dataPath , anomaly_strt_end , json_path = None, json_available = 1):

        self.json_path = json_path
        self.dataPath = dataPath
        self.anomaly_strt_end = anomaly_strt_end
        self.json_available = json_available

        self.rewards_obj = Rewards(self.dataPath, self.anomaly_strt_end)
        self.full_df = self.rewards_obj.df
        self.rewards_df = self.__get_rewards_df()

    def __get_rewards_df(self):

        # using the json path
        if self.json_available == 1:
            if self.json_path is None:
                raise Exception('Please provide json path or change json_available')
            else:
                json.load(open(self.json_path))
                jdf = pd.read_json(self.json_path, typ='series')
                reward_df = jdf.to_frame()
                reward_df.columns = ['Reward']
                return reward_df

        # using the input path of the raw data file and the input start and end timestamps for anomalous periods
        elif self.json_available == 0:
            reward_dict = self.rewards_obj.get_rewards_dict()
            reward_df = pd.DataFrame.from_dict(reward_dict, orient = 'index', columns = ['Reward'])
            return reward_df
        else:
            raise Exception('Please provide a valid json_available value - 1 or 0')

