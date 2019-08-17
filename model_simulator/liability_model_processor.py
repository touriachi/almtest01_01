import json
import logging
import logging.config
import os
import jsonschema
from ALM_model import Contract
import copy
import time


class LiabilityModel():

    def __init__(self, contract):
        # self.asset_model = contract.inputs.asset_model
        self.asset_model = copy.deepcopy(contract.inputs.asset_model)
        self.request = contract.request


    @classmethod
    def process(cls, data, execution_folder, log_file):

        """
            complete   asset , correlation matrix

            Parameters
            ----------
            input_file, : str
               asset , correlation matrix
            log_file: str
                log  file

            Returns
            -------
            Simulation object

        """

        path = os.path.dirname(os.path.abspath(__file__))
        logging.config.fileConfig(path + '\\logging.ini')

        fileh = logging.FileHandler(log_file, 'a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileh.setFormatter(formatter)

        logger = logging.getLogger('Acquisition')
        logger.addHandler(fileh)
        logger.debug('--Start Asset Simulation')




    def step02(self):



        i=1



