"""
    The asset simulation approach ....
    ...
"""
import getopt
import sys
import time
import logging
import os
import copy
from logging.config import fileConfig
from model_simulator.helper import Helper
from model_simulator.asset_model_processor import  AssetModelProcessor






class  Simulator () :

    def __init__(self, input_file, logger):
        # self.asset_model = contract.inputs.asset_model
        self.input = copy.deepcopy(input_file)
        self.logger = logger

    @classmethod
    def run_process(cls,filename, input_file,json_result, logger):


        logger.debug('Start process in calculator ' + filename)


        processor = cls(input_file, logger)

        print("Step 1 started : Read and Convert json object to Python")

        data = Helper.read_data(input_file, logger)
        print("Step 1 finished.")

        result = AssetModelProcessor.process(data, logger)

        print("Step 2 finished.")




# First, we load the current  ALM into memory as an array of lines
if __name__ == "__main__":
    filename ="sdsd"
    input_file="input_fileinput_fil"
    json_result=dict()
    azure_logger=""

    Simulator.run_process (filename, input_file,json_result, azure_logger)

