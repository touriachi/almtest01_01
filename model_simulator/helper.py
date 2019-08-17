import json
import logging
import logging.config
import os
import jsonschema
from model_simulator.ALM_model  import *






class Helper():

    def initialize(self):
        self.data = []

    @classmethod
    def read_data(self, input_file, logger):

        data = input_file

        try:
            ##TODO TO  add  flag
            #jsonschema.validate(data, alm_schema)
            logger.debug("Well formed JSON file input")
            logger.debug("Valid JSON file input contract")
            i=1

        except jsonschema.exceptions.ValidationError as e:
            print("Well-formed but invalid JSON:", e)
            logger.error("Well-formed but invalid JSON:")
        except json.decoder.JSONDecodeError as e:
            print("Poorly-formed text, not JSON:", e)
            logger.error("Poorly-formed text, not JSON:")

        # create object
        contract = Contract()
        try:

            contract.set_version_control(data['Version Control'])
            # 'application' code
            contract.set_request(data['Request'])
            contract.set_inputs(data['Inputs'])

            logger.debug('Contract version :   %s', contract.version_control.conv)
            logger.debug('Calculator version : %s', contract.version_control.calv)
            logger.debug('Request execution type : %s', contract.request.et)
            logger.debug('Request execution request id: %s', contract.request.rid)
            logger.debug('Request execution request user: %s', contract.request.uid)
            logger.debug('--Acquisition done')
            print('Request execution type :', contract.request.et)
            return contract
        except:
            print("Unexpected error :")
            logger.error("Unexpected error")




