from flask import Flask, request, abort
from azure.storage.blob import BlockBlobService, PublicAccess
from azure.storage.common import CloudStorageAccount
from model_simulator import simulator
import logging
import json


account_name = 'toufikstorage01'
account_key ='XFoSjF0tQgpP4BRypJjbB/HGtHKDa68tqd+anK19LCCFIAPnSHr3mzFdgIYbfO2A0deffu4zmIfRCv5xYYpdGw=='
account = CloudStorageAccount(account_name, account_key)
container_name='jsoncontainer'


app = Flask(__name__)
logging.basicConfig(filename='alm.log', level=logging.DEBUG)

def get_jsonfile( account, ocntainer_name, file):
    try:
        block_blob_service = BlockBlobService(account.account_name, account.account_key)
        # Download the blob(s).
        inputJSON = block_blob_service.get_blob_to_bytes(container_name, file)
        stringInput = (inputJSON.content).decode('utf8')
        return  json.loads(stringInput)
    except Exception as e:
        app.logger.error('Error occurred in get_jsonfile.', e)


@app.route('/api/updates', methods=['POST'])
def process():
    if request.method == 'POST':
        req = request.json
        app.logger.error("hello  toufik process start .\n")
        input_file_name=req[0]['data']['key']

        # on recupere  le bloob store
        json_inputfile = get_jsonfile(account, container_name, input_file_name)

        ###-------------------------------------- debut appel du calculateur
        json_result = dict()

        simulator.Simulator.run_process(input_file_name, json_inputfile, json_result, app.logger)
        app.logger.info('Tratiement fini avec succes .\n')
        return '', 200
    else:
        abort(400)


@app.route('/')
def hello_world():
    app.logger.info('le site est fonctionnel')

    input_file_name = 'Input_file_002.json'

    # on recupere  le bloob store
    json_inputfile = get_jsonfile(account, container_name, input_file_name)
    app.logger.info('Fichier lu .\n')

    ###-------------------------------------- debut appel du calculateur
    json_result = dict()

    simulator.Simulator.run_process(input_file_name, json_inputfile, json_result, app.logger)
    app.logger.info('Tratiement fini avec succes .\n')
    return 'Hello World!'


if __name__ == '__main__':
    app.run()   
    #app.run(host='127.0.0.1', port=5000, threaded=True)

