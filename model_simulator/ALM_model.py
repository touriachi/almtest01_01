import pandas as pd
import copy

# base class
class Contract:

    def __init__(self):
        self.version_control = []
        self.request = []
        self.inputs = []

    def set_version_control(self, data):
        self.version_control = Version_Control(data)

    def set_request(self, data):
        self.request = Request(data)

    def set_inputs(self, data):
        self.inputs = Inputs.create_items(data)


class Version_Control:
    def __init__(self, data):
        self.conv = data["CONV"]
        self.calv = data["CALV"]


class Request:
    def __init__(self, data):
        self.et = data["ET"]  # Execution Type
        self.sr = data["SR"]  # Stats Required*
        self.fmt = data["FMT"]  # Formatted Required*
        self.rid = data["RID"]  # Request ID
        self.uid = data["UID"]  # User ID

#---------------------------------------------------------------------------------------------------------
# Inputs icnlude :
#       Asset Model
#       liability Model
#       Portofolio
#       Raw Data
#       Sensitivty_scenarios
#       Stats
#       LTCMA
#---------------------------------------------------------------------------------------------------------

class Inputs:

    def __init__(self, data):
        asset_model = []
        liablilty_model = []
        ltcma_model=[]

    @classmethod
    def create_items(cls, data):
        input = cls(data)
        if "Asset Model" in data:
            input.asset_model=AssetModel.create_items(data["Asset Model"])
        if  "Liability Model" in data:
            input.liablilty_model =LiabilityModel.create_items(data["Liability Model"])
        if "LTCMA" in data:
            input.ltcma_model = LTCMAModel.create_items(data["LTCMA"])
        return input
#---------------------------------------------------------------------------------------------------------
# Asset Modelinludes :
#       general_config
#       Assets
#       Assets Child
#       Corelation matrix
#       Adjustement Factor
#---------------------------------------------------------------------------------------------------------

class AssetModel:
    def __init__(self, data):
        general_config = []
        assets = []
        assets_child = []
        correlation_matrix = []
        adjustement_factor=[]

    @classmethod
    def create_items(cls, data):
        asset_model = cls(data)

        # general config
        asset_model.general_config = data["General config"]

        # assets
        asset_model.assets = pd.DataFrame.from_dict(data["Assets"]["Items"])
        asset_model.assets.index = asset_model.assets['MN']
        del asset_model.assets['MN']

        # assets  child
        asset_model.assets_child = (pd.DataFrame.from_dict(data["AssetChild"]["Items"]))

        # Correlation matrix
        asset_model.correlation_matrix = pd.DataFrame(data["Correlation Matrix"]["rows"],
                                                      columns=data["Correlation Matrix"]["columns name"],
                                                      index=data["Correlation Matrix"]["columns name"])
        # Adjustment Factor
        asset_model.adjustement_factor= (pd.DataFrame.from_dict(data["Adjustment Factor"]["Items"])).set_index(['MN', 'FY'])

        return asset_model

#---------------------------------------------------------------------------------------------------------
# LiabilityModel includes :
#       general_config
#       plans
#       stochastic_AA_details
#       historicalPerformance
#---------------------------------------------------------------------------------------------------------


class LiabilityModel:
    def __init__(self, data):
        general_config = []
        plans= []
        stochastic_AA_details=[]
        historical_performance= []


    @classmethod
    def create_items(cls, data):
        liabilityModel = cls(data)

        #general config
        liabilityModel.general_config = data["General config"]
        json_plans = data["Plans"]

        #Plans
        pp = []
        for item in json_plans["items"]:
            plan = {}
            #get plan general config
            plan["general_config"] =item["General config"].copy()
            list6=[]
            for  item2 in  item["items"]:
                 detail={}
                 detail["name"]=item2["name"]
                 detail["data"] = pd.DataFrame(item2["data"]["rows"], columns=item2["data"]["columns name"])
                 list6.append(detail)
            plan["items"] = list6
            pp.append(plan)
        liabilityModel.plans =pp

        #stochastic_AA_details
        liabilityModel.stochastic_AA_details = Stochastic_AA_Details.create_items(data["Stochastic AA details"])

        #historical Performance
        liabilityModel.historical_performance = pd.DataFrame.from_dict(data["Historical Performance"]["items"])
        liabilityModel.historical_performance.index = liabilityModel.historical_performance['KEY']
        del liabilityModel.historical_performance['KEY']

        return liabilityModel


class Stochastic_AA_Details:
    def __init__(self, data):
        general_config = []
        private_market_parameter = []
        overUnder = []


    @classmethod
    def create_items(cls, data):
        stochastic_AA_Details= cls(data)

        # general config
        stochastic_AA_Details.general_config = data["General config"]

        # private_market parameter
        stochastic_AA_Details.private_market_parameter = pd.DataFrame.from_dict(data["Private Market Parameter"]["items"])
        stochastic_AA_Details.private_market_parameter.index = stochastic_AA_Details.private_market_parameter['MN']
        del stochastic_AA_Details.private_market_parameter['MN']


        #OverUnder
        stochastic_AA_Details.overUnder= pd.DataFrame.from_dict(
            data["OverUnder"]["items"])
        stochastic_AA_Details.overUnder.index = stochastic_AA_Details.overUnder['MN']
        del stochastic_AA_Details.overUnder['MN']


        return  stochastic_AA_Details



#---------------------------------------------------------------------------------------------------------
# LTCMA inludes :
#       general_config
#       Bond return YC
#       Moments
#       Correlation matrix
#---------------------------------------------------------------------------------------------------------

class LTCMAModel:
    def __init__(self, data):
        general_config = []
        bonds_return = []
        moments =[]
        correlation_matrix = []


    @classmethod
    def create_items(cls, data):
        ltcma_model = cls(data)

        # general config
        ltcma_model.general_config = data["General config"]

        # bonds_return
        ltcma_model.bonds_return=pd.DataFrame(data["Bond Return YC"]["rows"],columns=data["Bond Return YC"]["columns name"])
        # ltcma_model.bonds_return=.set_index(['CTRY', 'YCT'])

        # moments
        xx = pd.DataFrame(data["Moments"]["rows"], columns=data["Moments"]["columns name"])
        ltcma_model.moments  = xx.set_index(['MN'])


        #Correlation matrix
        ltcma_model.correlation_matrix = pd.DataFrame(data["Correlation Matrix"]["rows"], columns=data["Correlation Matrix"]["columns name"]
                                                            ,index=data["Correlation Matrix"]["columns name"])



        return ltcma_model


