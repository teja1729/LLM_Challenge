# import os
# import sys
import pandas as pd
import openpyxl
import math
# import ipdb
import pytz
import itertools 
import numpy as np

timezone = pytz.timezone('Asia/Kolkata')
# import gurobipy as gp
# from datetime import date
# from django.conf import settings
from datetime import date,datetime, timedelta
# from apps.utils import setup_gurobi_license, upload_to_s3
from pyomo.environ import (
    SolverFactory, ConcreteModel,
    Set, Var, Param,NonNegativeIntegers,
    Objective, Constraint,
    NonNegativeReals, Binary,
    maximize, minimize,ConstraintList
)

def secondary(input_file='/app/media/data/modify_EAL 410_Input.xlsx', depots=['D253'], mip_gap=0.1, maximize_prob=True, use_integer=False):
  
  # setup_gurobi_license()
  print('Preprocessing')
  
  start_time = (datetime.now(timezone).time().hour * 60) + datetime.now(timezone).time().minute
  
  def PlantVehicleStopRadius(plant,VehicleType):
    return 50

  
  N = 30

  #reading input file
  df = pd.read_excel(input_file, sheet_name=None)

  #reading all required input sheets
  data_df = df['clean_data'].loc[(df['clean_data']['Check'] == 'Y')].copy(deep=True)
  plant_master_df = df['Plant Master']
  plant_vehicle_master_df = df['Plant Vehicle Master']
  vehicle_contract_df = df['VehicleContract']
  incompatibility_df = df['CustomerVehicle Incompatibility']
  d2c_diatance_df = df['D2C distance']
  c2c_distance_df = df['C2C distance']
  material_master_df = df['Material Master']
  loading_bay_df = df['Loading Bay Master']
  vehicle_master_df = df['Vehicle Master']
  uom_conversion_df = df['UOM Conversion']
  customer_master_df = df['Customer Master']

  #Creating all Plant_Zone_Customer_VehicleType_set
  unique_plant_zone_customer = data_df[['Plant', 'Zone','Customer']].drop_duplicates()
  PlantZoneCustomerVehicel_df = unique_plant_zone_customer.merge(vehicle_contract_df, on=['Plant', 'Zone'], how='inner')
  PlantZoneCustomerVehicel_df = PlantZoneCustomerVehicel_df.merge(incompatibility_df, how='left', indicator=True)
  PlantZoneCustomerVehicel_df = PlantZoneCustomerVehicel_df[PlantZoneCustomerVehicel_df['_merge'] != 'both'][['Plant', 'Zone', 'Customer', 'VehicleType']]
  
  #Creating sets for Route variable without bay label
  RouteWithoutBay_df1 = PlantZoneCustomerVehicel_df.copy(deep=True) #.loc[PlantZoneCustomerVehicel_df.index.repeat(get_vehicle_count())].copy(deep=True)
  RouteWithoutBay_df = pd.concat([RouteWithoutBay_df1] * N, ignore_index=True)
  RouteWithoutBay_df['Vehicle No.'] = np.repeat(range(1, N + 1), len(RouteWithoutBay_df1))
  RouteWithoutBay_df = RouteWithoutBay_df[['VehicleType','Vehicle No.','Plant','Customer']].drop_duplicates()
  
  #Creating sets for Plant vehicle
  PlantVehicles_df = RouteWithoutBay_df[['VehicleType','Plant']].copy(deep=True)
  PlantVehicles_df = PlantVehicles_df.groupby(['Plant']).apply(lambda x: x['VehicleType'].unique().tolist()).reset_index(name='vehicles')

  #Creating sets for PlantVehicle_customers
  PlantVehicle_customers_df = RouteWithoutBay_df.groupby(['VehicleType','Vehicle No.','Plant']).apply(lambda x: x['Customer'].unique().tolist()).reset_index(name='Customers')
  
  #Creating sets for Stops variable
  PlantZoneVehicleType_customers_df = PlantZoneCustomerVehicel_df.groupby(['Plant', 'Zone','VehicleType']).apply(lambda x: x['Customer'].unique().tolist()).reset_index(name='Customers').copy(deep=True)
  PlantZoneVehicleType_customers_index = PlantZoneVehicleType_customers_df.iloc[:, :].apply(tuple, axis=1)
  result = []
  for (p,z,v,customers) in PlantZoneVehicleType_customers_index:
    if len(customers)>1:
      for customer, from_customer, to_customer in list(itertools.product(customers, repeat=3)):
        if customer != to_customer:
          distance0 = c2c_distance_df.loc[(c2c_distance_df['From customer']==customer) & (c2c_distance_df['To customer']==to_customer),'Distance KMS'].iloc[0]
          if distance0 <=PlantVehicleStopRadius(p,v) and from_customer != to_customer:
            distance1 = c2c_distance_df.loc[(c2c_distance_df['From customer']==customer) & (c2c_distance_df['To customer']==from_customer),'Distance KMS'].iloc[0]
            if distance1 <= PlantVehicleStopRadius(p,v):
              result.append([v, p, z, customer, from_customer, to_customer])
  Stops_df1 = pd.DataFrame(result, columns=['VehicleType', 'Plant', 'Zone','Customer','From customer', 'To customer'])
  Stops_df = pd.concat([Stops_df1] * N, ignore_index=True)
  Stops_df['Vehicle No.'] = np.repeat(range(1, N + 1), len(Stops_df1))
  Stops_df = Stops_df[['VehicleType','Vehicle No.','Plant','Customer','From customer','To customer']].drop_duplicates()
  
  #Creating sets for Stopscy_ccx without orders
  StopsWithoutOrdercy_ccx_df = Stops_df.copy(deep=True)
  StopsWithoutOrdercy_ccx_df['Combined_ccx'] = StopsWithoutOrdercy_ccx_df.apply(lambda row: (row['Customer'], row['From customer']), axis=1).copy(deep=True)
  StopsWithoutOrdercy_ccx_df = StopsWithoutOrdercy_ccx_df.groupby(['VehicleType','Vehicle No.','Plant','To customer']).apply(lambda x: x['Combined_ccx'].unique().tolist()).reset_index(name='Combined_ccx')
  StopsWithoutOrdercy_ccx_df = StopsWithoutOrdercy_ccx_df[['VehicleType','Vehicle No.','Plant','To customer','Combined_ccx']]

  #Creating sets for stopscy_ccx with oders
  Stopscy_ccx_df = Stops_df.copy(deep=True)
  Stopscy_ccx_df['Combined_ccx'] = Stopscy_ccx_df.apply(lambda row: (row['Customer'], row['From customer']), axis=1).copy(deep=True)
  Stopscy_ccx_df = Stopscy_ccx_df.groupby(['VehicleType','Vehicle No.','Plant','To customer']).apply(lambda x: x['Combined_ccx'].unique().tolist()).reset_index(name='Combined_ccx')
  Stopscy_ccx_df = Stopscy_ccx_df.merge(data_df, left_on=['Plant','To customer'] , right_on = ['Plant','Customer'] , how = 'inner')
  Stopscy_ccx_df = Stopscy_ccx_df[['VehicleType','Vehicle No.','Plant','To customer','DeliveryNo','DeliveryItemNo','Material No','SalesUom','Combined_ccx']]
  
  #Creating sets for Stopsc_cxcy without orders
  Stopsc_cxcy_df = Stops_df.copy(deep=True)
  Stopsc_cxcy_df['Combined_ccx'] = Stopsc_cxcy_df.apply(lambda row: (row['From customer'], row['To customer']), axis=1).copy(deep=True)
  Stopsc_cxcy_df = Stopsc_cxcy_df.groupby(['VehicleType','Vehicle No.','Plant','Customer']).apply(lambda x: x['Combined_ccx'].unique().tolist()).reset_index(name='Combined_ccx')
  Stopsc_cxcy_df = Stopsc_cxcy_df[['VehicleType','Vehicle No.','Plant','Customer','Combined_ccx']]
  
  #Creating sets for Stops_FromCustomers
  Stops_FromCustomers_df = Stops_df.groupby(['VehicleType','Vehicle No.','Plant','Customer','To customer']).apply(lambda x: x['From customer'].unique().tolist()).reset_index(name='Customers').copy(deep=True)
  
  #Creating sets for Stops_ToCustomers
  Stops_ToCustomers_df = Stops_df.groupby(['VehicleType','Vehicle No.','Plant','Customer','From customer']).apply(lambda x: x['To customer'].unique().tolist()).reset_index(name='Customers').copy(deep=True)
  
  #Creating sets for Fill_Quantity variables:
  unique_all_orders = data_df[['DeliveryNo','DeliveryItemNo','Plant', 'Zone','Customer','Material No','SalesUom']].drop_duplicates()
  Fill_Quantity_df = unique_all_orders.merge(RouteWithoutBay_df, on=['Plant', 'Customer'], how='inner')
  Fill_Quantity_df = Fill_Quantity_df[['VehicleType','Vehicle No.','Plant','Customer','DeliveryNo','DeliveryItemNo','Material No','SalesUom']]

  #Creating sets for PZNV_COLMUom
  PZNV_COLMUom_df = Fill_Quantity_df.copy(deep=True)
  PZNV_COLMUom_df['colmuom'] = PZNV_COLMUom_df.apply(lambda row: (row['Customer'],row['DeliveryNo'],row['DeliveryItemNo'],row['Material No'],row['SalesUom']), axis=1)
  PZNV_COLMUom_df = PZNV_COLMUom_df.groupby(['VehicleType','Vehicle No.','Plant']).apply(lambda x: x['colmuom'].unique().tolist()).reset_index(name='colmuom')
  
  #Creating sets for Order_vehicles
  Order_vehicles_df = Fill_Quantity_df.copy(deep=True)
  Order_vehicles_df['Vehicle'] = Order_vehicles_df.apply(lambda row: (row['VehicleType'],row['Vehicle No.']), axis=1)
  Order_vehicles_df = Order_vehicles_df.groupby(['Plant','Customer','DeliveryNo','DeliveryItemNo','Material No','SalesUom']).apply(lambda x: x['Vehicle'].unique().tolist()).reset_index(name='Vehicles')
  Order_vehicles_df = Order_vehicles_df[['Plant','Customer','DeliveryNo','DeliveryItemNo','Material No','SalesUom','Vehicles']]
  
  #Creating index for Order_vehicles_df
  PCOLMUOM_index = Order_vehicles_df.iloc[:, :].apply(tuple, axis=1)
  
  #Creating sets for Vehicle_customers
  Vehicle_customers_df = Fill_Quantity_df.copy(deep=True)
  Vehicle_customers_df = Vehicle_customers_df.groupby(['Plant','VehicleType','Vehicle No.','DeliveryNo','DeliveryItemNo','Material No','SalesUom']).apply(lambda x: x['Customer'].unique().tolist()).reset_index(name='Customers')
  
  #Creating sets for PST variables
  PST_df = RouteWithoutBay_df[['VehicleType','Vehicle No.','Plant']].drop_duplicates().copy(deep=True)
  PST_df = PST_df.merge(loading_bay_df, on='Plant', how='left')

  #Creating sets for SQ variables
  PlantVehiclesWithNo_df = RouteWithoutBay_df.copy(deep=True)
  PlantVehiclesWithNo_df['vehicles'] = PlantVehiclesWithNo_df.apply(lambda x: (x['VehicleType'],x['Vehicle No.']), axis=1)
  PlantVehiclesWithNo_df = PlantVehiclesWithNo_df.groupby(['Plant']).apply(lambda x: x['vehicles'].unique().tolist()).reset_index(name='vehicles')
  merged_df = PlantVehiclesWithNo_df.merge(loading_bay_df, on='Plant', how='left')
  result = []
  for _,row in merged_df.iterrows():
    p,b,vehicles = row['Plant'],row['BayNo'],row['vehicles']
    vehicles.append((p,p))
    for (v,n),(v1,n1) in list(itertools.permutations(vehicles,2)):
      if v!=v1 or n!=n1:
        result.append((v,n,v1,n1,p,b))
  SQ_df = pd.DataFrame(result, columns=['Vehicle_x','VehicleNo_x','Vehicle_y','VehicleNo_y','Plant', 'BayNo']).drop_duplicates()

  #Creating sets for SQ_v1n1
  SQ_v1n1_df = SQ_df.copy(deep=True)
  SQ_v1n1_df['vehicle'] = SQ_v1n1_df.apply(lambda row: (row['Vehicle_x'],row['VehicleNo_x']),axis=1)
  SQ_v1n1_df = SQ_v1n1_df.groupby(['Vehicle_y','VehicleNo_y','Plant', 'BayNo']).apply(lambda x: x['vehicle'].unique().tolist()).reset_index(name='Vehicle')

  #Creating sets for SQ_vnp
  SQ_vnp_df = SQ_df.copy(deep=True)
  SQ_vnp_df['vehiclebay'] = SQ_vnp_df.apply(lambda row: (row['Vehicle_x'],row['VehicleNo_x'],row['BayNo']),axis=1)
  SQ_vnp_df = SQ_vnp_df.groupby(['Vehicle_y','VehicleNo_y','Plant']).apply(lambda x: x['vehiclebay'].unique().tolist()).reset_index(name='vehiclebay')
  
  #Creating sets for SQ_pb_vehicle_df
  SQ_pb_vehicle_df = SQ_df.copy(deep=True)
  SQ_pb_vehicle_df['vehicle'] = SQ_pb_vehicle_df.apply(lambda row: (row['Vehicle_x'],row['VehicleNo_x']),axis=1)
  SQ_pb_vehicle_df = SQ_pb_vehicle_df.groupby(['Plant','BayNo']).apply(lambda x: x['vehicle'].unique().tolist()).reset_index(name='vehicle')
  
  #Plant
  PlantBay = {}
  PlantCustomer = {}
  # PlantOpeningTime = {}
  # PlantClosingTime = {}
  # PlantQueueTime = {}

  #PlantVehicle
  PlantVehicleWeightKG = {}
  PlantVehicleVolumeCCM = {}
  PlantVehicleLoadingTime = {}
  PlantVehicleUnloadingTime = {}
  PlantVehicleMaxStop = {}
  # PlantVehiclePlantVehicleStopRadius = {}
  # PlantVehicleMinWeightLoad = {}
  # PlantVehicleMaxWeightLoad = {}
  # PlantVehicleMaxVolumeLoad = {} 
  # PlantMaxVehicles = {} 

  for _,row in PlantVehicles_df.iterrows():
    print(p)
    p,vehicles = row['Plant'],row['vehicles']
    PlantBay[p] = list(set(loading_bay_df[df['Loading Bay Master']['Plant']==p]['BayNo']))
    PlantCustomer[p] = list(set(data_df[data_df['Plant']==p]['Customer']))
    # PlantOpeningTime[p] = plant_master_df.loc[(plant_master_df['Plant']==p),'OpeningTime'].iloc[0]
    # PlantClosingTime[p] = plant_master_df.loc[(plant_master_df['Plant']==p),'ClosingTime'].iloc[0]
    # PlantQueueTime[p] = plant_master_df.loc[(plant_master_df['Plant']==p),'OueueTime'].iloc[0]
    for v in vehicles:
      PlantVehicleWeightKG[(p,v)] = vehicle_master_df.loc[(vehicle_master_df['Plant']==p) & (vehicle_master_df['VehicleType']==v),'VehiclePayload(KG)'].iloc[0]
      PlantVehicleVolumeCCM[(p,v)] = vehicle_master_df.loc[(vehicle_master_df['Plant']==p) & (vehicle_master_df['VehicleType']==v),'Volume CCM'].iloc[0]
      PlantVehicleLoadingTime[(p,v)] = plant_vehicle_master_df.loc[(plant_vehicle_master_df['Plant']==p) & (plant_vehicle_master_df['VehicleType']==v),'LoadingTime'].iloc[0]*60
      PlantVehicleUnloadingTime[(p,v)] = plant_vehicle_master_df.loc[(plant_vehicle_master_df['Plant']==p) & (plant_vehicle_master_df['VehicleType']==v),'UnloadingTime'].iloc[0]*60
      PlantVehicleMaxStop[(p,v)] = plant_vehicle_master_df.loc[(plant_vehicle_master_df['Plant']==p) & (plant_vehicle_master_df['VehicleType']==v),'MaxStops'].iloc[0]
      # PlantVehiclePlantVehicleStopRadius[(p,v)] = vehicle_master_df.loc[(vehicle_master_df['Plant']==p) & (vehicle_master_df['VehicleType']==v),'PlantVehicleStopRadius'].iloc[0]
      # PlantVehicleMinWeightLoad[(p,v)] = vehicle_master_df.loc[(vehicle_master_df['Plant']==p) & (vehicle_master_df['VehicleType']==v),'MinWeightLoad'].iloc[0]
      # PlantVehicleMaxWeightLoad[(p,v)] = vehicle_master_df.loc[(vehicle_master_df['Plant']==p) & (vehicle_master_df['VehicleType']==v),'MaxWeightLoad'].iloc[0]
      # PlantVehicleMaxVolumeLoad[(p,v)] = vehicle_master_df.loc[(vehicle_master_df['Plant']==p) & (vehicle_master_df['VehicleType']==v),'MaxVolumeLoad'].iloc[0]
      # PlantMaxVehicles[(p,v)] = plant_vehicle_master_df.loc[(plant_vehicle_master_df['Plant']==p) & (plant_vehicle_master_df['VehicleType']==v),'MaxVehicles'].iloc[0]
  
  def min_time(m):
    time_delta = timedelta(minutes=m)
    reference_time = datetime.strptime("12:00 AM", "%I:%M %p")
    converted_time = reference_time + time_delta
    formatted_time = converted_time.strftime("%I:%M %p")
    return formatted_time

  def PlantUomConversionFactorToCase(p,m,uom):
    factor1 = uom_conversion_df.loc[(uom_conversion_df['Product ID']==m) & (uom_conversion_df['Target UoM']=='C/S'),'UOM Conversion Factor'].iloc[0]
    factor2 = uom_conversion_df.loc[(uom_conversion_df['Product ID']==m) & (uom_conversion_df['Target UoM']==uom),'UOM Conversion Factor'].iloc[0]
    return factor1/factor2

  def CustomerAllOrders(c):
    index = data_df[data_df['Customer']==c][['DeliveryNo', 'DeliveryItemNo', 'Material No', 'SalesUom']].iloc[:, :].apply(tuple, axis=1)
    return index

  def PlantMaterialWeightKG(p,m):
    weight = material_master_df.loc[material_master_df['SKU']==m,'Gross Weight (Kg)'].iloc[0]
    return weight

  def PlantMaterialVolumeCCM(p,m):
    volume = material_master_df.loc[material_master_df['SKU']==m,'Volume CCM'].iloc[0]
    return volume

  def PlantRoutesVehicle_customer(p,v):
    customer = list(set(RouteWithoutBay_df[(RouteWithoutBay_df['Plant']==p) & (RouteWithoutBay_df['VehicleType']==v)]['Customer']))
    return customer

  def PlantOpeningTime(p):
    return 570

  def PlantClosingTime(p):
    return 1140

  def PlantQueueTime(p):
    return 15

  def PlantCustomerOpeningTime(p,c):
    return 600

  def PlantCustomerClosingTime(p,c):
    return 1140

  def PlantCustomerQueueTime(p,c):
    return 15
  
  def PlantVehicleSpeed(p,v):
    return 40/60

  def PlantMaxVehicles(p,v):
    return 10
  
  def PlantMaterialUomCustomerNDP(p,m,uom,c):
    return 100000

  def PlantCustomerStopCharge(p,v,c):
    return 100

  def MaxSplit(p,c):
    return 2

  def MaxStop(p,v):
    return 2

  def PlantVehicleStopRadius(p,v):
    return 50

  def PlanningRunTime():
    return 30

  def CurrentTime():
    current_time = datetime.now(timezone).time()
    minutes = (current_time.hour * 60) + current_time.minute
    return 0

  def PlantToCustomerDistance(p,c):
    distance = d2c_diatance_df.loc[(d2c_diatance_df['Plant']==p) & (d2c_diatance_df['Customer']==c), 'Distance KMS'].iloc[0]
    return distance

  def CustomerToCustomerDistance(c_x,c_y):
    distance = c2c_distance_df.loc[(c2c_distance_df['From customer']==c_x) & (c2c_distance_df['To customer']==c_y),'Distance KMS'].iloc[0]
    return distance

  def customers_ccx(v,n,p,c):
    try:
      List = StopsWithoutOrdercy_ccx_df.loc[(StopsWithoutOrdercy_ccx_df['VehicleType']==v) & (StopsWithoutOrdercy_ccx_df['Vehicle No.']==n) & (Stops_FromCustomers_df['Plant']==p) & (StopsWithoutOrdercy_ccx_df['To customer']==c),'Combined_ccx'].iloc[0]
    except IndexError:
      List = []
    return List

  # def customers_ccx

  #defining function for calculating total cost
  def New_Objective_function():
    TotalTime = sum(model.PST[i] for i in PST_index) + sum(model.CST[i] for i in Routes_index)
    TotalFreightRatePerTripBasis = 0
    TotalFreightRatePerKGBasis = 0
    TotalFreightRateperCaseBasis = 0
    DeadFreightPerKGBasis = 0
    NonDeliveryPenalty = 0
    for (p,z,v,customers) in PlantZoneVehicleType_customers_index:
      ContractedRateUOM = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate UOM'].iloc[0]
      ContractedRate = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate'].iloc[0]
      for n in range(1,N+1):
        if ContractedRateUOM == 'RatePerTrip':
          TotalFreightRatePerTripBasis += sum(model.Route[v,n,p,c] for c in customers)*ContractedRate
          DeadFreightPerKGBasis += (sum(model.Route[v,n,p,c] for c in customers)*PlantVehicleWeightKG[(p,v)] - sum(model.FQ[v,n,p,c,o,l,m,uom]*PlantUomConversionFactorToCase(p,m,uom)*PlantMaterialWeightKG(p,m) for c in customers for (o,l,m,uom) in CustomerAllOrders(c)))*(ContractedRate/PlantVehicleWeightKG[(p,v)])
        elif ContractedRateUOM == 'Rate PMT':
          TotalFreightRatePerKGBasis += sum(model.Route[v,n,p,c]*PlantVehicleWeightKG[(p,v)] for c in customers)*ContractedRate/1000
          DeadFreightPerKGBasis += (sum(model.Route[v,n,p,c] for c in customers)*PlantVehicleWeightKG[(p,v)] - sum(model.FQ[v,n,p,c,o,l,m,uom]*PlantUomConversionFactorToCase(p,m,uom)*PlantMaterialWeightKG(p,m) for c in customers for (o,l,m,uom) in CustomerAllOrders(c)))*(ContractedRate/1000)
        elif ContractedRateUOM == 'Rate per Case':
          TotalFreightRateperCaseBasis += sum(model.FQ[v,n,p,c,o,l,m,uom]*PlantUomConversionFactorToCase(p,m,uom) for c in customers for (o,l,m,uom) in CustomerAllOrders(c))*ContractedRate
    for (p,c,o,l,m,uom,vehicles) in PCOLMUOM_index:  
      NonDeliveryPenalty += (data_df.loc[(data_df['DeliveryNo']==o) & (data_df['DeliveryItemNo']==l),'SQ'].iloc[0] - sum(model.FQ[v,n,p,c,o,l,m,uom] for (v,n) in vehicles))*1000000000
    return TotalFreightRatePerTripBasis + DeadFreightPerKGBasis + TotalFreightRatePerKGBasis + TotalFreightRateperCaseBasis + NonDeliveryPenalty + TotalTime/1000

  print('Building Model ---------------------------------->\n')

  model = ConcreteModel()

  model.P = Set(initialize = list(set(data_df['Plant'])))

  zones = {}
  for p in model.P:
    for c in PlantCustomer[p]:
      zones[c]=data_df[data_df['Customer']== c]['Zone'].values[0]

  # Creating Route variables
  Routes_index = RouteWithoutBay_df.iloc[:, :].apply(tuple, axis=1)
  model.Route = Var(Routes_index, within=Binary)
  
  #Creating Stop variables
  Stops_index = Stops_df.iloc[:,:].apply(tuple,axis=1)
  model.Stop = Var(Stops_index, within=Binary)
  
  #Creating Filling variables
  Filling_index = Fill_Quantity_df.iloc[:, :].apply(tuple, axis=1)
  model.FQ = Var(Filling_index, within=NonNegativeIntegers)
  
  #Creating subtour control variables
  model.U1 = Var(Routes_index)
  
  #Creating Start Loading time variables
  PST_index = PST_df.iloc[:, :].apply(tuple, axis=1)
  model.PST = Var(PST_index, within=NonNegativeReals,bounds=(0, 1260))
  
  #Creating Start Unloading time variables
  model.CST = Var(Routes_index, within=NonNegativeReals, bounds=(0, 1260))
  
  #Creating Squenceing variables for vehicles
  SQ_index = SQ_df.iloc[:, :].apply(tuple, axis=1)
  model.SQ = Var(SQ_index, within=Binary)
  
  #Creating subtour control variables
  model.U = Var(PST_index)

  #Creating Objective function
  model.Objective = Objective(expr = (New_Objective_function() + sum(model.Stop[i]*100 for i in Stops_index)))

  #Creating route and stop balancing constraints
  #one vehicle assign to one route
  model.route_rule1_constraint = ConstraintList()
  PlantVehicle_customers_index = PlantVehicle_customers_df.iloc[:, :].apply(tuple, axis=1)
  for (v,n,p,customers) in PlantVehicle_customers_index:
    model.route_rule1_constraint.add(sum(model.Route[v,n,p,c] for c in customers) <= 1)

  #stop only for allocated vehicle for given route
  model.stopping_rule1_constraint = ConstraintList()
  for (v,n,p,c,c_x,c_y) in Stops_index:
    model.stopping_rule1_constraint.add(expr = model.Stop[v,n,p,c,c_x,c_y] <= (model.Route[v,n,p,c]))
    
  model.stopping_rule2_constraint = ConstraintList()
  for (v,n,p,c,c_x,c_y) in Stops_index:
    if c!=c_x:
      fromCustomers = Stops_FromCustomers_df.loc[(Stops_FromCustomers_df['VehicleType']==v) & (Stops_FromCustomers_df['Plant']==p) & (Stops_FromCustomers_df['Customer']==c) & (Stops_FromCustomers_df['To customer']==c_x),'Customers'].iloc[0]
      model.stopping_rule2_constraint.add(expr = model.Stop[v,n,p,c,c_x,c_y] <= sum(model.Stop[v,n,p,c,c_z,c_x] for c_z in fromCustomers))
    
  #a vehicle arrive only onces to the customer
  model.stopping_rule3_cosnstraint = ConstraintList()
  Stops_ToCustomers_index = Stops_ToCustomers_df.iloc[:, :].apply(tuple, axis=1)
  for (v,n,p,c,c_x,customers) in Stops_ToCustomers_index:
    model.stopping_rule3_cosnstraint.add(sum(model.Stop[v,n,p,c,c_x,c_y] for c_y in customers) <=1 )

  #a vehicle departure only onces to the customer
  model.stopping_rule4_cosnstraint = ConstraintList() 
  Stops_ToCustomers_index = Stops_FromCustomers_df.iloc[:, :].apply(tuple, axis=1)
  for (v,n,p,c,c_y,customers) in Stops_ToCustomers_index:
    model.stopping_rule3_cosnstraint.add(sum(model.Stop[v,n,p,c,c_x,c_y] for c_x in customers) <=1 )

  # #subtour removing constraint
  # model.stopping_rule5_cosnstraint = ConstraintList()
  # for (v,n,p,c,c_x,c_y) in Stops_index:
  #   model.stopping_rule5_cosnstraint.add(expr = model.U1[v,n,p,c_y] - model.U1[v,n,p,c_x] >=  10 - (1 - model.Stop[v,n,p,c,c_x,c_y])*(100000))

  #Creating problam validation constraints
  #total loaded quantity should be less than order quantity
  model.total_quantity_constraint = ConstraintList()
  for (p,c,o,l,m,uom,vehicles) in PCOLMUOM_index:
    model.total_quantity_constraint.add(sum(model.FQ[v,n,p,c,o,l,m,uom] for (v,n) in vehicles) <= data_df.loc[(data_df['DeliveryNo']==o) & (data_df['DeliveryItemNo']==l),'SQ'].iloc[0])

  #filling only if vehicle allocated to the customer
  model.wight_load_vehicle_constraints1 = ConstraintList()
  for (v,n,p,c,o,l,m,uom) in Filling_index:
    customers = customers_ccx(v,n,p,c)
    if len(customers)>0:
      model.wight_load_vehicle_constraints1.add(expr = model.FQ[v,n,p,c,o,l,m,uom] <= (data_df.loc[(data_df['DeliveryNo']==o) & (data_df['DeliveryItemNo']==l),'SQ'].iloc[0])*(model.Route[v,n,p,c]+sum(model.Stop[v,n,p,c_x,c_y,c] for (c_x,c_y) in customers)))
    else:
      model.wight_load_vehicle_constraints1.add(expr = model.FQ[v,n,p,c,o,l,m,uom] <= (data_df.loc[(data_df['DeliveryNo']==o) & (data_df['DeliveryItemNo']==l),'SQ'].iloc[0])*(model.Route[v,n,p,c]))

  # #filling only if vehicle allocated to the customer
  # model.wight_load_vehicle_constraints2 = ConstraintList()
  # for (v,n,p,c,o,l,m,uom,customers) in StopsCy_CCx_index:
  #   model.wight_load_vehicle_constraints2.add(expr = model.FQ[v,n,p,c,o,l,m,uom] >= (model.Route[v,n,p,c]+sum(model.Stop[v,n,p,c_x,c_y,c] for (c_x,c_y) in customers)))

  # Maximum Weight Load Ability Constraint
  model.max_weight_load_ability_constraint = ConstraintList()
  PZNV_COLMUom_index = PZNV_COLMUom_df.iloc[:, :].apply(tuple, axis=1)
  for (v,n,p,colmuom) in PZNV_COLMUom_index:
    model.max_weight_load_ability_constraint.add(expr = sum(model.FQ[v,n,p,c,o,l,m,uom]*PlantUomConversionFactorToCase(p,m,uom)*PlantMaterialWeightKG(p,m) for (c,o,l,m,uom) in colmuom) <= PlantVehicleWeightKG[(p,v)]) #* model.MWL[p, v]

  # Maximum Volume Load Ability Constraint
  model.max_valume_load_ability_constraint = ConstraintList()
  for (v,n,p,colmuom) in PZNV_COLMUom_index:
    model.max_valume_load_ability_constraint.add(expr = sum(model.FQ[v,n,p,c,o,l,m,uom]*PlantUomConversionFactorToCase(p,m,uom)*PlantMaterialVolumeCCM(p,m) for (c,o,l,m,uom) in colmuom) <= PlantVehicleVolumeCCM[(p,v)])#* model.MVL[p, v] 

  #Max No. of Stops constraint for a vehicle type
  model.max_no_of_stops_constraint = ConstraintList()
  StopsC_CxCy_index = Stopsc_cxcy_df.iloc[:, :].apply(tuple, axis=1)
  for (v,n,p,c,customers) in StopsC_CxCy_index:
    model.max_no_of_stops_constraint.add(expr = sum(model.Stop[v,n,p,c,c_x,c_y] for (c_x,c_y) in customers) <= PlantVehicleMaxStop[(p,v)])

  #Max number of splits 
  # def max_splits_rule():
  #   for _,row in StopsCy_CCx.iterrows():
  #     v,n,p,c,o,l,m,uom,customers = row['VehicleType'],row['Vehicle No.'],row['Plant'],row['Customer'],row['DeliveryNo'],row['DeliveryItemNo'],row['Material No'],row['SalesUom'],row['Combined_ccx']
  #      model.max_splits_constraint.add(sum(model.Route[v, n, p, c]+sum(model.Stop[v,n,p,c_x,c_y,c] for (c_x,c_y) in customers_ccx(c)) for v in vehicles for n in range(PlantMaxVehicles(p,v))) <= MaxSplit(p,c))
  # model.max_splits_constraint = ConstraintList()
  # max_splits_rule()


  # def max_splits_rule():
  #   for (c,o,l,m,uom,vehicles) in PCOLMUOM_index:
  #     model.max_splits_constraint.add(sum(model.FQ[v,n,p,c,o,l,uom]))
  # model.max_splits_constraint = ConstraintList()
  # max_splits_rule()
  
  
  #####################################################################################
  
  model.scheduling1_constraint = ConstraintList()
  SQ_v1n1_index = SQ_v1n1_df.iloc[:, :].apply(tuple, axis=1)
  for (v1,n1,p,b,vehicles) in SQ_v1n1_index:
    model.scheduling1_constraint.add(expr = sum(model.SQ[v,n,v1,n1,p,b] for (v,n) in vehicles) == sum(model.SQ[v1,n1,v,n,p,b] for (v,n) in vehicles))

  model.scheduling2_constraint = ConstraintList()
  SQ_vnp_index = SQ_vnp_df.iloc[:, :].apply(tuple, axis=1)
  for (v,n,p,vehiclebay) in SQ_vnp_index:
    if v!=n:
      model.scheduling2_constraint.add(expr = sum(model.SQ[v1,n1,v,n,p,b] for (v1,n1,b) in vehiclebay) == sum(model.Route[v,n,p,c] for c in PlantRoutesVehicle_customer(p,v)))
  
  model.scheduling3_constraint = ConstraintList()
  SQ_pb_vehicle_index = SQ_pb_vehicle_df.iloc[:, :].apply(tuple, axis=1)
  for (p,b,vehicles) in SQ_pb_vehicle_index:
    model.scheduling3_constraint.add(expr = sum(model.SQ[p,p,v,n,p,b] for (v,n) in vehicles if v!=n) <= 1)

  model.scheduling4_constraint = ConstraintList()
  for (v,n,v1,n1,p,b) in SQ_index:
    if v!=n and v1!=n1:
      model.scheduling4_constraint.add(expr = model.U[v1,n1,p,b] - model.U[v,n,p,b] >=  10 - (1- model.SQ[v,n,v1,n1,p,b])*(200000))

  # start time of vehicle constraint
  model.start_time_vehicle_constraint = ConstraintList()
  for (v,n,p,b,vehicles) in SQ_v1n1_index:
    if v!=p:
     model.start_time_vehicle_constraint.add(expr= model.PST[v,n,p,b] >= (PlanningRunTime()+CurrentTime()+PlantQueueTime(p))*(sum(model.SQ[v1,n1,v,n,p,b] for (v1,n1) in vehicles)))

  # start_time of vehicle constraint1
  model.start_time_vehicle_constraint1 = ConstraintList()
  for (v,n,p,b,vehicles) in SQ_v1n1_index:
    if v!=p:
      model.start_time_vehicle_constraint1.add(expr = model.PST[v,n,p,b] >= (PlantOpeningTime(p) + PlantQueueTime(p))*(sum(model.SQ[v1,n1,v,n,p,b] for (v1,n1) in vehicles)))

  # start time plus load time less than plant close time
  model.start_time_plus_load_time_less_than_plant_close_time = ConstraintList()
  for (v,n,p,b,vehicles) in SQ_v1n1_index:
    if v!=p:
      model.start_time_plus_load_time_less_than_plant_close_time.add(expr = model.PST[v,n,p,b] <= (PlantClosingTime(p)-PlantVehicleLoadingTime[(p,v)])*(sum(model.SQ[v1,n1,v,n,p,b] for (v1,n1) in vehicles)))

  # start time vehicle greater than start time of previous vehicle
  model.start_time_vehicle_greater_than_start_time_of_previous_vehicle_constraint = ConstraintList()
  for (v,n,v1,n1,p,b) in SQ_index:
    if v!=p and v1!=p:
      model.start_time_vehicle_greater_than_start_time_of_previous_vehicle_constraint.add(expr = model.PST[v1,n1,p,b] >= model.PST[v,n,p,b] + PlantVehicleLoadingTime[(p,v)] - 100000*(1-model.SQ[v,n,v1,n1,p,b]))
  
  model.start_time_unloading_first_customer_constraint = ConstraintList()
  for (v,n,p,c) in Routes_index:
    model.start_time_unloading_first_customer_constraint.add(expr = model.CST[v,n,p,c] >= sum(model.PST[v,n,p,b] for b in PlantBay[p]) + PlantVehicleLoadingTime[(p,v)] + PlantToCustomerDistance(p,c)/PlantVehicleSpeed(p,v) + PlantCustomerQueueTime(p,c) - 100000*(1 - model.Route[v,n,p,c]))

  model.next_customer_after_unloading_at_the_previous_customer_constraint = ConstraintList()
  for (v,n,p,c,customers_cxcy) in StopsC_CxCy_index:
    for (c_x,c_y) in customers_cxcy:
      model.next_customer_after_unloading_at_the_previous_customer_constraint.add(expr = model.CST[v,n,p,c_y] >= model.CST[v,n,p,c_x] + PlantVehicleUnloadingTime[(p,v)] + PlantCustomerQueueTime(p,c_y) + CustomerToCustomerDistance(c_x,c_y)/PlantVehicleSpeed(p,v) - 100000*(1 - model.Stop[v,n,p,c,c_x,c_y]))
        
  # start_time_unloading_every_customer
  model.start_time_unloading_every_customer_constraint = ConstraintList()
  for (v,n,p,c) in Routes_index:
    model.start_time_unloading_every_customer_constraint.add(expr = model.CST[v,n,p,c] >= (PlantCustomerOpeningTime(p,c) + PlantCustomerQueueTime(p,c))*(model.Route[v,n,p,c]+sum(model.Stop[v,n,p,c_x,c_y,c] for (c_x,c_y) in customers_ccx(v,n,p,c)))) 
          
  #unloading_every_customer_site
  model.unloading_every_customer_site_constraint = ConstraintList()
  for (v,n,p,c) in Routes_index:
    model.unloading_every_customer_site_constraint.add(expr = model.CST[v,n,p,c] <= PlantCustomerClosingTime(p,c)*(model.Route[v,n,p,c]+sum(model.Stop[v,n,p,c_x,c_y,c] for (c_x,c_y) in customers_ccx(v,n,p,c))))
  
  print('Solving Model ---------------------------------->\n')
  
  model.Objective.sense = minimize
  opt = SolverFactory("gurobi", solver_io='python')
  result = opt.solve(model, load_solutions=True, tee=True)

  print('Solver Status:',result.solver.status)
  print('Solver termination condition: ',result.solver.termination_condition)
  print('\nObjective = ', model.Objective())

  print('Routes')
  for i in model.Route:
    if model.Route[i].value == 1:
      print(i) 

  print("Stops")
  for i in model.Stop:
    if model.Stop[i].value == 1:
      print(i)
  
  df7 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'DeliveryNo':[],'DeliveryItemNo':[],'Delivery Date':[],'Zone':[],'Customer':[],'Ship to party name':[],'Material':[],'Material Description':[],'Order Quantity':[],'UOM':[],'Fill Quantity':[],'Shortfall':[],'Remarks':[]})
  
  for (p,c,o,l,m,uom,vehicles) in PCOLMUOM_index:
    fill_qunatity = sum(model.FQ[v,n,p,c,o,l,m,uom].value for (v,n) in vehicles)
    data = data_df.loc[(data_df['DeliveryNo']==o) & (data_df['DeliveryItemNo']==l)].iloc[0]
    df7.loc[len(df7.index)] = [p,data['PlantName'],o,l,data['DeliveryDate'],data['Zone'],c,data['CustomerName'],m,data['MaterialName'],data['SQ'],uom,fill_qunatity,data['SQ']-fill_qunatity,'Unknown']
  
  df4 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'VehicleType':[],'Vehicle No.':[],'Trip Date':[],'Zone':[],'Customer':[],'Ship to party name':[],'Arrival Time':[],'Queue Time (mins)':[],'UnLoading Time (mins)':[],'Departure Time':[],'Delivery qty in Cases':[],'Delivery Quantity in Kg':[]})
  df2 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'VehicleType':[],'Vehicle No.':[],'DeliveryNo':[],'DeliveryItemNo':[],'Delivery Date':[],'Zone':[],'Customer':[],'Ship to party name':[],'Material':[],'Material Description':[],'Order Quantity':[],'UOM':[],'Allocated Quantity':[],'Allocated Quantity in cases':[],'Allocated Quantity in Kg':[]})
  
  vehicle_wise_map = {}
  print("Fill Qunatity") 
  for i in model.FQ:
    if model.FQ[i].value !=0:
      v,n,p,c,o,l,m,uom = i
      print(i," ",model.FQ[i].value)#,"  ",model.GW[i[4]].value)
      
      data = data_df.loc[(data_df['DeliveryNo']==o) & (data_df['DeliveryItemNo']==l)].iloc[0]
      AllocatedQuantity_in_cases = model.FQ[i].value*PlantUomConversionFactorToCase(p,m,uom)
      AllocatedQuantity_in_Kg = AllocatedQuantity_in_cases*PlantMaterialWeightKG(p,m)
      df2.loc[len(df2.index)] = [p,data['PlantName'],v,n,o,l,data['DeliveryDate'],data['Zone'],c,data['CustomerName'],m,data['MaterialName'],data['SQ'],uom,model.FQ[i].value,round(AllocatedQuantity_in_cases),AllocatedQuantity_in_Kg] 

      today =  date.today()
      TripDate = today.strftime("%d/%m/%Y")
      StartLoadingTime = min_time(sum(model.PST[v,n,p,b].value for b in PlantBay[p]))
      ArrivalTime = min_time(model.CST[v,n,p,c].value - PlantCustomerQueueTime(p,c))
      DepartureTime = min_time(model.CST[v,n,p,c].value + PlantVehicleUnloadingTime[(p,v)])
      UnloadedQuantityUOM = 0
      df4.loc[len(df4.index)] = [p,data['PlantName'],v, n, TripDate, data['Zone'],c,data['CustomerName'],ArrivalTime, PlantCustomerQueueTime(p,c), PlantVehicleUnloadingTime[(p,v)],DepartureTime,round(AllocatedQuantity_in_cases),AllocatedQuantity_in_Kg]

      fq = model.FQ[i].value*PlantUomConversionFactorToCase(p,m,uom) 
      wgt = PlantMaterialWeightKG(p,m)
      vol = PlantMaterialVolumeCCM(p,m)
      z = zones[c]
      
      ContractedRateUOM = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate UOM'].iloc[0]
      # ContractedRate = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate'].iloc[0]
      
      if ContractedRateUOM == 'Rate per Case':
        value = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate'].iloc[0]
      else:
        value = 0 
      
      key = (v,n,p,z,data['PlantName'])
      if key not in vehicle_wise_map:
        vehicle_wise_map[key] = {
          'weight': fq * wgt,
          'volume': fq * vol,
          # 'total_rate_per_case': fq * value
          'No_of_case': fq, 
        }
      else:
        vehicle_wise_map[key]['weight'] += fq * wgt
        vehicle_wise_map[key]['volume'] += fq * vol
        # vehicle_wise_map[key]['total_rate_per_case'] += fq * value
        vehicle_wise_map[key]['No_of_case'] += fq
  
  OverallTotalFreightCharges = 0
  OverallTotalStopCharges = 0
  OverallTotalDeadFreightcost = 0

  df3 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'VehicleType':[],'Vehicle No.':[],'VehiclePayload_KG':[],'VehicleVolume CCM':[],'Actual_Payload_KG':[],'Actual_Volume_CCM':[],'WeightLoadability %':[],'WeightLoadabilityLoss %':[],'VolumeLoadability %':[],'VolumeLoadabilityLoss %':[]})
  df8 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'Bay':[],'VehicleType':[],'Vehicle No.':[],'Trip Date':[],'Queue Time (mins)':[],'Loading Time (mins)':[],'Plant Loading Time':[],'Allocated Quantity in cases':[],'Allocated Quantity in Kg':[]})
  df5 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'VehicleType':[],'Vehicle No.':[],'VehiclePayload_KG':[],'Zone':[],'Contracted Rate':[],'Contracted Rate UOM':[],'Contracted Stop Charges':[],'No. of cases':[],'Gross weight':[],'No of stops':[],'Total Stop Charges':[],'Total Freight Charges':[],'Total Delivery Cost':[],'Dead Freight Cost':[],'WeightLoadability %':[],'WeightLoadabilityLoss %':[]})
  
  Plant_wise_map = {}
  for key, data in vehicle_wise_map.items():
    
    (v,n,p,z,PlantName) = key
    weight = round((data['weight'] / PlantVehicleWeightKG[(p,v)]) * 100, 3)
    volume = round((data['volume'] / PlantVehicleVolumeCCM[(p,v)]) * 100, 3)
   
    for b in PlantBay[p]:
      
      if model.PST[v,n,p,b].value >0:
        today =  date.today()
        TripDate = today.strftime("%d/%m/%Y")
        PlantLoadingTime = min_time(sum(model.PST[v,n,p,b].value for b in PlantBay[p]))
        df8.loc[len(df8.index)] = [p,PlantName,b,v,n,TripDate,PlantQueueTime(p),PlantVehicleLoadingTime[(p,v)],PlantLoadingTime,round(data['No_of_case']),data['weight']] 

    df3.loc[len(df3.index)] = [p,PlantName,v,n,PlantVehicleWeightKG[(p,v)],PlantVehicleVolumeCCM[(p,v)],data['weight'],data['volume'], weight, 100-weight, volume, 100-volume]

    print('Vehicle: {} | Weight: {}({}%) | Volume: {}({}%)'.format(key, 
      data['weight'], round(data['weight'] / PlantVehicleWeightKG[(p,v)] * 100, 3),
      data['volume'], round(data['volume'] / PlantVehicleVolumeCCM[(p,v)] * 100, 3)
    ))

    ContractedRateUOM = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate UOM'].iloc[0]
    ContractedRate = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Rate'].iloc[0]
    ContractedStopCharges = vehicle_contract_df.loc[(vehicle_contract_df['Plant']==p) & (vehicle_contract_df['Zone']==z) & (vehicle_contract_df['VehicleType']==v),'Contracted Stop Charges'].iloc[0]
    
    Noofstops = 0
    for i in model.Stop:
      if model.Stop[i].value==1:
        if i[0]==v and i[1]==n and i[2]==p:
          Noofstops += 1
    TotalStopCharges = ContractedStopCharges*Noofstops
    
    if ContractedRateUOM == 'RatePerTrip':
      TotalFreightCharges = ContractedRate
      TotalDeadFreightcost = (PlantVehicleWeightKG[(p,v)] - data['weight'])*ContractedRate/PlantVehicleWeightKG[(p,v)]
    
    elif ContractedRateUOM == 'Rate PMT':
      TotalFreightCharges = PlantVehicleWeightKG[(p,v)]*ContractedRate/1000
      TotalDeadFreightcost = (PlantVehicleWeightKG[(p,v)] - data['weight'])*ContractedRate/1000
    
    elif ContractedRateUOM == 'Rate per Case':
      TotalFreightCharges = data['No_of_case']*ContractedRate
      TotalDeadFreightcost = 0
    
    key = (p,PlantName)
    if key not in Plant_wise_map:
        Plant_wise_map[key] = {
          'freight_cost': TotalFreightCharges,
          'deadfreight_cost': TotalDeadFreightcost,
          'stop_charge': TotalStopCharges,
          'No_of_case':data['No_of_case'],
          'weight':data['weight'] 
        }
    else:
      Plant_wise_map[key]['freight_cost'] += TotalFreightCharges
      Plant_wise_map[key]['deadfreight_cost'] += TotalDeadFreightcost
      Plant_wise_map[key]['stop_charge'] += TotalStopCharges
      Plant_wise_map[key]['No_of_case'] += data['No_of_case']
      Plant_wise_map[key]['weight'] += data['weight']

    df5.loc[len(df5.index)] = [p,PlantName,v,n,PlantVehicleWeightKG[(p,v)],z,ContractedRate, ContractedRateUOM, ContractedStopCharges, round(data['No_of_case']), data['weight'], Noofstops, TotalStopCharges, TotalFreightCharges, TotalStopCharges+TotalFreightCharges, TotalDeadFreightcost,weight, 100-weight]

  print('SQ')
  for i in model.SQ:
    if model.SQ[i].value==1:
      print(i,model.SQ[i].value)

  print('PST')
  for i in model.PST:
    if model.PST[i].value >0:
      print(i,model.PST[i].value)

  print('CST')
  for i in model.CST:
    if model.CST[i].value >0:
      print(i,model.CST[i].value)

  # end_time = (datetime.now(timezone).time().hour * 60) + datetime.now(timezone).time().minute
  # runtime = end_time - start_time 

  df6 = pd.DataFrame({'Depot Code':[],'Depot Name':[],'Total Freight Cost':[],'Total Stop charges':[],'Total Delivery Cost':[],'Total Dead Freight cost':[],'Allocated Quantity in cases':[],'Allocated Quantity in Kg':[]})
  
  for key, data in Plant_wise_map.items():
    df6.loc[len(df6.index)] = [key[0],key[1],data['freight_cost'],data['stop_charge'],data['freight_cost']+data['stop_charge'],data['deadfreight_cost'],round(data['No_of_case']),round(data['weight'])]
  
  path = '/app/media/data/output_data/modify_EAL 410_OutputFile2.xlsx'

  with pd.ExcelWriter(path) as writer:
    # writer.book = openpyxl.load_workbook(path)
    df6.to_excel(writer,sheet_name='summary',index=False)
    df7.to_excel(writer,sheet_name='Fill Quantity',index=False)
    df2.to_excel(writer, sheet_name='Vehicle Allocation',index=False)
    df8.to_excel(writer, sheet_name='Vehicle Loading Details',index=False)
    df4.to_excel(writer, sheet_name='Vehicle Trip Details',index=False)
    df3.to_excel(writer,sheet_name='Loadability',index=False)
    df5.to_excel(writer,sheet_name='Delivery Cost',index=False)
  
if __name__ == '__main__':
  secondary()

