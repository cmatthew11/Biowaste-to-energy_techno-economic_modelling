# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:58:09 2022

@author: Chris

Main Inputs

1. Type of facility
    a. AD + biogas for heat
    b. AD + biogas for electricity
    c. AD + CHP
2. Size of facility
    a. m3/tonnes - pretreatement and AD
    b. kW - biogas boiler
    c. kW - CHP 
3. Costs
    a. CAPEX/OPEX of facility
    b. Transportation costs (results from CVRP solver)
    c. networking (elec/district heat)
    
Taking the above outputs, calculates the levelised cost of energy for each 
scenario configuration of tehcnology, catchment area and collection costs
"""

import pandas as pd
import numpy as np
from ast import literal_eval
from copy import deepcopy

#%% 

density = {'food_processing':1000, 
           'fish_farm':1000, 
           'land_farm':1000, 
           'distillery':1000,
           'brewery':500,
           'food_waste':500
           }

# Avoided costs in £/tonne, costs are positive, savings are negative
avoided_costs = {'fish_processing_demersal':-1.6,#https://www.scottishwater.co.uk/your-home/your-charges/your-charges-2019-2020/metered-charges-2019-2020
                 'fish_processing_pelagic':-1.6,
                 'dairy_processing_dairy':-1.6,
                 'meat_processing_meat':-1.6,
                 'shellfish_processing_shellfish':-1.6, 
                 'fish_farm_pelagic':-36.5,#https://www.gov.scot/binaries/content/documents/govscot/publications/research-and-analysis/2016/05/zero-waste-report-finfish-mortalities-in-scotland/documents/finfish-mortalities-in-scotland-summary-report/finfish-mortalities-in-scotland-summary-report/govscot%3Adocument/finfish%2Bmorrtalities%2Bsummary.pdf
                 'farm_cow':-200,#https://www.thefarmernetwork.co.uk/wp-content/uploads/2020/10/Leo-Group-Robinson-Mitchell-Price-List-Final.pdf
                'farm_sheep':-155,
                 'distillery_draff':0,#24.18,  probably from pau
                 'distillery_pot_ale':0,#4.4,probably from pau
                'distillery_spent_lees':-1.6, # assume waste water
                'brewery_spent_grain':0,
                'brewery_spent_hops':0, 
                'brewery_spent_yeast':0,
                'household_household_food_waste':-98.6,  # https://revenue.scot/taxes/scottish-landfill-tax/slft-rates-accounting-periods
                'hotel_hotel_food_waste':-98.6,
                'restaurant_restaurant_food_waste':-98.6,
                'large_food_shop_large_food_shop_food_waste':-98.6,
                'cafe_cafe_food_waste':-98.6, 
                'pub_pub_food_waste':-98.6}

cap_factor = {"AD":1,
               "BB":0.5,
               "OCGT":0.06,
               "CHP":0.9
               }

therm_eff = {"BB":0.8,
             "CHP":0.41}

elec_eff = {"OCGT":0.39,
            "CHP":0.39}

ad_demand = {"elec":0.05,
             "therm":0.2,
             "both":0.1015}

#TAKE FROM NEWCASTLE-UNDER-LYME recycling and waste service review - WRAP 2014/15 GET REF
veh_costs = {"capex":50000,
             "opex":6250,
             "opex_per_driver":28000,
             "opex_per_km":0.96} #THIS IS JUST FUEL

lifetime = 20

#%% plant costings

"""This section is all taken from the Strathclyde whisky CHP economic model
in the same folder  - https://www.esru.strath.ac.uk//EandE/Web_sites/10-11/Whisky/economic_tool.html"""

def GetCapacity(annual_energy,eff,cap_factor):
    return (annual_energy * eff)/(8760*cap_factor)


def ADCost(res_annuals, annual_biogas): # NEEDS TO BE DICT
    # density in kg/m3; res in kg
    retention_time = 3 # days
    vol = {}
    for i in res_annuals:
        vol[i] = res_annuals[i]/365/density[i] #m3/day
    capacity = retention_time * sum(vol.values()) #m3
    capex = annual_biogas * 300 # £/MWh
    opex = annual_biogas * 25 # £/MWh
    out = {"capex":capex, # £
            "opex":opex, # £/year
            "size_ad":capacity} # m3 

    return out

def BBCost(annual_biogas):
    # annual_biogas in MWh
    annual_biogas *= 1000 # NOW IN KW
    eff_t = therm_eff["BB"]
    cf = cap_factor["BB"]
    
    capacity = GetCapacity(annual_biogas, eff_t, cf) # kW; 0.5 - 50% cap factor (Duguid, Strachan 2016)
    ad_demand_t = annual_biogas*ad_demand["both"]
    annual_t = (capacity * 8760 * cf) - ad_demand_t #kWh
    # capacity now in kW
    capex = (13.97 * capacity)+57484
    opex = 0.05 * capex
    out = {"capex":capex, # £
           "opex":opex, # £/year
           "size_t":capacity, # kW
           "size_e":0,
           "ad_demand_t": ad_demand_t, # kWh
           "annual_t": annual_t,#kWh
           "annual_e": 0, #kWh
           "annual_biogas": annual_biogas} #kWh
    return out

def BiogasStorage(biogas_mass): #kg
    cost = 3
    return {"capex":cost * biogas_mass,
            "opex": cost * biogas_mass * 0.05}

def OCGTCost(annual_biogas): 
    # annual_biogas in MWh
    annual_biogas_mass = annual_biogas * 72 #72kg/MWh
    storage_cost = BiogasStorage(annual_biogas_mass * 0.5) #50% storage is a guess IDK???
    annual_biogas *= 1000 # NOW IN KW
    
    eff_e = elec_eff["OCGT"]
    cf = cap_factor["OCGT"]
    
    capacity = GetCapacity(annual_biogas, eff_e, cf) # kW; 0.5 - 50% cap factor (Duguid, Strachan 2016)
    ad_demand_e = annual_biogas*ad_demand["both"]
    annual_e = (capacity * 8760 * cf) - ad_demand_e #kWh
    # capacity now in kW
    capex = (372.43 * capacity)+44908
    opex = 0.05 * capex
    out = {"capex": capex + storage_cost["capex"], # £
        "opex": opex + storage_cost["opex"], # £/year
        "size_e": capacity, # kW
        "size_t": 0,
        "ad_demand_e": ad_demand_e, #kWh
        "annual_e": annual_e, #kWh
        "annual_t": 0, #kWh
        "annual_biogas": annual_biogas} #kWh
    """ DOES NOT INCLUDE NETWORK/GRID CONNECTION COST"""
    return out

def CHPCost(annual_biogas):
    # annual_biogas in MWh
    annual_biogas *= 1000 # NOW IN KW
    eff_e = elec_eff["CHP"]
    eff_t = therm_eff["CHP"]
    cf = cap_factor["CHP"]
    
    capacity_e = GetCapacity(annual_biogas, eff_e,cf) #kW
    ad_demand_e = annual_biogas*eff_e*ad_demand["elec"]
    annual_e = (capacity_e * 8760 * cf) - ad_demand_e #kWh
    
    capacity_t = GetCapacity(annual_biogas, eff_t,cf) #kW
    ad_demand_t = annual_biogas*eff_t*ad_demand["therm"]
    annual_t = (capacity_t * 8760 * cf) - ad_demand_e #kWh         
                                                 
    # cost is based on the electrical capacity
    capex = (475.62 * capacity_t) + 54968
    opex = 0.05 * capex
    out = {"capex":capex, #£
            "opex":opex, #£/year
            "size_e":capacity_e, #kW
            "size_t":capacity_t, #kW
            "ad_demand_e": ad_demand_e, #kWh
            "ad_demand_t": ad_demand_t, #kWh
            "annual_e":annual_e, #kWh
            "annual_t":annual_t, #kWh
           "annual_biogas": annual_biogas}#kWh
    return out

# def VehCost(num_veh, annual_dist, ferry_costs, share_of_days=1):
def VehCost(num_veh, annual_dist, ferry_costs,driver_number = 1):
    capex = num_veh * veh_costs["capex"] 
    opex = (veh_costs["opex"] + ferry_costs + (veh_costs["opex_per_driver"] * driver_number)) * num_veh
    opex +=  veh_costs["opex_per_km"] * annual_dist
    out = {"capex":capex,
           "opex":opex,
           "annual_dist":annual_dist}
    return out
    

def GridCost(cap, cap_factor):
    out = {"capex": 0,
            "opex": 0}
    #   https://www.ssen.co.uk/about-ssen/library/charging-statements-and-information/scottish-hydro-electric-power-distribution/
    if cap==0: #this is for BB only
        return {"capex":0,
                "opex":0}
    annual_e = cap * cap_factor
    days = cap_factor * 365
    out["opex"] = (6.43*days) + (-0.75 * annual_e/365) # average of the banded hourly rate per kWh
    
    # https://www.ssen.co.uk/globalassets/connection-offer-expenses/connection-offer-expenses-customer-guide-rev3.00.pdf
    if annual_e < 250:
        out["capex"] = 938
    else:
        out["capex"] = 2556
    return out

def ThermGridCost(annual_t, cat = "mean"): #annual_t in kWh
    """this is based on average of a bulk scheme (DECC 2015)- capex range is from 
    410-1496£/MWh, average of £923 (0.41-1.496/// 0.923 £/kWh
                                    
    assume opex at 2.8%"""
    
    if annual_t==0: #this is for OCGT only
        return {"capex":0,
                "opex":0}
    
    if cat == "mean":
        k = 0.923
    elif cat == "high":
        k = 1.496
    elif cat == "low":
        k = 0.41

    capex = annual_t * k
    
    opex = capex * 0.028
    
    return {"capex":capex,
            "opex":opex}

#%% LCOE

# FROM IRENA 2018:
WACC = 0.075 # this is the discount rate

def discount(val, rate, year):
    return val/((1+rate)**year)

def LCOE(capex, opex, annual_energy,lifetime):
    # AD and transport/fuel costs are included in the capex/opex here
    if annual_energy == 0:
        return np.nan
    total_costs = [discount(opex,WACC,i) for i in range(lifetime)]
    total_costs[0] += capex # ok cause rate in year 0 is 1
    total_energy = [discount(annual_energy,WACC,i) for i in range(lifetime)]
    lcoe = sum(total_costs)/sum(total_energy)
    return lcoe

#%%

output_dic = {"annual_e":0, # kWh
          "annual_t":0, # kWh
          "size_e":0, # kW
          "size_t":0, # kW
          "size_ad":0, # m3
          "annual_biogas":0, #kWh
            "capex_total":0, #£
            "opex_total":0, #£/year
            "capex_gen":0, #£
            "opex_gen":0, #£/year
            "capex_fuel":0, #£
            "opex_fuel":0, #£/year
            "capex_therm_grid":0,#£
            "opex_therm_grid":0}#£/year

def EnergyBalance(res_annuals, biogas_annual, tech_type,dist=False,
                  t_grid_cost = "mean"):
                   # num_veh, annual_dist, ferry_costs, share_of_days=1):
    # res_annual is a dict of the materials and weights in kg,
    # other inputs are single string
    
    # tech_func needs to be one of BBCost, OCGTCost, CHPCost 
    ad_cost = ADCost(res_annuals, biogas_annual)
    if tech_type == "OCGT":
        tech_cost = OCGTCost(biogas_annual)
        
    elif tech_type == "BB":
        tech_cost = BBCost(biogas_annual)
        
    elif tech_type == "CHP":
        tech_cost = CHPCost(biogas_annual)
    else:
        raise Exception("NOT THE RIGHT TECH")
    
    grid_cost = GridCost(tech_cost["size_e"],cap_factor[tech_type])
    if not dist:
        therm_grid_cost = ThermGridCost(tech_cost["annual_t"],t_grid_cost)
    else:
        therm_grid_cost = ThermGridCost(0)
          
    output = deepcopy(output_dic) 
    output["annual_e"] = tech_cost["annual_e"] # kWh
    output["annual_t"] = tech_cost["annual_t"] # kWh
    output["size_e"] = tech_cost["size_e"] # kW
    output["size_t"] = tech_cost["size_t"] # kW
    output["size_ad"] = ad_cost["size_ad"] # m3
    output["annual_biogas"] = tech_cost["annual_biogas"] #kWh
    output["res_annual"] = sum(res_annuals.values())
    
    output["capex_therm_grid"] = therm_grid_cost["capex"]
    output["opex_therm_grid"] = therm_grid_cost["opex"]
    
    for j,i in enumerate([ad_cost,tech_cost,grid_cost,therm_grid_cost]):
        # print(j,i,"\n\n")
        output["capex_total"] += i["capex"]
        output["opex_total"] += i["opex"]
        
        if i in [ad_cost]:#, avoided_costs]:#add ad/veh costs for fuel total
            output["capex_fuel"] += i["capex"]
            output["opex_fuel"] += i["opex"]
        else:                                   #add others for gen costs only
            output["capex_gen"] += i["capex"]
            output["opex_gen"] += i["opex"]
    
    output["capex_veh"] = 0
    output["opex_veh"] = 0
    output["annual_dist"] = 0
    # print(output)
    return output



def LCOECalcs(output,lifetime,mw=False):
    
    output["res_annual"] = output["res_annual"]/1000
    
    if mw:
        output["annual_t"] = output["annual_t"]/1000
        output["annual_e"] = output["annual_e"]/1000
        output["annual_biogas"] = output["annual_biogas"]/1000
        
                
    output["lcoe_total_t"] = LCOE(output["capex_total"], output["opex_total"],
                                output["annual_t"],lifetime)
    output["lcoe_total_e"] = LCOE(output["capex_total"], output["opex_total"],
                                output["annual_e"],lifetime)
    output["lcoe_gen_t"] = LCOE(output["capex_gen"], output["opex_gen"],
                                output["annual_t"],lifetime)
    output["lcoe_gen_e"] = LCOE(output["capex_gen"], output["opex_gen"],
                                output["annual_e"],lifetime)
    output["lcoe_fuel"] = LCOE(output["capex_fuel"], output["opex_fuel"],

                                output["annual_t"]+output["annual_e"],lifetime)
    output["lcoe_total"] = LCOE(output["capex_total"], output["opex_total"],
                                output["annual_t"]+output["annual_e"],lifetime)
    output["lcoe_total_res"] = LCOE(output["capex_total"], output["opex_total"],
                                output["res_annual"],lifetime)
    output["lcoe_veh_res"] = LCOE(output["capex_veh"], output["opex_veh"],
                                   output["res_annual"],lifetime)
    output["npv_veh"] = LCOE(output["capex_veh"], output["opex_veh"],
                                   1,lifetime)
    output["lcoe_veh_dist"] = LCOE(output["capex_veh"], output["opex_veh"],
                                output["annual_dist"],lifetime)


    return output

OUTPUT_COLS = ["capex_total","opex_total","annual_t","annual_e",
              "capex_gen","opex_gen","capex_fuel","opex_fuel",
              "annual_biogas","res_annual","capex_veh","opex_veh",
              "annual_dist","capex_therm_grid","opex_therm_grid"]

def LCOECost(res_annuals, biogas_annuals, tech_type, 
           veh_costs, lifetime, t_grid_cost, refs, incentive,
           avoided = False):
    
    energy_balance = {}
    for i in refs:
        dist = "distillery" in i
        energy_balance[i] = EnergyBalance(res_annuals[i],biogas_annuals[i],
                                          tech_type, dist, t_grid_cost)
       
    total_output = {}
    for i in OUTPUT_COLS+list(output_dic.keys()):
        total_output[i] = sum([energy_balance[j][i] for j in energy_balance])
    
    veh_cost = {"capex":0,
                "opex":0,
                "annual_dist":0}
    for i in refs:
        for j in ["food","animal"]:
            veh_cost["capex"] += veh_costs[i][j]["capex"]
            veh_cost["opex"] += veh_costs[i][j]["opex"]
            veh_cost["annual_dist"] += veh_costs[i][j]["annual_dist"]
            veh_cost["opex"] += avoided[i]

    # return energy_balance
    total_output["capex_total"] += veh_cost["capex"]
    total_output["opex_total"] += veh_cost["opex"]
    total_output["capex_fuel"] += veh_cost["capex"]
    total_output["opex_fuel"] += veh_cost["opex"]
    
    total_output["capex_veh"] += veh_cost["capex"]
    total_output["opex_veh"] += veh_cost["opex"]
    total_output["annual_dist"] += veh_cost["annual_dist"]

    if incentive:
        for i in total_output:
            if "capex" in i:
                total_output[i] *= 0.5 
    
    lcoe = LCOECalcs(total_output,lifetime,True)
    return lcoe
    

   
#%%

coll_area = "Islay and Jura"

df = pd.read_csv("biomass_resource_df.csv",
                 index_col = 0, low_memory=False)
df = df.loc[df.coll_area==coll_area]
df["res_biweekly_kg"] = round(df.res_biweekly*1000)

GRAPH = "islay_jura_osmnx_graphs.pickle"
graphs = dict(zip([coll_area],[pd.read_pickle(GRAPH)]))

times = pd.read_pickle("dist_matrices_by_metric.pickle")["time_s"]

colldf = pd.read_csv("transport_costs.csv", 
                     index_col = 0)
colldf.dropna(inplace=True)

veh_num = pd.read_csv("vehicle_numbers.csv",
                      index_col=0)
for i in veh_num.index:
    n=1
    if veh_num.at[i,"veh_type"]=="food":
        n = 2
    veh_num.at[i,"cost"] = [VehCost(veh_num.at[i,"num_vehs"],veh_num.at[i,"annual_dist"],
            veh_num.at[i,"annual_ferry_cost"],n)]
    veh_num.at[i,"npv"] = LCOE(veh_num.at[i,"cost"][0]["capex"],veh_num.at[i,"cost"][0]["opex"],
                               1,lifetime)

#Add veh capex and opex:
for i in veh_num.index:
    d = veh_num.at[i,"annual_dist"]
    n = veh_num.at[i,"num_vehs"]
    f = veh_num.at[i,"annual_ferry_cost"]
    q = 1
    if veh_num.at[i,"veh_type"] == "food":
        q = 2
    c = VehCost(n, d, f, q)
    veh_num.at[i,"capex"] = c["capex"]
    veh_num.at[i,"opex"] = c["opex"]
    
veh_cost_dic = {}
veh_time_dic = {}

for res in veh_num.res_scen.unique():
    veh_cost_dic[res] = {}
    veh_time_dic[res] = {}
    for fac in veh_num.fac_scen.unique():
        veh_cost_dic[res][fac] = {}
        veh_time_dic[res][fac] = {}
        for la in veh_num.la.unique():
            veh_cost_dic[res][fac][la] = {}
            veh_time_dic[res][fac][la] = {}
            for i in veh_num.veh_type.unique():
                q = veh_num[(veh_num.res_scen==res)&(veh_num.fac_scen==fac)&(veh_num.la==la)&(veh_num.veh_type==i)]
                veh_cost_dic[res][fac][la][i] = {"capex":q["capex"].sum(),
                                                 "opex":q["opex"].sum(),
                                                 "annual_dist":q["annual_dist"].sum()}
                veh_time_dic[res][fac][la][i] = q.total_time.sum()
    

fac_la_map = {}
fac_coll_map = {}

for i in colldf.fac_scen.unique():
    fac_la_map[i] = {}
    fac_coll_map[i] = {}
    for j in df[i].unique():
        fac_la_map[i][j] = df[df[i]==j]["la"].iat[0]
        fac_coll_map[i][j] = df[df[i]==j]["coll_area"].iat[0]
        
fac_la_map["industrial_ONLY"] = fac_la_map["industrial_group_ref"]
fac_coll_map["industrial_ONLY"] = fac_coll_map["industrial_group_ref"]

coll_la_map = df.groupby("coll_area").first()["la"].to_dict()

cost_thres = {"OCGT":313,#£/MWh
              "CHP":135, #£/MWh
              "BB":130} #£/MWh

#%% RUN THE MODEL

""" CHANGE THIS TO HIGH, MEAN OR LOW TO ADJUST THERMAL GRID COSTS"""
t_grid_cost = "low"
avoided = True
incentive = True

fac_scens = ["coll_area","group_road_connected",
                         "industrial_group_ref"]#,"industrial_ONLY"]

def ResScenDf(df,scen):
    out = deepcopy(df)
    out["res_annual_scen"] = out["res_annual_scen"].apply(lambda q: literal_eval(q)[scen]).astype(float)
    out["res_annual_scen"].replace(0,np.nan,inplace=True)
    out.dropna(subset="res_annual_scen",inplace=True)
    out["res_biweekly"] = out["res_annual_scen"]/26
    out["res_biweekly_kg"] = (out["res_biweekly"]*1000).round()
    out["biogas_energy"] = out["res_annual_scen"] * out["biogas_yield"]
    out = out[out.res_biweekly>0]
    return out

resfac = pd.DataFrame(index=[],columns = ["res_scen","tech","fac_scen"]+list(output_dic.keys()))

# this is for the avoided costs dic
df.loc[:,"avoided_ref"] = df[["occ","type"]].apply(lambda q: q.iat[0]+"_"+q.iat[1],axis=1)

for res_scen in ["high","mean","low"]:
# for res_scen in ["mean"]:

    new_df = ResScenDf(df,res_scen)
    for fac_scen in ["industrial_group_ref"]:
        # for fac_scen in ["industrial_group_ref"]:
        for tech in ["BB","CHP","OCGT"]:
        #this is input to energy balance function
        
            print(tech,fac_scen)
            # this affects the resource grouping and veh_costs- clustering 
            # needs to output resource totals for smaller than island groups
            ind = "_".join([res_scen,tech,fac_scen])
            if fac_scen=="industrial_ONLY":
                ind += "_ONLY"

            if fac_scen == "coll_area":
                new_df["to_group_by"] = new_df.coll_area
                to_group_by = new_df.coll_area.unique()
            elif fac_scen == "group_road_connected":
                new_df["to_group_by"] = new_df.group_road_connected
                to_group_by = new_df.group_road_connected.unique()
            elif fac_scen == "industrial_group_ref":
                new_df["to_group_by"] = new_df.industrial_group_ref
                to_group_by = new_df.industrial_group_ref.unique()
            elif fac_scen == "industrial_ONLY":
                new_df = new_df[new_df.ref.isin(new_df.industrial_group_ref)]
                new_df["to_group_by"] = new_df.industrial_group_ref
                to_group_by = new_df.industrial_group_ref.unique()
            else:
                raise Exception("FAC_SCEN NEEDS TO BE ONE OF THREE")
            
            # input to energybalance needs to be dic of groups where each is a
            # dic of the res types
            res_dic = {}
            avoided_dic = {}
            biogas_dic = {}
            
            """ CAPEX OPEX for fuel is wrong somewhere - low for collarea highfor industrial
            MUST BE IN INPUT TO VEH COSTS FUNC"""

            veh_cost_ind = {}
            for fac in to_group_by:
                ind_fac = ind+"_"+fac
                new_df2 = new_df[new_df.to_group_by==fac]
                res_dic[fac] = (new_df2.groupby("main_type").res_annual_scen.sum()*1000).to_dict()
                biogas_dic[fac] = new_df2.biogas_energy.sum()#MWh
                
                if avoided and fac_scen:
                    avoided_dic[fac] = sum(new_df2[["res_annual_scen","avoided_ref"]].apply(lambda q: q.iat[0] * avoided_costs[q.iat[1]],axis=1))
                else: 
                    avoided_dic = False
                
                # Get vehicle stats- need per la number of vehicles divided 
                # as a proportion of the number of days per facility
                la = new_df2.la.iat[0]
                coll_area = new_df2.coll_area.iat[0]
                
                
                if fac_scen == "industrial_ONLY":
                    x = {"capex":0,
                          "opex":0,
                          "annual_dist":0}
                    x=dict(zip(["animal","liquid","food"],[x]*3))
                    veh_cost_ind[fac] = x
                    
                else:
                    # veh costs are now allocated by LA
                    veh_cost_la = veh_cost_dic[res_scen][fac_scen][la]
                    veh_time_la  = veh_time_dic[res_scen][fac_scen][la]
                    veh_time_fac = colldf[(colldf.res_scen == res_scen)&(colldf.fac_scen==fac_scen)&(colldf.ref==fac)]
                    veh_cost_ind[fac] = {}
                    for i in veh_cost_la:
                        veh_cost_ind[fac][i] = {}
                        q = veh_time_fac[veh_time_fac.veh_type==i]
                        if len(q)!=0:
                            q = sum(q.total_time.values)
                        else:
                            q = 0
                        veh_cost_ind[fac][i]["capex"] = veh_cost_la[i]["capex"] * (q/veh_time_la[i])
                        veh_cost_ind[fac][i]["opex"] = veh_cost_la[i]["opex"] * (q/veh_time_la[i])
                        veh_cost_ind[fac][i]["annual_dist"] = veh_time_fac[veh_time_fac.veh_type==i]["annual_dist"].sum()
                        
                # get the costs for the individual facilities
        ######## BIOGAS ANNUAL IS NOW IN MWH ######################
                lcoe_fac = LCOECost(res_dic,biogas_dic,
                                    tech,veh_cost_ind,lifetime,
                                    t_grid_cost, [fac], incentive,
                                    avoided_dic)
                # raise Exception
                resfac.loc[ind_fac,["res_scen","tech","fac_scen","la","ref"]] = [res_scen,tech,fac_scen,la,fac]
                resfac.at[ind_fac,"coll_area"] = fac_coll_map[fac_scen][fac]


