import capacity_planning as cp
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as arima
from sklearn.metrics import mean_squared_error as mse
import sys
import statsmodels.formula.api as smf

def sensitivity_values(filepath, baseline, lead_times, buffers, target_csls, ub_racks, to_csv = False, csv_name = None):
    '''
    filename: name of csv file where data is stored
    split: percentage of data used for training the model
    baseline: baseline parameters to initialize the model, these should be stored in a dictionary
    lead_times, buffers, target_csls, ub_racks  are all lists of lower bound and upper bounds [lb, ub] for the factorial experiment
    to_csv: specifies whether or not the data generated should be written to a csv or not
    csv_name: specifies the desired name of the csv file written, if it is written. If it is not specified the name will be 
              "Sensitivity_Values.csv"
    '''
    
    df = pd.read_excel(filepath, parse_dates=['Date_A'], index_col = 'Date_A')
    ts = df['End_alloc'].dropna()

    alpha = baseline['alpha']
    split = round(len(ts)*alpha) # define the training/test split
    train = ts.iloc[0:split].values
    test = ts.iloc[split:].values
    idx = baseline['init_cap_idx']
    init_cap = max(ts.iloc[split+idx], ts.iloc[split] + 120)
    numsims = baseline['numsims'] 
    
    # Due to the current implementation of the lead time, this is how things must be changed (by defining a function here)
    def lead_time():
        '''
        Define the lead time distrubution, this can be changed depending on the dataset
        # This lead time is defined by utilizing dif from below
        '''
        lead_time = np.random.gamma(shape = 1.9, scale = 23.2)/7 + np.random.triangular(1.5,2.25,3) + dif
        lead_time = round(lead_time)
        ret = round(max(lead_time, 1))

        return ret
    
    # use these lists to build up data for dataframe
    
    # Outputs
    avg_util = []
    std_util = []
    frequency = []
    cross_pts = []
    
    # Inputs
    lead_time_dif = []
    minbuffer = []
    target_csl_ = []
    max_racks_ = []
    
    if 0 not in lead_times:
        lead_times.append(0)
        
    global dif # define dif to be global for the lead_time function
    
    for dif in lead_times:
        for minbuf in buffers:
            for targ_csl in target_csls:
                for max_rack in ub_racks:

                    cap_plan = cp.CapacityPlanning(train = train.copy(),
                                                   test = test.copy(),
                                                   init_cap = init_cap,
                                                   numsims = numsims,
                                                   orderlist = cp.OrderList([],lead_time),
                                                   minbuffer = minbuf,
                                                   target_csl = targ_csl,
                                                   max_racks = max_rack
                                                  )

                    cap_plan.simulate() # run the model
                    capdata = cap_plan.get_capacity_data()
                    capdata = capdata[: min(len(capdata), len(test))]
                    utilization = [min(i/j, 1.) for (i,j) in zip(test[:len(capdata)], capdata)]
                    avg_utilization = np.mean(utilization)
                    std_utilization = np.std(utilization)
                    ord_freq = np.mean(cap_plan.get_order_data())
                    cross_points = sum([1 if i < j else 0 for (i,j) in zip(capdata, test[:len(capdata)])])

                    # outputs
                    avg_util.append(avg_utilization)
                    std_util.append(std_utilization)
                    frequency.append(ord_freq)
                    cross_pts.append(cross_points)

                    #Inputs 
                    minbuffer.append(cap_plan.minbuffer)
                    lead_time_dif.append(dif)
                    target_csl_.append(targ_csl)
                    max_racks_.append(max_rack)

    data = np.array([avg_util, std_util, frequency, cross_pts, minbuffer, lead_time_dif, target_csl_, max_racks_]).T
    df = pd.DataFrame(data=data, columns = ['avg_util', 'std_util', 'frequency', 'cross_pts', 'minbuffer', 'lead_time_dif', 'target_csl_', 'max_racks_'])
    
    if to_csv:
        if csv_name:
            df.to_csv(csv_name)
        else:
            df.to_csv('Sensitivity_Values.csv')
            
    return df     

def analysis(df):
    def normalize(arr):
        # Normalizes data for standardized regression coefficients. 
        avg = np.mean(arr, axis = 0)
        sd = np.std(arr,axis = 0)
        arr = (arr-avg)/sd

        return arr
    
    ndf = normalize(df.copy())
    nlm = smf.ols(formula='avg_util ~ minbuffer + lead_time_dif + target_csl_ + max_racks_', data = ndf).fit()
    lm = smf.ols(formula='avg_util ~ minbuffer + lead_time_dif + target_csl_ + max_racks_', data = df).fit()

    print('Regression Results: \n')
    print(lm.summary())
    print('\nStandardized Regression Results: \n')
    print(nlm.summary())

                        