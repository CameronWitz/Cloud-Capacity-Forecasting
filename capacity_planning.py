# This script houses the classes that will be used for building the simulation model

# These are the dependencies for the simulation model
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as arima
from sklearn.metrics import mean_squared_error as mse
import sys
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore') # This is only used to ignore some of the warnings generated from the ARIMA output about convergence issues,
# If that information is of value then you may comment out this line. 

################################################
# HELPER FUNCTIONS
# Helper functions that the models below rely on
def mapper_inc(order):
    '''
    Helper function for advance_t, which advances all of the orders in the orderlist by 1 time period
    '''
    order.inc()
    return order

def mapper_sim_time(order):
    '''
    Helper function that generates random arrival time for simulation given an order has been placed
    '''
    order.simulate_arrival_time()
    return order

def gridsearch_arima(p_ub, train, how = 'AIC'):
    '''
    Helper function that performs a gridsearch to find a good arima model to use for forecasting demand.
    The optimal model can be chosen with AIC, BIC, or MSE as criterion for comparison, but AIC is recommended based on
    our testing (It is set as the default. 
    '''
    # This should ideally only filter warnings within this function
    import warnings
    warnings.filterwarnings("ignore")
    
    min_score = np.inf
    p_best = 0 # ideal number of AR terms
    
    for p in range(1, p_ub+1):
        for q in range(0, 1):
            order = (p, 1, 0)
            try:
                fit = arima(train, order = order).fit()
                if(how == 'AIC'):
                    score = fit.aic
                elif(how == 'BIC'):
                    score = fit.bic
                else: 
                    score = np.mean(fit.resid.values**2) # MSE
                if score < min_score:
                    min_score = score
                    p_best = p
            except: 
                # This occurs when the model is unable to be fit with p terms. 
                pass
    if min_score == np.inf:
        # Never was able to fit a model.
        raise ValueError('Unable to fit model in the provided bounds')
    else:
        return p_best


# Specify a lead time distribution, this can be modified

def lead_time(dif = 0):
    '''
    Define the lead time distrubution, this can be changed depending on the dataset,
    however we found this distribution best captured the general behavior of all the datasets 
    observed in aggregate. 

    The triangle distribution for install time has also been implemented in this function. 

    An optional argument dif is provided in the event that one wants to determine the models behavior
    with an avg decrease or increase in lead time of dif. 
    '''
    lead_time = np.random.gamma(shape = 1.9, scale = 23.2)/7 + np.random.triangular(1.5,2.25,3) + dif
    ret = round(max(lead_time, 1)) # rounding is done to ensure an integer number of weeks

    return ret

# END OF HELPER FUNCTIONS
################


class Order:
    '''
    This is the most basic element of the simulation, an order has an increment method
    Days Left Method, and arrived method that are relied on heavily in the simulation. 
    
    The lead time we use is defined above.
    '''
    def __init__(self, time_since_placement=0, lead_time = lead_time):
        '''
        Initialized the order, and generate a random lead time
        '''
        self.time_since_placement = time_since_placement
        self.lead_time = lead_time()
        self.sim_arrival_time = None
    
    def inc(self):
        '''
        Method that increments the time since placement. 
        '''
        self.time_since_placement +=1
    
    def DaysLeft(self):
        '''
        Method that returns the number of days left until the order arrives...
        Right now this is deterministic, but will change later to encorporate randomness...
        '''
        return self.lead_time - self.time_since_placement
    
    def arrived(self):
        '''
        Returns either True or False depending on if the order has arrived or not
        '''
        # evaluate whether the order has arrived
        if self.DaysLeft() <= 0:
            return True
        else: 
            return False
    

    # Simulation funcions are used to modify the order without changing the actual time we determined for it's arrival upon 
    # initialization. 
    def simulate_arrival_time(self):
        # generate a random arrival time, conditional on that it has not arrived yet
        if self.arrived() == False:
            rand_time = lead_time()
            # Assign simulated arrival time from conditional probability
            while(rand_time < self.time_since_placement):
                rand_time = lead_time()
            self.sim_arrival_time = rand_time
        else:
            print(self.lead_time)
            print(self.DaysLeft())
            raise ValueError('Trying to simulate an order that has already arrived')
            
    def sim_DaysLeft(self):
        return self.sim_arrival_time - self.time_since_placement
    
    def sim_arrived(self):
        if self.sim_DaysLeft() <= 0:
            return True
        else:
            return False


       
class OrderList:
    '''
    This class is used to store all of the orders that are currently in the system.
    It has a few simple methods that simplify working with individual orders. 
    '''
    def __init__(self, list_ = [], lead_time = lead_time):
        '''
        The initializer is very simple, it just sets the order list "self.list_" to be an empty list
        '''
        self.list_ = list_
    
    def add_order(self, order = None):
        if order:
            self.list_.append(order)
        else:
            newOrder = Order(0, lead_time)
            self.list_.append(newOrder)
    
    def advance_t(self):
        '''
        Advances the time for all of the items in the order list. (Map applies the helper function 'mapper_inc' to 
        each order in the list)
        '''
        self.list_ = list(map(mapper_inc, self.list_)) # Advance all items by one time period
        return self.clear_orders() # Clear all of the orders that have arrived, we return the number of orders arrived
    
    
    def sim_initialize_arrival(self):
        self.list_ = list(map(mapper_sim_time, self.list_))
    
    def clear_orders(self):
        '''
        Clears all of the orders that have arrived
        '''
        cleared_list = []
        for order in self.list_:
            if order.arrived() :
                pass
            else: 
                cleared_list.append(order)
    
        added_capacity = len(self.list_) - len(cleared_list) # The difference in these two lists is the additional capacity
        self.list_ = cleared_list
        return added_capacity # return the additional capacity
    
    def amount_at_t(self, days_out):
        '''
        Calculates the amount of capacity we will have gained in back orders by a specified time 'days_out'
        '''
        count = 0
        for order in self.list_:
            if order.DaysLeft() < days_out:
                count += 1
        return count
    
    def sim_amount_at_t(self, days_out):
        '''
        Calculates the amount of orders we expect to have gained conditioned on our simulated arrival times being true. 
        '''
        count = 0
        for order in self.list_:
            if order.sim_DaysLeft() < days_out:
                count += 1
        return count
    
    
    def get_orders(self):
        '''
        Returns a copy of the orderlist to observe. 
        This is helpful for debugging but likely won't be used in the simulation.
        '''
        return self.list_.copy()


class CapacityPlanning:
    '''
    This class does the bulk of the simulation work, and relies on the other two classes to get things done.
    '''
    def __init__(self, train, test, p_ub=6, init_cap = 0, p=1, how = 'AIC', orderlist = OrderList(list_=[],lead_time=lead_time), current_time = 0,
        numsims = 1000, window=1, range_=15, target_csl=0.99, max_racks = 5, buffer_met = 0.95, minbuffer = 100, refit=0):
        
        '''
        The initializer comes largely with default values that will not be specified by the user when they create a 
        CapacityPlanning object. 
        
        '''
        # Starting parameters:
        self.refit = refit #__________How many weeks between refitting the model (1 = every week, 0 = never refit, 2 = every other week etc).
        self.minbuffer = minbuffer #__The minimum difference we desire between projected demand and projected capacity.
        self.buffer = buffer_met #____The minimum percentage of the time we desire our buffer (minbuffer) to be satisfied.
        self.max_racks = max_racks #__The maximum number of racks we will allow to be purchased in any one week.
        self.rack_to_pod = 120 #______The number of pods that come in a rack (this can be changed if the conversion ever changes)
        self.window = window #________The smoothing window used on the training data. 1 = no smoothing. 
        self.train_len = len(train)#__Length of the training dataset when the model is initialized (this is only used for summary output)
        self.target_csl = target_csl#_Target csl that we want our model to aim for
        self.capacity = init_cap #____Initial capacity as specified on creation, default is 0, but should be changed to a sensible value depending on the data center. 
        self.train = train #__________Training set at start of the simulation, this gets added to as we move through time
        self.test_chg = test #________This is the test set that we diminish to add to the training set as we simulate moving through time
        self.test = test #____________Test set that remains stagnant for final analysis
        self.p_ub = p_ub #____________Upper bound on # of AR terms for gridsearch
        self.p = p #__________________Starting number of AR terms, but will be overwritten in gridsearch
        self.arima_model = None #_____At creation time we don't have an ARIMA model built
        self.how = how #______________Specify how to evaluate the gridsearch in choosing a model, AIC or BIC reccommended, but MSE is available
        self.orderlist = orderlist #__Specify an initial list of orders, by default it's empty, but can account for backorders if input with a starting list
        self.current_time = 0 #_______Keeps track of the current time so that we don't simulate past the data in our test set.
        self.hist_cap = [init_cap] #__Tracks the capacity level to evaluate when the simulation is finished. At creation it is a list with only the initial capacity
        self.stop_time = len(test) #__Stopping time for when we are out of validation data
        self.orders_placed = [] #_____Keep track of the order times and frequencies. 
        self.build_iter = 0 #_________Criteria for gridsearch to keep track of how many times we are looking for new parameters on build failure
        self.numsims = numsims #______The number of simulations to base order quantity on
        self.range = range_ #_________The number of weeks to look ahead to when evaluating our order quantities. 
    
    def modify_bounds(self, new_p_ub):
        '''
        Changes the bounds for the gridsearch
        '''
        self.p_ub = new_p_ub
    
    def build_ARIMA(self, optimize = False, verbose = 0):
        # Constructs the AR model, using gridsearch if it's the first time period
        if self.build_iter > 2:
            raise ValueError('Unable to fit arima model to dataset, try using different training data')

        try:
            train = pd.Series(self.train).rolling(self.window).mean().values
            train = train[~np.isnan(train)]

            refit = ~(self.current_time%self.refit) if self.refit != 0 else 0
            if(optimize or self.current_time == 0):  
            # gridsearch for optimal # of AR terms p
                self.p = gridsearch_arima(self.p_ub, train, self.how)
                order = (self.p, 1, 0)
                self.arima_model = arima(train, order=order).fit()

            elif(refit == -1):
                order = (self.p, 1, 0) # uses the same order as the previous fit
                self.arima_model = arima(train, order=order).fit()

        except: 
            self.build_iter += 1
            self.p_ub +=1
            self.build_ARIMA(optimize = True, verbose=1)

        self.build_iter = 0
    
    def sim_demand(self):
        # Simulates a single demand instance
        def rescale_forecast(forecast, initial_val):
            forecast_cumsum = np.cumsum(forecast, axis=0)
            scaled_forecast = forecast_cumsum + initial_val
            return scaled_forecast

        coefs = self.arima_model.params
        
        intercept = coefs[0]
        coefs = np.flip(np.delete(coefs,[0]), axis = 0)
        prediction = []

        # AR coefficients
        #ar_coefs = coefs[self.q:]
        ar_coefs = coefs
        
        train = pd.Series(self.train).rolling(self.window).mean().values
        train = train[~np.isnan(train)]
        tdata = np.diff(train)[-self.p:]
        initial_val = train[-1]

        mean = np.mean(self.arima_model.resid)
        std = np.std(self.arima_model.resid)

        for i in range(0, self.range):
            # epsilon error is used for simulation and becomes the next most recent residual observation...
            try:
            	epsilon = np.random.normal(mean, std, 1) # for some reason sometimes the residuals are Nan
            except:
            	print("Model parameters= ", self.p, self.q)
            	print("Model coefs = " , self.arima_model.params)
            	print("Model Residuals:")
            	print(self.arima_model.resid)
            	raise ValueError

            yhat = intercept  + sum(ar_coefs * (tdata-intercept)) + epsilon

            tdata = np.append(tdata[1:], yhat)

            prediction.append(yhat)

        prediction = rescale_forecast(prediction, initial_val)
        return prediction
    
    def eval_order_level(self, order_qty):
        '''
        Method to evaluate a single order level on the basis of csl and a minimum buffer
        '''
        order_list = OrderList(list_ = self.orderlist.list_.copy())
        # Add the proposed order level amount of orders, this is what we are evaluating 
        for i in range(order_qty):
            order_list.add_order()
        
        csl_counter = 0 # keep track of the number of instances where demand is completely satisfied
        avg_csl = 0
        avg_buffer_met = 0
        buffer_met = 0 # make sure we have a buffer on demand

        for i in range(self.numsims):
            order_list.sim_initialize_arrival() # Simulate random arrival times 
            projected_cap = [self.capacity + (self.rack_to_pod * order_list.sim_amount_at_t(t)) for t in range(self.range)]
            
            # Calculate metrics to gauge how good this order level performs....
            pred = self.sim_demand() # simulate an instance of demand
            count = 0
            for (dem, cap) in zip(pred, projected_cap):
                if dem <= cap:
                    count += 1
                    if (cap - dem >= self.minbuffer):
                        buffer_met += 1
                else:
                    pass
            t_csl = count/len(pred)
            t_buffer_met = buffer_met/len(pred)
            avg_csl += t_csl
            avg_buffer_met += t_buffer_met
        
        avg_csl = avg_csl/self.numsims
        avg_buffer_met = avg_buffer_met/self.numsims
        
        return avg_csl, avg_buffer_met
        
    
    def MakeOrders(self, newdata = True, testing = False):
        if newdata == True:
            self.build_ARIMA()
        
        ub = self.train[-1]*1.5    
        rem = ub%self.rack_to_pod
        ub = ub - rem if rem <= self.rack_to_pod/2 else ub - rem + self.rack_to_pod
        ub = ub//self.rack_to_pod
        bound = min(self.max_racks, ub)# this seems to be a reasonable upper bound on the number of racks we would ever order at a time
        
        best_qty = None
        order_qty = 0
        while(order_qty < bound):
            avg_csl, avg_buffer_met = self.eval_order_level(order_qty = order_qty)
            if avg_buffer_met >= self.buffer:
                if avg_csl >= self.target_csl:
                    best_qty = order_qty
                    break
            order_qty += 1
        
        if best_qty == None:
            best_qty = bound
        
        # Make the orders at the quantity reccommended
        if testing:
            return best_qty
    
        for i in range(int(best_qty)):
            self.orderlist.add_order()

        self.orders_placed.append(best_qty)
    
    def time_step(self, testing=False):
        if self.current_time == 0:
            self.MakeOrders(newdata = True)
            self.current_time += 1
            added_capacity = self.orderlist.advance_t()
            self.capacity += added_capacity*self.rack_to_pod
            self.hist_cap.append(self.capacity)
       
        elif (self.current_time >= self.stop_time):
            pass # We are done so we should just do nothing here...
            
        else:
            added_cap = self.orderlist.advance_t() # Advance the current time in our orders list
            self.capacity += added_cap*self.rack_to_pod
            self.current_time += 1 # Advance the time by 1 for the simulation
            self.hist_cap.append(self.capacity) # store the added capacity to our historical data
            
            self.train = np.append(self.train, self.test_chg[0])
            self.test_chg = self.test_chg[1:]
            self.MakeOrders(newdata = True, testing = testing)
            
    def simulate(self):
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('*'*0, 100/50*(0)))
        sys.stdout.flush()
        for i in range(len(self.test)):
            # Advance in time
            self.time_step()
            # add to the loading bar
            x = ((i+1)*50)//len(self.test)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('*'*x, 100/len(self.test)*(i+1)))
            sys.stdout.flush()
   
    def get_order_data(self):
        return self.orders_placed.copy()
    
    def get_capacity_data(self):
        '''
        returns the capacity data over all of the time in this simulation, for plotting and analysis. 
        '''
        return self.hist_cap.copy()

    def summary(self, plot = False):
        buy_data = self.get_order_data()
        capdata = self.get_capacity_data()
        capdata = capdata[: min(len(capdata), len(self.test))]
        buy_amounts = [np.nan if i == 0 else i*self.rack_to_pod for i in buy_data]
        if plot:
            plt.figure(figsize = (13,8))
            plt.plot(capdata, color = 'teal', label = 'Capacity Level')
            plt.plot(self.test[:len(capdata)], color= 'red', label='Demanded PODs')
            plt.plot(buy_amounts, 'g*', label = 'Orders Placed')
            plt.title('Simulation Results')
            plt.legend(loc='best')
            plt.show()

        utilization = [min(i/j, 1.) for (i,j) in zip(self.test, capdata)]
        avg_utilization = np.mean(utilization)
        std_utilization = np.std(utilization)
        racks = [i for i in buy_data] # convert to the number of racks we are ordering
        order_freq = np.mean(racks)
        max_order = max(racks)
        tr_test_spl = self.train_len/(self.train_len + len(self.test))
        cross_points = sum([1 if i < j else 0 for (i,j) in zip(capdata, self.test[:len(capdata)] ) ] )
        # Calculate the number of cross points

        print('Simulation Parameters:')
        print('Demand Instances Simulated = %d'%self.numsims)
        print('Planning Horizon = %d weeks'%self.range)
        print('Starting Capacity = %d (demand 20 weeks from most recent observation)'% capdata[0])
        print('Train-Test Split = %.2f - %.2f' % (tr_test_spl, 1- tr_test_spl))
        print('\nOutput Statistics:')
        print('Avg Utilization = %.3f\nStdDev Utilization = %.3f'%(avg_utilization, std_utilization))
        print('Average Order Frequency (racks): %.2f' % order_freq)
        print('Maximum Order Size (racks): %d'%max_order)
        print('# of Weeks where Demand not met: %d'%cross_points)
        print('Capacity Buffer Used = %d' % self.minbuffer)
        print('')


# Function to initialize an order list with back orders to be used in the capacity planning model when existing orders need to be 
# taken into account

def initialize_orderlist(weeksago):
    OList = OrderList([])
    for time in weeksago:
        order = Order(time_since_placement = time)
        while(order.lead_time <= time):
            order.lead_time = lead_time()
        OList.add_order(order)
    return OList










