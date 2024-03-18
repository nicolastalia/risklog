import numpy as np
import xpress as xp
import pandas as pd
import time

def shortest_path(order):
    '''
    A function which takes as input an order which is a list of item numbers. 
    
    Returns the shortest path to collect all items, when starting and ending at the packaging station, and the 
    total distance to fulfill the order. 
    '''
    
    # SORTING SETS, PARAMETERS, VARIABLES
    
    # Sort the order into ascending order
    order.sort()
    
    # Drop the 0 values
    order = [i for i in order if i != 0]
    
    # Obtain the length of n
    n = len(order)
    
    # Create a copy to generate the I set
    order_2 = order.copy()
    
    # Define I to be the order with 0 attached 
    order_2.insert(0, 0)
    I = order_2
    
    # Insert "Packaging into the order list so we can obtain the desired columns 
    order.insert(0,"Packaging")
    
    # Import data file of distances
    distances_data = pd.read_excel('DistanceMatrix.xlsx', sheet_name = "DistanceMatrix Meters")

    # Drop the first column of indices
    d_dat = distances_data.drop(columns = "Index")
    
    # Select desired rows and columns
    d_data = distances_data.loc[distances_data["Index"].isin(order)] # Filter rows
    d_data = d_data[order] # Filter columns

    # Rename the packaging column to 0
    d_data = d_data.rename(columns = {"Packaging" : "0"}) 

    # Select and drop the final row
    first_r = d_data.loc[d_data["0"] == 0]
    d_data = d_data.drop(index = [96])

    # Concat the dataframes so final row is first row
    d = pd.concat([first_r, d_data])
    
    # Turn the data into an array for use
    d = d.to_numpy()
    
    # Generate a list of indices for the set
    I_ind = list(range(0, len(I)))    
    
    # Define the x variable
    x = np.array([xp.var(vartype = xp.binary, name = 'x_{0}_{1}'.format(i, j)) for i in I for j in I], 
                 dtype = xp.npvar).reshape(len(I), len(I))
    
    # Define the y variable
    y = np.array([xp.var(vartype = xp.integer, name = 'y_{0}'.format(i)) for i in I], 
                 dtype = xp.npvar)
    
    
    
    # DEFINE THE PROBLEM, DECLARE VARIABLES AND CONSTRAINTS
    
    # Set the problem
    prob = xp.problem(name = "Prob")
    
    # Add the decision variable to the problem
    prob.addVariable(x)
    prob.addVariable(y)

    
    # Add the constraints:
    
    # only one arc into each node
    prob.addConstraint(
        xp.Sum(x[i, j] for i in I_ind) == 1 for j in I_ind
    )

    # only one arc out of each node
    prob.addConstraint(
        xp.Sum(x[i, j] for j in I_ind) == 1 for i in I_ind
    )

    # have to go to a different node
    prob.addConstraint(
        x[i, i] == 0 for i in I_ind
    )

    # can't go back to the node you just came from, unless there is only 1 item to collect
    if len(I_ind) != 2:
        prob.addConstraint(
            x[i, j] + x[j, i] <= 1 for i in I_ind for j in I_ind
        )
        
    # no sub-networks
    for i in I_ind :
        for j in I_ind :
            if i != j and i != 0 and j != 0:
                prob.addConstraint(
                    y[i] - y[j] + n*x[i, j] <= n - 1
                )
        
        
    # DEFINE AND ADD OBJECTIVE
    
    # Define the objective function
    obj = xp.Sum(xp.Sum(x[i, j]*d[i, j] for i in I_ind) for j in I_ind)

    # Set the problems objective function
    prob.setObjective(obj, sense = xp.minimize)
    
    
    # WRITE AND SOLVE PROBLEM
    # Write and solve the problem
    prob.write("problem","lp") # Used to look for cause of infeasibility
    prob.solve()
    
    
    # DEFINE OUTPUTS
    
    # Obtain optimal x values, and objective value
    soln = prob.getSolution(x)
    total_distance = prob.getObjVal()
    
    # Set an empty array for arcs
    arcs = []
    
    # Determine the arcs
    for i in I_ind :
        for j in I_ind :
            if soln[i, j] == 1 :
                arcs.append([I[i], I[j]])

    # Obtain the y solution
    y_soln = prob.getSolution(y)
    
    # Create a dataframe of the nodes and the y values
    df = pd.DataFrame({"I" : I, 'y' : y_soln})
    
    # Sort based on the y values
    df2 = df.sort_values(by = ["y"])

    # The order of collection
    sequence = df2['I'].values.tolist()
    
    return total_distance, sequence

def distance(orders):
    '''
    A function which takes as input a data frame of orders. 
    
    Returns the total distance required to fulfill all orders.
    
    '''
    
    # Define the intital total distance as 0
    tot_distance = 0

    # Loop over the number of orders
    for i in range(0, len(orders)):
        
        # Obtain the order information from the data frame
        order = orders.iloc[i].values.tolist()
        
        # Run the shortest path function and obtain the distance required for given order
        total_distance, sequence = shortest_path(order)
        
        # Add this distance to the running total
        tot_distance += total_distance
    
    # Return the total distance
    return tot_distance