import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def allocator(orders,distance):
    total_items = []
    for order_list in orders.values:
        for item in order_list:
            total_items.append(item)
    
    count = dict(Counter(total_items))
    del count[0]
    sorted_count = dict(sorted(count.items(),key=lambda item: item[1], reverse=True))

    bin_df = pd.DataFrame.from_dict(sorted_count, orient='index').reset_index()
    bin_df.columns = ['Product','Occurance']

    new_allocation = distance_df.merge(bin_df,left_index=True, right_index=True)[['Shelve','Product']]

    return new_allocation

def order_modifier(orders,new_allocation):
    equivalent_values_dict = new_allocation.set_index('Product')['Shelve'].to_dict()
    new_ord = orders.copy()
    positions = ['Position 1','Position 2','Position 3','Position 4','Position 5']
    for col in positions:
        new_ord[col] = new_ord[col].map(equivalent_values_dict)
        new_ord[col] = new_ord[col].astype('Int64')
    new_ord = new_ord.fillna(0)
    
    return new_ord

def swaper(allocation,a,b):
    # Load in the data file of orders
    orders = pd.read_excel('OrderList.xlsx', sheet_name = "Orders")

    # Drop the order numbers
    orders = orders.drop(columns = "Order No.")

    # a = np.random.randint(allocation['Shelve'].min(), high=allocation['Shelve'].max(), size=None, dtype=int)
    # b = np.random.randint(allocation['Shelve'].min(), high=allocation['Shelve'].max(), size=None, dtype=int)
    # while a==b:
    #     b = np.random.randint(allocation['Shelve'].min(), high=allocation['Shelve'].max(), size=None, dtype=int)
    
    product_a = allocation['Product'][allocation['Shelve']==a].values[0]
    product_b = allocation['Product'][allocation['Shelve']==b].values[0]

    allocation['Product'][allocation['Shelve']==a] = product_b
    allocation['Product'][allocation['Shelve']==b] = product_a

    orders2 = order_modifier(orders,allocation)

    return allocation, orders2