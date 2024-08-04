import numpy as np
import pandas as pd
import math

def getEuclideanDistance(elem1, elem2): 
    add = 0
    for i in range(1, len(elem1)-1): #Ignores the name column and the CPI column (first and last element)
        add += (elem1.iloc[i] - elem2[i]) ** 2  # Used iloc to access elements by position
    return math.sqrt(add)

def getKNN(df, newElement, k):
    distances = []
    for index, row in df.iterrows():
        distance = getEuclideanDistance(row, newElement)  #Pass the whole row to handle the omission of the first and last element internally
        distances.append((distance, row.iloc[0]))
    
    distances.sort(key=lambda x: x[0])  # Sort the distances list by the first element of the tuple (distance)
    nearest_neighbors = distances[:k]  # Take the first k elements of the sorted list
    
    return nearest_neighbors

# Script
CPI = pd.read_csv('CPI.csv')
df = pd.DataFrame(CPI)

newElement = ["Russia", 67.62, 31.68, 10.00, 3.87, 12.90]  # We have 1 less item cause we dont now the CPI

#Print the k nearest neighbors to the new element
k = 3
neighbors = getKNN(df, newElement, k)
for neighbor in neighbors:
    print(neighbor)
    
