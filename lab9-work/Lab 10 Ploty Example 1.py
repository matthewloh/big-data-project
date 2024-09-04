# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:11:11 2023

@author: cwbong
"""
import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title("Dashbaord")

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def displayChart():
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "polar"}],
               [{"type": "domain"}, {"type": "scene"}]],
        )

    fig.add_trace(go.Bar(y=[2, 3, 1]),
                  row=1, col=1)

    fig.add_trace(go.Barpolar(theta=[0, 45, 90], r=[2, 3, 1]),
                  row=1, col=2)

    fig.add_trace(go.Pie(values=[2, 3, 1]),
                  row=2, col=1)

    fig.add_trace(go.Scatter3d(x=[2, 3, 1], y=[0, 0, 0],
                          z=[0.5, 1, 2], mode="lines"),
                          row=2, col=2)

    fig.update_layout(height=700, showlegend=False)

    plot(fig)

        
import pandas as pd
import matplotlib.pyplot as plt

def plotfromCSV():

    # Load big data into a Pandas dataframe
    df = pd.read_csv('weight-height.csv')
    
    fig = go.Figure(data=[go.Scatter(mode='markers', x=df['Height'], y=df['Weight'])])
    fig.update_traces(marker=dict(size=2,
                             line=dict(width=2,
                             color='DarkSlateGrey')),
                             selector=dict(mode='markers'))
    plot(fig)

# Create the labels and entries for task name and due date
firstlabel = tk.Label(root, text="Click left button to show Multiple charts")
firstlabel.grid(row=0, column=1, padx=20, pady=20)
secondlabel = tk.Label(root, text="Click Right button to show CSV Data")
secondlabel.grid(row=0, column=2, padx=20, pady=20)

# Create the buttons for adding, editing, and deleting tasks
add_task_button = tk.Button(root, text="Display Multiple Charts", command=displayChart)
add_task_button.grid(row=1, column=1, padx=5, pady=5)
add_task_button = tk.Button(root, text="Display from CSV", command=plotfromCSV)
add_task_button.grid(row=1, column=2, padx=20, pady=20)

# Start the main loop
root.mainloop()
