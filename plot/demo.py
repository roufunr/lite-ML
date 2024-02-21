import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import itertools
import statistics
import math
import matplotlib.lines as mlines
import os


parameter_dic = {}
value_dic = {}

def normalize(data):
	if ((np.max(data) - np.min(data))==0):
		return [1]*len(data)

	returnvalue=(data - np.min(data)) / (np.max(data) - np.min(data))
	returnvalue = returnvalue + 0.1
	
	return returnvalue

def draw(xyz_axis_names, value_axis_name, aggregation):
    x_value=xyz_axis_names[0]
    y_value=xyz_axis_names[1]
    z_value=xyz_axis_names[2]

    d1=parameter_dic.get(x_value)
    d2=parameter_dic.get(y_value)
    d3=parameter_dic.get(z_value)
    r=value_dic.get(value_axis_name)

    u1, ind1 = np.unique(d1, return_inverse=True)
    u2, ind2 = np.unique(d2, return_inverse=True)
    u3, ind3 = np.unique(d3, return_inverse=True)
    
    dic={}
    for i in range(0, len(ind1)):
        key=(ind1[i], ind2[i], ind3[i])
        if(key in dic):
            value=dic.get(key)
            value.append(r[i])
            dic[key]=value
        else:
            dic[key]=[r[i]]

    r=[]
    xs=[]
    ys=[]
    zs=[]
    for key in dic:
        if(aggregation=="mean"):
            r.append(statistics.mean(dic.get(key)))
        elif(aggregation=="max"):
            r.append(max(dic.get(key)))
        elif(aggregation=="min"):
            r.append(min(dic.get(key)))
        elif(aggregation=="median"):
            r.append(statistics.median(dic.get(key)))
        else:
            r.append(statistics.mean(dic.get(key)))

        xs.append(key[0])
        ys.append(key[1])
        zs.append(key[2])

    
    r_normalized = normalize(r)
    
    color_code=[]
    max_z = max(r_normalized)-0.1
    for z in r_normalized:
        z = z-0.1
        if(z==max_z):
            color_code.append("red")######
        elif(z<0.5):
            color_code.append("darkcyan")
        elif(0.50<=z and z<0.70):
            color_code.append("darkorange")
        elif(0.70<=z and z<0.80):
            color_code.append("magenta")
        elif(0.80<=z and z<0.90):
            color_code.append("green")
        elif(0.90<=z):
            color_code.append("blue")

    marker_code=[]
    for z in r_normalized:
        z = z - 0.1
        if(z == max_z):
            marker_code.append("*")

        elif(z<0.5):
            marker_code.append("+")
        elif(0.50<=z and z<0.70):
            marker_code.append("s")
        elif(0.70<=z and z<0.80):
            marker_code.append("^")
        elif(0.80<=z and z<0.90):
            marker_code.append("x")
        elif(0.90<=z):
            marker_code.append("o")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    rn=[]
    for rr in r_normalized:
    	rn.append(rr)
    
    for i in range(len(r)): 
        ax.scatter(xs[i], ys[i], zs[i], color=color_code[i], s=60, marker=marker_code[i]) 

    
    ax.set_title(value_axis_name)
    ax.set_xlabel(x_value, labelpad=5)
    ax.set_ylabel(y_value, labelpad=10)
    ax.set_zlabel(z_value, labelpad=1)


    ax.set_xticks(list(set(ind1)))
    ax.set_xticklabels(u1)

    ax.set_yticks(list(set(ind2)))
    ax.set_yticklabels(u2)

    ax.set_zticks(list(set(ind3)))##3
    ax.set_zticklabels(u3)##3

    marker1 = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=6, label='max')
    marker2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6, label='[0.9 - 1.0]')
    marker3 = mlines.Line2D([], [], color='green', marker='x', linestyle='None', markersize=6, label='[0.8 - 0.9)')
    marker4 = mlines.Line2D([], [], color='magenta', marker='^', linestyle='None', markersize=6, label='[0.7 - 0.8)')
    marker5 = mlines.Line2D([], [], color='darkorange', marker='s', linestyle='None', markersize=6, label='[0.5 - 0.7)')
    marker6 = mlines.Line2D([], [], color='darkcyan', marker='+', linestyle='None', markersize=6, label='[0 - 0.5)')			

    plt.legend(handles=[marker1, marker2, marker3, marker4, marker5, marker6],  bbox_to_anchor=(0, 0.6), loc=1, borderaxespad=0.1)

    os.makedirs("/home/rouf-linux/4dplot", exist_ok=True)
    path = f"/home/rouf-linux/4dplot/{xyz_axis_names[0]}_{xyz_axis_names[1]}_{xyz_axis_names[2]}_{value_axis_name}.png"
    plt.savefig(path)
    plt.show()
    plt.clf()
    plt.close()

    print(f"Done plotting {path}")