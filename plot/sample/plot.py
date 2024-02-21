import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import itertools
import statistics
import math
import matplotlib.lines as mlines
import pandas as pd





def NormalizeData(data):
	if ((np.max(data) - np.min(data))==0):
		return [1]*len(data)

	returnvalue=(data - np.min(data)) / (np.max(data) - np.min(data))
	returnvalue=returnvalue+0.1
	
	return returnvalue

parameter_dic={}
value_dic={}

def inserter(key, value, diction):
	old_value=[]
	if(key in diction):
		old_value=diction.get(key)
	old_value.append(value)
	diction[key]=old_value


def KNNParser(line):
	splitted=line.split("\t")
	id=splitted[0]
	
	algorithm=splitted[1]
	metric=splitted[2]
	n_neighbors=int(splitted[3])
	p=int(splitted[4])
	weights=splitted[5]
	leaf_size=int(splitted[6])
	
	PI_inferenceTime=float(splitted[7])
	Jetson_inferenceTime=float(splitted[8])
	PC_inferenceTime=float(splitted[9])
	PI_f1score=float(splitted[10])
	Jetson_f1score=float(splitted[11])
	PC_f1score=float(splitted[12])
	PI_cpu=float(splitted[13])
	Jetson_cpu=float(splitted[14])
	PC_cpu=float(splitted[15])
	PI_memory=float(splitted[16])
	Jetson_memory=float(splitted[17])
	PC_memory=float(splitted[18])

	inserter("algorithm", algorithm, parameter_dic)
	inserter("metric", metric, parameter_dic)
	inserter("n_neighbors", n_neighbors, parameter_dic)
	inserter("p", p, parameter_dic)
	inserter("weights", weights, parameter_dic)
	inserter("leaf_size", leaf_size, parameter_dic)

	inserter("PI_inferenceTime", PI_inferenceTime, value_dic)
	inserter("Jetson_inferenceTime", Jetson_inferenceTime, value_dic)
	inserter("PC_inferenceTime", PC_inferenceTime, value_dic)
	
	inserter("PI_f1score", PI_f1score, value_dic)
	inserter("Jetson_f1score", Jetson_f1score, value_dic)
	inserter("PC_f1score", PC_f1score, value_dic)

	inserter("PI_cpu", PI_cpu, value_dic)
	inserter("Jetson_cpu", Jetson_cpu, value_dic)
	inserter("PC_cpu", PC_cpu, value_dic)

	inserter("PI_memory", PI_memory, value_dic)
	inserter("Jetson_memory", Jetson_memory, value_dic)
	inserter("PC_memory", PC_memory, value_dic)

def MLPParser(line):
	splitted=line.split("\t")
	id=splitted[0]
	

	hidden_layer_sizes=int(splitted[1])
	activation=splitted[2]
	solver=splitted[3]
	learning_rate=splitted[4]
	warm_start=splitted[5]

	
	PI_inferenceTime=float(splitted[6])
	Jetson_inferenceTime=float(splitted[7])
	PC_inferenceTime=float(splitted[8])
	PI_f1score=float(splitted[9])
	Jetson_f1score=float(splitted[10])
	PC_f1score=float(splitted[11])
	PI_cpu=float(splitted[12])
	Jetson_cpu=float(splitted[13])
	PC_cpu=float(splitted[14])
	PI_memory=float(splitted[15])
	Jetson_memory=float(splitted[16])
	PC_memory=float(splitted[17])

	inserter("hidden_layer_sizes", hidden_layer_sizes, parameter_dic)
	inserter("activation", activation, parameter_dic)
	inserter("solver", solver, parameter_dic)
	inserter("learning_rate", learning_rate, parameter_dic)
	inserter("warm_start", warm_start, parameter_dic)
	
	inserter("PI_inferenceTime", PI_inferenceTime, value_dic)
	inserter("Jetson_inferenceTime", Jetson_inferenceTime, value_dic)
	inserter("PC_inferenceTime", PC_inferenceTime, value_dic)
	
	inserter("PI_f1score", PI_f1score, value_dic)
	inserter("Jetson_f1score", Jetson_f1score, value_dic)
	inserter("PC_f1score", PC_f1score, value_dic)

	inserter("PI_cpu", PI_cpu, value_dic)
	inserter("Jetson_cpu", Jetson_cpu, value_dic)
	inserter("PC_cpu", PC_cpu, value_dic)

	inserter("PI_memory", PI_memory, value_dic)
	inserter("Jetson_memory", Jetson_memory, value_dic)
	inserter("PC_memory", PC_memory, value_dic)

def plotter3d(model, averaging):
	

	key_list=list(parameter_dic.keys())
	measures=list(value_dic.keys())
	combinations = list(itertools.combinations(key_list, 3))

	counter=0
	for measure in measures:
		for c in combinations:

			if(counter!=35):
				counter=counter+1
				continue

			#print(counter, c[0], c[1])
			#counter=counter+1

			x_value=c[0]
			y_value=c[1]
			z_value=c[2]

			d1=parameter_dic.get(x_value)
			d2=parameter_dic.get(y_value)
			d3=parameter_dic.get(z_value)
			r=value_dic.get(measure)

			#print (len(d1),len(d2),len(d3),len(r) )
	
			u1, ind1 = np.unique(d1, return_inverse=True)
			u2, ind2 = np.unique(d2, return_inverse=True)
			u3, ind3 = np.unique(d3, return_inverse=True)##3
			#xs=ind1
			#ys=ind2
			#zs=ind3

			#print (len(ind1),len(ind2),len(ind3),len(r) )
			#print (u1,u2,u3,len(r) )

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
				
				if(averaging=="mean"):
					r.append(statistics.mean(dic.get(key)))
				elif(averaging=="max"):
					r.append(max(dic.get(key)))
				elif(averaging=="min"):
					r.append(min(dic.get(key)))
				elif(averaging=="median"):
					r.append(statistics.median(dic.get(key)))
				else:
					r.append(statistics.mean(dic.get(key)))
				
				xs.append(key[0])
				ys.append(key[1])
				zs.append(key[2])

			#print (len(xs),len(ys),len(zs),len(r) )
			#print (r)

			#print (len(xs),len(ys),len(zs), len (r) )
			#print (u1,u2,u3)
			#print (len(ind1),len(ind2),len(ind3), len (r) )
			#print (ind1.shape,len(ind2),len(ind3), len (r) )
			#print (len(d2),len(d3),len(d4), len (r) )
			#print (d2)
			#print (ind1)


			

			r_normalized=NormalizeData(r)
			#print (r_normalized)



			'''
			colors=[(0.4,0 ,0.8), (0,0,1), (0.5,0,0), (0,0.5,0)]
			color_code=[]
			for z in r_normalized:
				if (z>=0.75):
					color_code.append(colors[0])
				elif (0.50<=z and z<0.75):
					color_code.append(colors[1])
				elif (0.25<=z and z<0.50):
					color_code.append(colors[2])
				elif (0<=z and z<0.25):
					color_code.append(colors[3])
			'''



			
			
			
			color_code=[]
			max_z=max(r_normalized)-0.1
			for z in r_normalized:
				z=z-0.1

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
					color_code.append("blue")######


			marker_code=[]
			for z in r_normalized:
				z=z-0.1

				if(z==max_z):
					marker_code.append("*")######

				elif(z<0.5):
					marker_code.append("+")
				elif(0.50<=z and z<0.70):
					marker_code.append("s")
				elif(0.70<=z and z<0.80):
					marker_code.append("^")
				elif(0.80<=z and z<0.90):
					marker_code.append("x")
				elif(0.90<=z):
					marker_code.append("o")######

	





			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111, projection='3d')
			#ax.scatter(xs, ys, zs, alpha=0.7, edgecolors='w', s=r)#, color=color_code)


			#ax.set_xticklabels(u1)
			#ax.set_yticklabels(u2)

			print (r_normalized)
			
			
			rn=[]
			for rr in r_normalized:
				#rn.append(math.log(rr*10))
				rn.append(rr)
				

			print (rn)
			



			for i in range(len(r)): #plot each point + it's index as text above
			    #ax.scatter(xs[i], ys[i], zs[i], color=color_code[i], s=r_normalized[i]*100) 
			    #ax.scatter(xs[i], ys[i], zs[i], color=color_code[i], s=rn[i]*100, marker=marker_code[i]) 
			    ax.scatter(xs[i], ys[i], zs[i], color=color_code[i], s=60, marker=marker_code[i]) 
			    #ax.text(xs[i], ys[i], zs[i],  '%s' % (str(r[i])), size=10, zorder=1,  color='k') 
			    #if (i==26):
			    #	break

			
			ax.set_title(measure)
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





			path="./"+model+"-3-"+averaging+"/"+measure+"/"+str(counter)+ 'png'
			plt.savefig(path)
			plt.show()
			print ("#"+path)
			plt.clf()
			plt.close()
			counter=counter+1

			#break
		#break

def plotter2d(model, averaging):
	

	key_list=list(parameter_dic.keys())
	measures=list(value_dic.keys())
	combinations = list(itertools.combinations(key_list, 2))

	counter=0
	for measure in measures:
		for c in combinations:

			#if(counter!=78):
			#	counter=counter+1
			#	continue

			#print(counter, c[0], c[1])
			#counter=counter+1

			x_value=c[0]
			y_value=c[1]
			z_value=measure

			d1=parameter_dic.get(x_value)
			d2=parameter_dic.get(y_value)
			d3=value_dic.get(z_value)
			r=value_dic.get(measure)



			#print (len(d1),len(d2),len(d3),len(r) )
	
			u1, ind1 = np.unique(d1, return_inverse=True)
			u2, ind2 = np.unique(d2, return_inverse=True)
			
			#xs=ind1
			#ys=ind2
			#zs=ind3

			#print (max(u3))

			#print (len(ind1),len(ind2),len(ind3),len(r) )
			#print (u1,u2,u3,len(r) )

			dic={}
			for i in range(0, len(ind1)):
				key=(ind1[i], ind2[i], d3[i])
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
				
				if(averaging=="mean"):
					r.append(statistics.mean(dic.get(key)))
				elif(averaging=="max"):
					r.append(max(dic.get(key)))
				elif(averaging=="min"):
					r.append(min(dic.get(key)))
				elif(averaging=="median"):
					r.append(statistics.median(dic.get(key)))
				else:
					r.append(statistics.mean(dic.get(key)))
				
				xs.append(key[0])
				ys.append(key[1])
				zs.append(key[2])

			#print (len(xs),len(ys),len(zs),len(r) )
			#print (r)

			#print (len(xs),len(ys),len(zs), len (r) )
			#print (u1,u2,u3)
			#print (len(ind1),len(ind2),len(ind3), len (r) )
			#print (ind1.shape,len(ind2),len(ind3), len (r) )
			#print (len(d2),len(d3),len(d4), len (r) )
			#print (d2)
			#print (ind1)


			colors=[(0.4,0 ,0.8), (0,0,1), (0.5,0,0), (0,0.5,0)]
			color_code=[]

			r_normalized=NormalizeData(r)
			#print (r_normalized)

			for z in r_normalized:
				if (z>=0.75):
					color_code.append(colors[0])
				elif (0.50<=z and z<0.75):
					color_code.append(colors[1])
				elif (0.25<=z and z<0.50):
					color_code.append(colors[2])
				elif (0<=z and z<0.25):
					color_code.append(colors[3])



			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111, projection='3d')
			#ax.scatter(xs, ys, zs, alpha=0.7, edgecolors='w', s=r)#, color=color_code)


			#ax.set_xticklabels(u1)
			#ax.set_yticklabels(u2)
			for i in range(len(r)): #plot each point + it's index as text above
			    ax.scatter(xs[i], ys[i], zs[i], color=color_code[i], s=r_normalized[i]*100) 
			    #ax.text(xs[i], ys[i], zs[i],  '%s' % (str(r[i])), size=10, zorder=1,  color='k') 
			    #if (i==26):
			    #	break


			ax.set_title(measure)
			ax.set_xlabel(x_value, labelpad=20)
			ax.set_ylabel(y_value, labelpad=20)
			ax.set_zlabel(z_value, labelpad=20)


			ax.set_xticks(list(set(ind1)))
			ax.set_xticklabels(u1)

			ax.set_yticks(list(set(ind2)))
			ax.set_yticklabels(u2)

			#ax.set_zticks(list(set(ind3)))##3
			#ax.set_zticklabels(u3)##3

			path="./"+model+"-2-"+averaging+"/"+measure+"/"+str(counter)+ 'png'
			plt.savefig(path)
			#plt.show()
			print ("#"+path)
			plt.clf()
			plt.close()
			counter=counter+1

			#break
		#break

def main():
    
	model="KNN"
	#model="MLP"
	dimension=3
	#dimension=2
	
	averaging="mean" 
	#averaging="median" 
	#averaging="max" 
	#averaging="min" 
	

	file=open(f"/home/rouf-linux/lite-ML/plot/sample/results_{model}", "r")
	for line in file:
		line=line.strip("\n\r")
		if ("Jetson_inferenceTime" in line):
			continue
		
		if(model=="KNN"):
			KNNParser(line)
		elif(model=="MLP"):
			MLPParser(line)

	file.close()

	if(dimension==3):
		plotter3d(model,averaging)
	if(dimension==2):
		plotter2d(model,averaging)

if __name__ == "__main__":
    main()





#model="MLP"
#dimension=3
#averaging="mean" 
#pngs 35, 5, 95





