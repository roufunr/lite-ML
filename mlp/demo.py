
def getModelRF(current_feature):
    bootstrap = current_feature[1]
    criterion=current_feature[2]
    max_features = current_feature[3]
    min_samples_leaf = float(current_feature[4])
    min_samples_split = float(current_feature[5])
    n_estimators = int(current_feature[6])
    oob_score = current_feature[7]
    warm_start = current_feature[8]


    if max_features not in ["auto", "None", "sqrt", "log2"]:
        max_features=float(max_features)

    if max_features == "None" or max_features == "auto":
        max_features = None

    bootstrap = eval(str(bootstrap))
    warm_start = eval(str(warm_start))
    
    if(oob_score=="None"):
        oob_score=None
    else:
        oob_score = eval(str(oob_score))

    clf = RandomForestClassifier(
        bootstrap=bootstrap, 
        criterion=criterion,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        oob_score=oob_score, 
        warm_start=warm_start,
        n_jobs = 1
        )

    clf.fit(X_train, Y_train)
     
    return clf


def getModelKNN(current_feature):

    algorithm=current_feature[1]
    metric=current_feature[2]
    n_neighbors=current_feature[3]
    p=current_feature[4]
    weights=current_feature[5]
    leaf_size=current_feature[6]


    n_neighbors=int(n_neighbors)
    p=int(p)
    leaf_size=int(leaf_size)


    clf = KNeighborsClassifier(algorithm=algorithm, metric=metric,\
    n_neighbors=n_neighbors, p=p,weights=weights,leaf_size=leaf_size)

    clf.fit(X_train, Y_train)
     
    return clf




def getModelSVC(current_feature):

    C=float(current_feature[1])
    kernel=current_feature[2]
    shrinking=current_feature[3]
    gamma=current_feature[4]

    if(gamma!="auto"):
        gamma=float(gamma)

    shrinking=eval(str(shrinking))
    
    clf = SVC(C=C, kernel=kernel, shrinking=shrinking, gamma=gamma)

    clf.fit(X_train, Y_train)
     
    return clf




def getModelLogisticRegression(current_feature):

    penalty=current_feature[1]
    C=current_feature[2]
    fit_intercept=current_feature[3]
    solver=current_feature[4]
    multi_class=current_feature[5]
    warm_start=current_feature[6]
    l1_ratio=current_feature[7]

    C=float(C)
    fit_intercepts=eval(str(fit_intercepts)) 
    warm_starts=eval(str(warm_starts)) 

    if(l1_ratio!="None"):
        l1_ratio=float(l1_ratio)
    else:
        l1_ratio=None

    clf = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, solver=solver, multi_class=multi_class, warm_start=warm_start, l1_ratio=l1_ratio) 
    clf.fit(X_train, Y_train)
     
    return clf


def getModelMLP(current_feature):

    hidden_layer_sizes=current_feature[1]
    activation=current_feature[2]
    solver=current_feature[3]
    learning_rate=current_feature[4]
    warm_start=current_feature[5]

    hidden_layer_sizes=int(hidden_layer_sizes)
    warm_start=eval(str(warm_start)) 
    

    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation=activation, solver=solver, learning_rate=learning_rate, warm_start=warm_start)
    clf.fit(X_train, Y_train)
     
    return clf




def inference_time_predicted_values(clf):
    
    inferenceTime_s=[]
    predictedValues=[]
     
    for i in range(0, len(X_test)):             #for every instance  
        
        start_time = datetime.datetime.now()
        prediction=clf.predict([X_test[i]])
        delta=datetime.datetime.now() - start_time
        inferenceTime=delta.total_seconds() * 1000
        inferenceTime_s.append(inferenceTime)
        
        prediction_list=prediction.tolist()
        predictedValues=predictedValues+ prediction_list

    accuracy=precision_recall_fscore_support(Y_test, predictedValues, average='macro')
    precision=accuracy[0]
    recall=accuracy[1]
    f1score=accuracy[2]
    
    median_time=statistics.median(inferenceTime_s)
    #print(median_time,precision,recall,f1score)
    return [median_time,precision,recall,f1score]


def cpu_memory_monitor(clf, iteration_number, sleep_time):

    num_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()  
    cpu_percents_s=[]
    mem_usage_s=[]
    
    for i in range(0, len(X_test)):             #for every instance  
        worker_process=Process(target=infer,  args=(clf, X_test[i], iteration_number))
        worker_process.start()
        p = psutil.Process(worker_process.pid)
        
        while worker_process.is_alive():
            try:
                memory_info = p.memory_info()
                cpu_info = p.cpu_percent()
                
                memory_usage_mb = memory_info.rss / (1024 ** 2)
                normalized_cpu_percent = cpu_info / float(num_cores)

                cpu_percents_s.append(normalized_cpu_percent)
                mem_usage_s.append(memory_usage_mb)
            except:
                aa=2

            if(sleep_time!=0):
                time.sleep(sleep_time)
        
        worker_process.join()

        if(i==0):
            print (len(cpu_percents_s), len(mem_usage_s))
            #print (cpu_percents_s)

    
    returnMemory=-1
    returnCPU=-1
    if(len(cpu_percents_s)!=0):
        returnCPU=statistics.median(cpu_percents_s)
    if(len(mem_usage_s)!=0):
        returnMemory=statistics.median(mem_usage_s)

    return [returnCPU, returnMemory]

    

if __name__ == '__main__':

    sleep_time=0.005    #decreasing this will increase frequency of our measurement
    iteration_number=500   #increasing it will increase overall execution time, but we will be able to make more measurements

    path="./parameters/SVC_parameters.csv"

    dataReader("./F/")
    feature_reader(path)

    file=open("./results_SVC_R1.csv", "w+")
    file.write("id, inferenceTime, precision, recall, f1score, cpu, memory\n")

    for parameterSet in feature_values: #test hyperparameter
        
        id = int(parameterSet[0])
        if(id<441):
            continue
        
        #model=getModelRF(parameterSet)
        #model=getModelKNN(parameterSet)
        model=getModelSVC(parameterSet)
        
        time_accuracy=inference_time_predicted_values(model)
        
        cpu_memory=cpu_memory_monitor(model, iteration_number, sleep_time)

        average_inferenceTime=time_accuracy[0]
        average_precision=time_accuracy[1]
        average_recall=time_accuracy[2]
        average_f1score=time_accuracy[3]

        average_cpu=cpu_memory[0]
        average_memory=cpu_memory[1]

        printline=str(id)+","
        printline=printline+str(average_inferenceTime)+","
        printline=printline+str(average_precision)+","
        printline=printline+str(average_recall)+","
        printline=printline+str(average_f1score)+","
        printline=printline+str(average_cpu)+","
        printline=printline+str(average_memory)+"\n"
        file.write(printline)
        file.flush()

        print (id, average_inferenceTime, average_precision, average_recall, average_f1score, average_cpu, average_memory)
        #break
    file.close()

