def tune_nEstimators(registered_or_casual,n_estimators):
    rmse_list=[]
    for i in n_estimators:
        t0 = time()
        gb = GradientBoostingRegressor(
                          n_estimators=i,
                          learning_rate=0.05,
                          max_depth=10,                      
                          min_samples_leaf=20,
                          random_state=2)
        gb.fit(fe_train_data, train_data[registered_or_casual])
        predicted_y = gb.predict(fe_dev_data)
        rmse = get_RMSE(actual_values = dev_data[registered_or_casual], predicted_values = predicted_y)
        rmse_list.append(round(rmse,3))
        print row_format.format("n_estimator", i, "RMSE", round(rmse,3),
                                "Runtime", round((time() - t0),3))
    plt.plot(n_estimators,rmse_list)
    plt.show()
    
    
def tune_learningRate(registered_or_casual,learning_rate):
    rmse_list=[]
    for i in learning_rate:
        t0 = time()
        gb = GradientBoostingRegressor(
                          n_estimators=115,
                          learning_rate=i,
                          max_depth=10,                      
                          min_samples_leaf=20,
                          random_state=2)
        gb.fit(fe_train_data, train_data[registered_or_casual])
        predicted_y = gb.predict(fe_dev_data)
        rmse = get_RMSE(actual_values = dev_data[registered_or_casual], predicted_values = predicted_y)
        rmse_list.append(round(rmse,3))
        print row_format.format("learning_rate", i, "RMSE", round(rmse,3),
                                "Runtime", round((time() - t0),3))
    plt.plot(learning_rate,rmse_list)
    plt.show()
    
    
def tune_maxDepth(registered_or_casual,max_depth):
    rmse_list=[]
    for i in max_depth:
        t0 = time()
        gb = GradientBoostingRegressor(
                          n_estimators=115,
                          learning_rate=0.05,
                          max_depth=i,                      
                          min_samples_leaf=20,
                          random_state=2)
        gb.fit(fe_train_data, train_data[registered_or_casual])
        predicted_y = gb.predict(fe_dev_data)
        rmse = get_RMSE(actual_values = dev_data[registered_or_casual], predicted_values = predicted_y)
        rmse_list.append(round(rmse,3))
        print row_format.format("max_depth", i, "RMSE", round(rmse,3),
                                "Runtime", round((time() - t0),3))
    plt.plot(max_depth,rmse_list)
    plt.show()
    
    
def tune_minSamplesLeaf(registered_or_casual,min_samples_leaf):
    rmse_list=[]
    for i in min_samples_leaf:
        t0 = time()
        gb = GradientBoostingRegressor(
                          n_estimators=115,
                          learning_rate=0.05,
                          max_depth=10,                      
                          min_samples_leaf=i,
                          random_state=2)
        gb.fit(fe_train_data, train_data[registered_or_casual])
        predicted_y = gb.predict(fe_dev_data)
        rmse = get_RMSE(actual_values = dev_data[registered_or_casual], predicted_values = predicted_y)
        rmse_list.append(round(rmse,3))
        print row_format.format("min_samples_leaf", i, "RMSE", round(rmse,3),
                                "Runtime", round((time() - t0),3))
    plt.plot(min_samples_leaf,rmse_list)
    plt.show()
