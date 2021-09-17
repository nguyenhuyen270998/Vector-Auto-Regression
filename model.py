#!/usr/bin/env python
# coding: utf-8

# In[31]:
import pandas as pd
from IPython.display import display
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import PIL, PIL.Image
# install pickle-mixin
import pickle
from PIL import Image


# In[32]:


#Function  return plot of all time series in data original, 
#note: trả về 1 mảng các điểm ảnh( để khi cần thì có thể lưu vào DB ở dạng Blob), muốn show ảnh ra, call function Show_Image
def Visualize_Data(data_input,col_indx):
    column=data_input.drop(col_indx,axis=1).columns.tolist()
    row=len(column)
    if row % 2==0:
        ax_row=int(row/2) 
    else:
        ax_row=int(row/2)+1
    fig, axes = plt.subplots(nrows=ax_row, ncols=2, dpi=120, figsize=(6,6))
    for col,ax in zip(column,axes.flatten()): 
        data_input.plot(kind='line',x=col_indx ,y=col, ax=ax, alpha=0.5, color='r')
    plt.tight_layout()
    fig.savefig('img.png')
    im = Image.open('img.png')
    im_ar= np.asarray(im)
    # Pickle
    im_pkl=pickle.dumps(im_ar)
    plt.close()
    return  im_pkl


# In[33]:


#Function detect outlier in each time series
def Detect_Outlier_Data(data_columns,MethodOutliers,WhisForBoxplot):
    #if data_columns is numeric
    if (data_columns.dtypes=='float64' or data_columns.dtypes=='int64'):
        if MethodOutliers=='boxplot':
            Q1=data_columns.quantile(.25)
            Q3=data_columns.quantile(.75)
            IQR=Q3-Q1
            data_outliers=data_columns.loc[(data_columns<Q1-WhisForBoxplot*IQR)|(data_columns>Q3+WhisForBoxplot*IQR)]
            return list(data_outliers.values)
        if MethodOutliers=='nouse':
            return list([])


# In[34]:


# Function replace outlier , (use mean or median)
def Replace_Outlier_Data(data_input,MethodOutliers,WhisForBoxplot,MethodReplace):
    numeric_features = list(data_input.select_dtypes(include=['int64', 'float64']).columns)
    #xac dinh va thay thế outlier trong từng chuỗi thời gian của dữ liệu gốc
    if MethodOutliers=='boxplot':
        for i in range(len(numeric_features)):
            list_outliers_of_columns_numeric=Detect_Outlier_Data(data_input[numeric_features[i]],MethodOutliers,WhisForBoxplot)
            if MethodReplace=='mean':
                data_input.loc[data_input[numeric_features[i]].isin(list_outliers_of_columns_numeric),numeric_features[i]]=data_input[numeric_features[i]].mean()
            else:
                data_input.loc[data_input[numeric_features[i]].isin(list_outliers_of_columns_numeric),numeric_features[i]]=data_input[numeric_features[i]].median()        
        else: 
             data_input=data_input


# In[35]:


# Function  Test Granger_Causation (kieems dinh nhan qua)
def Granger_Causation_Matrix(data_input,test='ssr_chi2test',maxlag=8):
    variables=data_input.columns.to_list()
    data= pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for i in data.columns:
        for j in data.index:
            test_result = grangercausalitytests(data_input[[j, i]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            data.loc[j, i] = min_p_value
    data.columns = [var + '_x' for var in variables]
    data.index = [var + '_y' for var in variables]
    return data      


# In[36]:


#Function Test Stationary
def Adfuller_Test(data_input, signif=0.05, verbose=False):
    result=[]
    for name, series in data_input.iteritems():
        test_adf = adfuller(series, autolag='AIC')
        output = {'test_statistic':round(test_adf[0], 4), 'pvalue':round(test_adf[1], 4)}
        p_value = output['pvalue'] 
        if p_value <= signif:
            result_adf='Series is Stationary'  
        else:
            result_adf='Series is Non-Stationary'   
        result.append([name,signif,output['test_statistic'],output['pvalue'],result_adf])
    df=pd.DataFrame(result,columns=['TestVariable','SignifiLevel',' TestStatistic ','Pvalue','ResultADF'])
    return df


# In[37]:


"""Function thực hiện sai phân để thu được chuỗi dừng nếu kết quả kiểm định  
ADFcho rằng chuỗi là ko dừng,  sai phân tối đa là 3 lần"""
def Differenced_Data(data_input,signif=0.05,verbose=False):
    result_adf= Adfuller_Test(data_input, signif=0.05, verbose=False)
    check=result_adf['ResultADF'].tolist()
    if check.count('Series is Non-Stationary')>2:
        #differnece 1
        data_diff_1=data_input.diff().dropna()
        result_adf_1=Adfuller_Test(data_diff_1, signif=0.05, verbose=False)
        check=result_adf_1['ResultADF'].tolist()
        if check.count('Series is Non-Stationary')>3:
            #difference 2
            data_diff_2=data_diff_1.diff().dropna()
            result_adf_2=Adfuller_Test(data_diff_2, signif=0.05, verbose=False)
            check==result_adf_2['ResultADF'].tolist()
            if check.count('Series is Non-Stationary')>4:
                data_diff_3=data_diff_2.diff().dropna()
                result_adf_3=Adfuller_Test(data_diff_2, signif=0.05, verbose=False)
                order_diff=3
                return [data_diff_3,result_adf_3,order_diff]
            else:   
                order_diff=2
                return [data_diff_2,result_adf_2,order_diff]       
        else:  
            order_diff=1
            return [data_diff_1,result_adf_1,order_diff()]
    else:
        order_diff=0
        return [data_input,result_adf,order_diff]


# In[38]:


"""Function đảo ngược lại giá trị dự báo để thu được giá trị dự báo thực tế, 
do khi model forecast thì dữ liệu dự báo đang ở trạng thái đã sai phân"""
def Invert_Transform(data_train,data_forecast,order_diff):
    if order_diff==3:
        Second_diff=True
        Third_diff=True
        First_diff=True
    if order_diff==2:
        Third_diff=False
        Second_diff=True
        First_diff=True
    if order_diff==1:
        Second_diff=False
        First_diff=True
    data_fc = data_forecast.copy()
    columns = data_train.columns
    for col in columns:        
        # Roll back 3 Diff
        if Third_diff:
            data_fc[str(col)+'_2d'] = (data_train[col].iloc[-2]-data_train[col].iloc[-3]) + data_fc[str(col)+'_3d'].cumsum()
        # Roll back 2 Diff
        #order_diff=2
        if  Second_diff:
            data_fc[str(col)+'_1d'] = (data_train[col].iloc[-1]-data_train[col].iloc[-2]) + data_fc[str(col)+'_2d'].cumsum()
        # Roll back 1 Diff
        if  First_diff:
            data_fc[str(col)+'_forecast'] = data_train[col].iloc[-1] + data_fc[str(col)+'_1d'].cumsum()
    return data_fc


# In[39]:


#function visualize result forecast vs actual
def Visualize_Data_Forecast(data_train,data_test,data_forecast,var_forecast,n_test):
    #draw all series
    fig=plt.figure(figsize=(12,10)) # can edit figure size
    row=len(data_train.columns)
    ax_row=int(row/2+1)
    for i,col in zip(range(row),data_train.columns):
        ax= plt.subplot(ax_row,3 ,i+1)
        data_forecast[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        data_test[col][-n_test:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig('img.png')
    im = Image.open('img.png')
    im_ar= np.asarray(im)
    # Pickle
    im_pkl=pickle.dumps(im_ar)
    plt.close()
    #draw some list_series
    fig=plt.figure(figsize=(12,10)) # note: can edit  figure size 
    row=len(var_forecast)
    ax_row=int(row/2+1)
    for i,col in zip(range(row),var_forecast):
        ax= plt.subplot(ax_row,3 ,i+1)
        data_forecast[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        data_test[col][-n_test:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig('img2.png')
    im2 = Image.open('img2.png')
    im2_ar= np.asarray(im2)
    # Pickle
    im2_pkl=pickle.dumps(im2_ar)
    plt.close()  
    return [im_pkl,im2_pkl]


# In[40]:


def Show_Image(w,h,image):
    im=pickle.loads(image)
    im= Image.fromarray(im)
    im = im.resize((w,h))
    #display(im)
    return im


# In[41]:


#ACCUARACY
from statsmodels.tsa.stattools import acf
def Forecast_Accuracy(data_forecast,data_test):
    res_tmp=[]
    for i,j in zip(data_test.columns,data_forecast.columns):
        mae = np.mean(np.abs(data_forecast[j] - data_test[i]))    # MAE
        mape = np.mean(np.abs(data_forecast[j] - data_test[i])/np.abs(data_test[i]))  # MAPE
        mse =  np.mean((data_forecast[j] - data_test[i])**2) # MSE
        rmse = np.mean((data_forecast[j] - data_test[i])**2)**.5  # RMSE
        res_tmp.append([i,j,mae,mape,mse,rmse])
    res_ac=pd.DataFrame(res_tmp,columns=['VARIABLE_TEST','VARIABLE_FORECAST','MAE','MAPE','MSE','RMSE'])    #if 
    return res_ac


# In[42]:


def Model_VAR(data_input,MethodOutliers,WhisForBoxplot,MethodReplace,Maxlag,Test_Size,col_indx,var_forecast):
    n_test = int(len(data_input) * Test_Size)
    n_test=4
    list_features = list(data_input.select_dtypes(include=['int64', 'float64']).columns)
    list_features.append(col_indx)
    data_input=data_input[list_features]
    data_input_tmp=data_input.set_index(col_indx,inplace=False).dropna()
    x_train, x_test = data_input[0:-n_test], data_input[-n_test:]
    #Visualize data raw, save as img, col_indx is name column date data """
    #-->col_indx='date'
    res_visualize_data=Visualize_Data(data_input,col_indx)
    res_visualize_data=Show_Image(800,800,res_visualize_data)
    #display(res_visualize_data)
    x_train_tmp=x_train.set_index(col_indx,inplace=False).dropna()
    x_test_tmp=x_test.set_index(col_indx,inplace=False).dropna()
    #Result test granger
    res_test_granger=Granger_Causation_Matrix(x_train_tmp,test='ssr_chi2test',maxlag=8)
    #TEST STATIONARY OF DATA, DIFFERENCED DATA
    res_diff_data=Differenced_Data(x_train_tmp,signif=0.05,verbose=False)
    # result test adf and x_train_new
    res_test_adf=res_diff_data[1]
    x_train_tmp_diff=res_diff_data[0]
    order_diff=res_diff_data[2]
    #fit model
    """--> Select order model"""
    model = VAR(x_train_tmp_diff)
    aic_ar=[]
    for i in range(0,Maxlag+1):
        result = model.fit(i)
        res_aic= round(result.aic,4)
        aic_ar.append(res_aic)
   # print(aic_ar)
    for j in range(len(aic_ar)-2):
        if (aic_ar[j+1]<aic_ar[j] and aic_ar[j+1]<aic_ar[j+2]):
            res=aic_ar[j+1]
        else:
            res=None
    if res==None:
        model_var = model.fit(maxlags=Maxlag,ic='aic')
        lag_order = model_var.k_ar    
    else:    
        order=aic_ar.index(res)
        model_var = model.fit(order)
        lag_order = model_var.k_ar  
    '''--> Result of model VAR is estimated parameters: '''
    res_model_var=model_var.params
    #Kiểm định sự tương quan của chuỗi phần dư
    from statsmodels.stats.stattools import durbin_watson
    test_resid = durbin_watson(model_var.resid)
    resid_ar=[]
    for col, val in zip(x_train_tmp_diff.columns,test_resid):
        resid_ar.append([col,round(val, 2)])
    res_test_resid=pd.DataFrame(resid_ar,columns=['VARIABLE','Value_DurbinWatson'])
    #Dự báo cho mốc thời gian mới
    '''--> Input data for forecasting test , lay lag_order giá trị trước đó của tập train để làm đầu vào thực hiện dự báo 
    cho tập test'''
    forecast_input = x_train_tmp_diff.values[-lag_order:]
    '''--> Forecast for x_test'''
    fc = model_var.forecast(y=forecast_input, steps=n_test)
    res_all_forecast_diff = pd.DataFrame(fc, index=data_input_tmp.index[-n_test:], columns=x_train_tmp_diff.columns + '_'+str(order_diff)+'d')#kết quả dự báo cho tất cả các chuỗi thời gian
    '''--> Đảo ngược sai phân để lấy kết quả dự báo gốc'''
    res_all_forecast_orinigal=Invert_Transform(x_train_tmp, res_all_forecast_diff,order_diff)
    col=x_train_tmp.columns
    list_col_fc=[]
    for i in col:
        list_col_fc.append(str(i)+'_forecast')
    '''-->Return results forecast for all variable of data'''
    res_all_forecast_orinigal=res_all_forecast_orinigal[list_col_fc] 
    '''--> Return results forecast of variable in var_forecast'''
    var_ar=[]
    for i in var_forecast:
        var_ar.append(str(i)+'_forecast')
    res_forecast_test=res_all_forecast_orinigal[var_ar]   
    #Visualize results Forecast vs Actuals
    res_plot_fc=Visualize_Data_Forecast(x_train_tmp,x_test_tmp,res_all_forecast_orinigal,var_forecast,n_test)
    '''--> Đồ thị kết quả dự báo và thực tế cho tất cả các chuỗi  thời gian co trong data'''
    res_plot_fc_all= res_plot_fc[0]
    res_plot_fc_all=Show_Image(800,800,res_plot_fc_all)
    '''--> Plot results forecast vs actuals  for time series in var_forecast '''
    res_plot_var_fc= res_plot_fc[1]
    res_plot_var_fc=Show_Image(500,500,res_plot_var_fc)
    #Accuaracy of model for all variable -time series in data
    res_accuracy= Forecast_Accuracy(res_all_forecast_orinigal,x_test_tmp)
    '''--> Accuaracy  of all variable in var_forecast:'''
    res_accuracy_2=res_accuracy[res_accuracy['VARIABLE_TEST'].isin(var_forecast)].reset_index(drop=True)
    #LƯU LẠI MODEL vao thu muc nao do để sử dụng
    '''C1: Lưu lại model vào thư mục nào đó để sử dụng, dùng C1 khi ko cần lưu vào DB'''
    #pickle.dump(model_var, open('C:/Users/Administrator/Downloads/code_model-FSS/SVM_inf/finalized_model.sav', 'wb'))
    #--> show model dùng lệnh: 
    #loaded_model = pickle.load(open('C:/Users/Administrator/Downloads/code_model-FSS/SVM_inf/finalized_model.sav', 'rb'))
    '''C2: Lưu model phục vụ cho việc đẩy vào DB được'''
    model_use=pickle.dumps(model_var)
    return[model_use,res_visualize_data,res_test_granger,res_test_adf,res_model_var,res_test_resid, res_all_forecast_orinigal,res_forecast_test, res_plot_fc_all,res_plot_var_fc,res_accuracy, res_accuracy_2]
    


