
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from  model import  Model_VAR
#RUN MODEL , Khai bao cac tham so dau vao 
def main():
    '''data_input has 8 time serise, include series : X1,X2,X3,..X8 and time columns is "date" '''
    data_input=pd.read_csv('dt.txt')
    MethodOutliers='nouse'
    WhisForBoxplot=1.5
    MethodReplace='mean'
    Maxlag=5
    Test_Size=0.2
    col_indx='date'
    var_forecast=['X1','X3','X5']
    #RUN
    res_model= Model_VAR(data_input,MethodOutliers,WhisForBoxplot,MethodReplace,Maxlag,Test_Size,col_indx,var_forecast)
    #1.--> Ảnh Đồ thị biểu diễn sự biến động  của các chuỗi trong tập dữ liệu gốc 
    print('Plot value of timeseries in data: ')
    res_visualize_data=res_model[1]
    #Show_Image(w,h, res_visualize_data)
    display(res_visualize_data)
    #model_use,res_visualize_data,res_test_granger,res_test_adf,res_model_var,res_test_resid, res_all_forecast_orinigal,res_forecast_test, res_plot_fc,res_plot_var_fc,res_accuracy, res_accuracy_2]
    #2--> Bảng kiểm định nhân quả granger: res_test_granger
    print('Result test granger_causion: ')
    res_test_granger=res_model[2]
    display(res_test_granger)
    #3-->Bảng kiểm định tính dừng: res_test_adf
    print('Result test stationary: ')
    res_test_adf=res_model[3]
    display(res_test_adf)
    #4--> Bảng kết quả  tham số ước lượng của model: res_model_var
    print('Results estimated parameter of model')
    res_model_var=res_model[4]
    display(res_model_var)
    #5-->Bảng kiểm định sự tương quan chuỗi phần dư: res_test_resid
    print('Results test residual: ')
    res_test_resid=res_model[5]
    display(res_test_resid)
    #6-->Bảng kết quả dự báo cho tập test cho tất cả chuỗi thời gian có trong dữ liệu
    print('Result forecast for data test of all time seties in data: ')
    res_all_forecast_orinigal=res_model[6]
    display(res_all_forecast_orinigal)
    #display(x_test)
    #7-->Bảng kết quả dự báo cho tập test với chuỗi thời gian cụ thể: res_forecast_test
    print('Result  forecast for data test with variable in var_forecast: ')
    res_forecast_test=res_model[7]
    display(res_forecast_test)
    #8-->Ảnh đồ thị biểu diễn kết quả dự báo và kết quả gốc cho tập test cho tất cả các chuỗi thời gian có trong tập dữ liệu.
    print('Plot forecast vs actual for all time series in data: ')
    res_plot_fc=res_model[8]
    display(res_plot_fc)
    #Show_Image(w,h,res_plot_fc)
    #9-->Ảnh đồ thị biểu diễn kết quả dự báo và kết quả gốc cho tập test của danh sách chuỗi thời gian đã chọn 
    print('Plot forecast vs actual of variables in var_forecast: ')
    res_plot_var_fc=res_model[9]
    display(res_plot_var_fc)
    #Show_Image(500,500,res_plot_var_fc)
    #10-->Bảng kiểm định độ chính xác của model: res_accuracy cho tat ca cac chuoi
    print('Accuaracy for all time series')
    res_accuracy=res_model[10]
    display(res_accuracy)
    #11 --> bang kiem dinh do chinh xac theo cac chuoi nam trong danh sach du bao var_forecast
    print('Accuaracy of time series in list var_forecast')
    res_accuracy_2=res_model[11]
    display(res_accuracy_2)
# In[44]:


if __name__ == "__main__":
    main()
