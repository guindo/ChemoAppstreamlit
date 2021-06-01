# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:16:38 2021

@author: Mahamed
"""

import pandas as pd 
import streamlit as st
import streamlit.components as stc

import numpy as np
from sklearn.decomposition import PCA


# Load Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

matplotlib.use('Agg') # TkAgg
import seaborn as sns
# from dtale.views import startup
# from autoviz.AutoViz_Class import AutoViz_Class
from IPython import get_ipython

#machine learning 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.ensemble import StackingRegressor,StackingClassifier
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier


#metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#download
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


#st.set_page_config(layout="wide")
@st.cache


# Fxn
def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "new_text_file_{}_.txt".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)


def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "new_text_file_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

# Class
class FileDownloader(object):
	"""docstring for FileDownloader
	>>> download = FileDownloader(data,filename,file_ext).download()

	"""
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### Download File ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)

st.title("Hello Rita thank you")

data_file = st.file_uploader("independant")
        
if data_file is not None:
            st.write(type(data_file))
            file_details = {"filename":data_file.name,
			"filetype":data_file.type,"filesize":data_file.size}
            st.write(file_details)
            df = pd.read_csv(data_file)
            
            if st.button("show",key='showbutton'):
                st.dataframe(df)
                if st.button("hide",key="hidebutton1"):
                        pass
            if st.button("details independant variables",key='detailsbutton1'):
             infoindependant=df.describe()
             st.write(infoindependant)
            
data_file2 = st.file_uploader("dependant",key="dependant")
            
if data_file2 is not None:
            st.write(type(data_file2))
            file_details2 = {"filename":data_file2.name,
			"filetype":data_file2.type,"filesize":data_file2.size}
            st.write(file_details2)
            df2 = pd.read_csv(data_file2)
            if st.button("show",key='showbutton1'):
               st.dataframe(df2)
               if st.button("hide",key="hidebutton2"):
                        pass
            if st.button("details",key='detailsbutton2'):
                infordependant=df2.describe()
                st.write(infordependant)
                
               
data_file3 = st.file_uploader("categorical",key="categorical")
            
if data_file3 is not None:
            st.write(type(data_file3))
            file_details3 = {"filename":data_file3.name,
			"filetype":data_file3.type,"filesize":data_file3.size}
            st.write(file_details3)
            df3 = pd.read_csv(data_file3)
            if st.button("show",key='showbutton2'):
               st.dataframe(df3)
               if st.button("hide",key="hidebutton3"):
                        pass
            if st.button("details",key='detailsbutton3'):
                infordependant=df3.describe()
                st.write(infordependant)


def main():

    st.title('HI Welcome to my App please upload your data')
    menu =["Home","EDA","ML","About"]
    choice = st.sidebar.selectbox("Menu",menu)
   
    if choice == "Home":
        st.subheader("Home")
        # if st.button("show",key='showbutton'):
        #     st.dataframe(df)
        # if st.button("details independant variables",key='detailsbutton1'):
        #      infoindependant=df.describe()
        #      st.write(infoindependant)

       
       
    # if st.button("show",key='showbutton1'):
    #     st.dataframe(df2)
    #     if st.button("hide",key="hidebutton2"):
    #                 pass
    #     if st.button("details",key='detailsbutton2'):
    #         infordependant=df2.describe()
    #         st.write(infordependant)

    # if st.beta_expander("Score"):
    #         st.success("Hello Score") 
    # elif st.beta_expander("Loading"):
    #         st.success("Hello Loading") 


    if choice == "EDA":
        st.subheader("Exploratory data analysis ")
        n= st.number_input("Enter column",1,10000,5,key="n")
       
                # X=df.iloc[:, [0,2]]

        # X=df.values[:,1:2]
        X=df.iloc[:, n]
        y=df2.values
        # y=df.iloc[:, m]
        dX = pd.DataFrame(data = X)
        dX["label"] = df3.values

        fig,ax = plt.subplots()
        ax.scatter(X, y,label="label", c='r', alpha=0.5)

        st.pyplot(fig)
        fig1,ax1= plt.subplots()

        # Density Plot and Histogram of all arrival delays
        sns.distplot(X, hist=True, kde=True, 
              bins=int(180/5), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 4})
        # from pandas.plotting import autocorrelation_plot
        

        st.pyplot(fig1)

        
        # corr = df.corr() # We already examined SalePrice correlations
        # st.write(corr)
        
        
        
        
        
        
    if choice == "ML":
      
      

        
       menu =["PCA","Regression","Classification","About"]
       choice = st.sidebar.selectbox("Menu",menu)
       if choice == "PCA":
          st.subheader("Decide the number of component  ")
 
          nb_comp= st.number_input("Please how many number of component",0,10000,3,key="n_comp")
          pca = PCA(n_components=nb_comp)
          X_main=df.values
          y=df2.values
          y1=df3.values
          pca.fit(X_main)
          transformed = pca.transform(X_main)
          if st.checkbox("Show reduced data",key='score'):
              st.write(transformed)
              pc_df = pd.DataFrame(data = transformed)
              csv_downloader(pc_df)

              
          if st.checkbox("Plot Pc vs Pc",key='Pc'):
    
              
          # my_expander = st.beta_expander("hello let's plot PCS")
          # clicked = st.button('Click me!')
          # with my_expander:
    
              columns = [f'PC {i}' for i in range(nb_comp)]
              len(columns)

             # pc_df = pd.DataFrame(data = transformed, 
             #         columns = [f'PC {i}' for i in range(nb_comp)])
              pc_df = pd.DataFrame(data = transformed)
              pc_df=pc_df.add_prefix('PC_')
              pc_df['Cluster'] = y1
              nb_comp1= st.number_input("Choose your first PC",0,len(columns)-1,0,key="n_comp1")
              # clicked1 = st.button('Click me!')
              
              nb_comp2= st.number_input("Choose your second PC",0,len(columns)-1,1,key="n_comp2")
              nb_comp3= st.number_input("Choose your third PC",0,len(columns)-1,1,key="n_comp3")
              # if st.checkbox("Plot Pc vs Pc ",key='Pc2'):
              fig3,ax3 = plt.subplots()
              ax3=sns.scatterplot(x=f'PC_{nb_comp1}', y=f'PC_{nb_comp2}',
                              data=pc_df, hue="Cluster",style="Cluster",
                              palette="deep",legend="full")
              st.pyplot(fig3)
              if st.checkbox("Plot Pc vs Pc vs Pc",key='Pc3'):

                  cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
                  fig3_3 = plt.figure(figsize=(6,6))
    
                  ax3 = Axes3D(fig3_3) # Method 1
                  s=ax3.scatter(pc_df[f'PC_{nb_comp1}'],
                              pc_df[f'PC_{nb_comp2}'],
                              pc_df[f'PC_{nb_comp3}'], c=y1, cmap=cmap)
                  ax3.legend(*s.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
                  ax3.set_xlabel('X Label')
                  ax3.set_ylabel('Y Label')
                  ax3.set_zlabel('Z Label')
                  
                  st.pyplot( fig3_3)
              # if st.checkbox("Plot Pc vs Pc vs Pc",key='Pc3'):

            
                    
              # if st.button("hide", key="hidescore"):
              #       pass
                  
          if st.checkbox("show explained variance",key='explaining'):
                 st.write(pca.explained_variance_ratio_)
          if st.checkbox("Loading data and plot",key='loading'):

            loadings = pca.components_
            num_pc = pca.n_features_
            pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
            loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
            loadings_df['variable'] = df.columns.values
            loadings_df = loadings_df.set_index('variable')
            st.write(loadings_df)
            fig4,ax4 = plt.subplots()
            sns.lineplot(data=loadings_df.values)

            # ax4.plot(loadings_df.values)
            ax4.legend()
            st.pyplot(fig4)
            
          # if st.button("plot PC",key="Plopc"):
       if choice == "Regression":
           X=df.values
           y=df2.values
           if st.checkbox("spit the data",key='split'):
              test_size= st.number_input("how many percent of the data you want use for test set",0.0,0.9,0.2,key="test_size")

              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 1)
              if st.checkbox("calibration X",key='X_train'):
                  st.write(X_train)
              if st.checkbox("calibration y",key='y_train'):
                  st.write(y_train)
              if st.checkbox("validation X",key='X_test'):
                  st.write(X_test)
              if st.checkbox("validation y",key='y_test'):
                  st.write(y_test)
              clicked = st.button('Compute RIDGE!',key="ridge")
              if clicked:
      
                  reg=Ridge(random_state=123)
                  reg.fit(X_train, y_train)
    
    
                  y_predtrain1=reg.predict(X_train)
                  y_predtest1=reg.predict(X_test)
    
    
    
                  Rtrain=r2_score(y_train,y_predtrain1)
                  Rtest=r2_score(y_test,y_predtest1)
                
                  Mse_train = mean_squared_error(y_train,y_predtrain1)
                  Msetest=mean_squared_error(y_test,y_predtest1)
                  Result={"RCalib":Rtrain,"RPred":Rtest,
                          "MseCalib":Mse_train,"MsePred":Msetest}
                  st.write("The result of the prediction",Result)
                  
    
                  figridg,aridg =plt.subplots()
                  aridg.scatter(y_test, y_predtest1,label='MSEP {}'.format(Msetest))
                # plt.plot([y_train.min(), y_train.max()], [y_predtrain1.min(), y_predtrain1.max()], color='blue')
                  aridg.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'b-', linewidth=2, markersize=1,label="Best fit" )
                
                # plt.plot([y_test.min(), y_test.max()], [y_predtest1.min(), y_predtest1.max()], color='red')
                  aridg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='k', marker='o', linestyle='dotted',linewidth=1,
                         markersize=1, label="Identity")
                  aridg.set_xlabel("True value")
                  aridg.set_ylabel("Predicted value")
                  # aridg.title("Prediction error plot")
                  aridg.legend()
                  
                  st.pyplot(figridg)
    
                  residualstest=-(y_predtest1.ravel()-y_test.ravel())
                  residualstrain=-(y_predtrain1.ravel()-y_train.ravel())
                  figreg = plt.figure(dpi=1200)
                # ax = fig.add_subplot(1,2,1)
                  ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
                
                  ax.scatter(y_predtest1.ravel(),residualstest.ravel(), color='b',label='MSEP {}'.format(Msetest))
                  ax.scatter(y_predtrain1.ravel(),residualstrain.ravel(),marker='*', color='r',label='MSEC {}'.format(Mse_train))
                  ax.axhline(y=0, color='k', linestyle='-')
                  ax.legend()
                
                  ax.set_xlabel("Predicted value")
                  ax.set_ylabel("Residuals")
                # mode 01 from other case
                # fig1 = plt.figure()
                # ax1=plt.subplot2grid((2, 1), (0, 0), rowspan=2, colspan=1).
                  ax1 = plt.subplot2grid((1, 3), (0, 2))
                
                # ax1 = fig.add_subplot(2,3,3)
                  ax1.hist(residualstrain, bins=10, color="red",label="Calibration",orientation='horizontal')
                  ax1.hist(residualstest, bins=10, color="b",label="Prediction",orientation='horizontal')
                  ax1.set_xlabel("Distribution")
    
                  ax1.legend()
                  plt.tight_layout()
                  st.pyplot(figreg)
                  
                  
              clicked1 = st.button('Compute Xgboost!',key="Xgboost")
              if clicked1:
                  reg=XGBRegressor()

                  reg.fit(X_train, y_train)
    
    
                  y_predtrain1=reg.predict(X_train)
                  y_predtest1=reg.predict(X_test)
    
    
    
                  Rtrain=r2_score(y_train,y_predtrain1)
                  Rtest=r2_score(y_test,y_predtest1)
                
                  Mse_train = mean_squared_error(y_train,y_predtrain1)
                  Msetest=mean_squared_error(y_test,y_predtest1)
                  Result={"RCalib":Rtrain,"RPred":Rtest,
                          "MseCalib":Mse_train,"MsePred":Msetest}
                  st.write("The result of the prediction",Result)
                  
    
                  figridg,aridg =plt.subplots()
                  aridg.scatter(y_test, y_predtest1,label='MSEP {}'.format(Msetest))
                # plt.plot([y_train.min(), y_train.max()], [y_predtrain1.min(), y_predtrain1.max()], color='blue')
                  aridg.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'b-', linewidth=2, markersize=1,label="Best fit" )
                
                # plt.plot([y_test.min(), y_test.max()], [y_predtest1.min(), y_predtest1.max()], color='red')
                  aridg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='k', marker='o', linestyle='dotted',linewidth=1,
                         markersize=1, label="Identity")
                  aridg.set_xlabel("True value")
                  aridg.set_ylabel("Predicted value")
                  # aridg.title("Prediction error plot")
                  aridg.legend()
                  
                  st.pyplot(figridg)
    
                  residualstest=-(y_predtest1.ravel()-y_test.ravel())
                  residualstrain=-(y_predtrain1.ravel()-y_train.ravel())
                  figreg = plt.figure(dpi=1200)
                # ax = fig.add_subplot(1,2,1)
                  ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
                
                  ax.scatter(y_predtest1.ravel(),residualstest.ravel(), color='b',label='MSEP {}'.format(Msetest))
                  ax.scatter(y_predtrain1.ravel(),residualstrain.ravel(),marker='*', color='r',label='MSEC {}'.format(Mse_train))
                  ax.axhline(y=0, color='k', linestyle='-')
                  ax.legend()
                
                  ax.set_xlabel("Predicted value")
                  ax.set_ylabel("Residuals")
                # mode 01 from other case
                # fig1 = plt.figure()
                # ax1=plt.subplot2grid((2, 1), (0, 0), rowspan=2, colspan=1).
                  ax1 = plt.subplot2grid((1, 3), (0, 2))
                
                # ax1 = fig.add_subplot(2,3,3)
                  ax1.hist(residualstrain, bins=10, color="red",label="Calibration",orientation='horizontal')
                  ax1.hist(residualstest, bins=10, color="b",label="Prediction",orientation='horizontal')
                  ax1.set_xlabel("Distribution")
    
                  ax1.legend()
                  plt.tight_layout()
                  st.pyplot(figreg)
              clicked2 = st.button('Compute ExtraTreesRegressor!',key="ExtraTreesRegressor")
              if clicked2:
                  reg=ExtraTreesRegressor()

                  reg.fit(X_train, y_train)
    
    
                  y_predtrain1=reg.predict(X_train)
                  y_predtest1=reg.predict(X_test)
    
    
    
                  Rtrain=r2_score(y_train,y_predtrain1)
                  Rtest=r2_score(y_test,y_predtest1)
                
                  Mse_train = mean_squared_error(y_train,y_predtrain1)
                  Msetest=mean_squared_error(y_test,y_predtest1)
                  Result={"RCalib":Rtrain,"RPred":Rtest,
                          "MseCalib":Mse_train,"MsePred":Msetest}
                  st.write("The result of the prediction",Result)
                  
    
                  figridg,aridg =plt.subplots()
                  aridg.scatter(y_test, y_predtest1,label='MSEP {}'.format(Msetest))
                # plt.plot([y_train.min(), y_train.max()], [y_predtrain1.min(), y_predtrain1.max()], color='blue')
                  aridg.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'b-', linewidth=2, markersize=1,label="Best fit" )
                
                # plt.plot([y_test.min(), y_test.max()], [y_predtest1.min(), y_predtest1.max()], color='red')
                  aridg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='k', marker='o', linestyle='dotted',linewidth=1,
                         markersize=1, label="Identity")
                  aridg.set_xlabel("True value")
                  aridg.set_ylabel("Predicted value")
                  # aridg.title("Prediction error plot")
                  aridg.legend()
                  
                  st.pyplot(figridg)
    
                  residualstest=-(y_predtest1.ravel()-y_test.ravel())
                  residualstrain=-(y_predtrain1.ravel()-y_train.ravel())
                  figreg = plt.figure(dpi=1200)
                # ax = fig.add_subplot(1,2,1)
                  ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
                
                  ax.scatter(y_predtest1.ravel(),residualstest.ravel(), color='b',label='MSEP {}'.format(Msetest))
                  ax.scatter(y_predtrain1.ravel(),residualstrain.ravel(),marker='*', color='r',label='MSEC {}'.format(Mse_train))
                  ax.axhline(y=0, color='k', linestyle='-')
                  ax.legend()
                
                  ax.set_xlabel("Predicted value")
                  ax.set_ylabel("Residuals")
                # mode 01 from other case
                # fig1 = plt.figure()
                # ax1=plt.subplot2grid((2, 1), (0, 0), rowspan=2, colspan=1).
                  ax1 = plt.subplot2grid((1, 3), (0, 2))
                
                # ax1 = fig.add_subplot(2,3,3)
                  ax1.hist(residualstrain, bins=10, color="red",label="Calibration",orientation='horizontal')
                  ax1.hist(residualstest, bins=10, color="b",label="Prediction",orientation='horizontal')
                  ax1.set_xlabel("Distribution")
    
                  ax1.legend()
                  plt.tight_layout()
                  st.pyplot(figreg)    
    if choice == "About":
        st.subheader("More features are coming soon stay tune ...just beginning ")
       
           

              


              
          


          # sns.lmplot( x=f'PC_{nb_comp1}', y=f'PC_{nb_comp2}',
          #     data=pc_df, 
          #     fit_reg=False, 
          #     legend=True,
          #     hue='Cluster',
          #     scatter_kws={"s": 30}) 
          # ax3.scatter(pc_df[f'PC_{nb_comp1}'], pc_df[f'PC_{nb_comp2}'],c=y1)
          # ax3.set_xlabel(f'PC_{nb_comp1}')
          # ax3.set_ylabel(f'PC_{nb_comp2}')
          # ax3.legend()
          
   


              
            


          
          
                



        
        
       
			
       
        
        
        
        
        
        

    
        
    
    
    # st.line_chart(df)

if __name__ == '__main__':
   main()
 


    
  
    
