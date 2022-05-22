#Importing Streamlit
import streamlit as st
#Putting a title for the web app 
st.set_page_config(
     page_title="Hotel Bookings Cancellations ",
     page_icon="🏨",
     layout="wide",
     initial_sidebar_state="expanded")

#Importing relevant libraries (some may be repeated)
import numpy as np
import base64
import pandas as pd
from re import T
import plotly.express as px
from PIL import Image
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
from statistics import mean
from statistics import stdev
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
from numpy import percentile
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from statistics import mean
from statistics import stdev
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold 
from numpy import percentile
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


#Defining an image
w1 = Image.open("Pictuure.png")

init_notebook_mode(connected = "true")
header=st.container()
data=st.container()

#Setting a navigation bar
choose = option_menu(None, ["Home", "Upload",'Data Exploration','Predictions & Insights'], 
              icons=['house', 'cloud-upload', "list-task"], 
              menu_icon="cast", default_index=0, orientation="horizontal",
              styles={
              "container": {"padding": "0!important", "background-color": "#000000"},
              "icon": {"color": "white", "font-size": "25px"}, 
              "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#000000"},
              "nav-link-selected": {"background-color": "#000000"},
             }
)


backgroundColor = "#161515"
#Our home page + adding a background
if choose == "Home": 
       main_bg_ext = r"Picture5.png"
        
       st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg_ext, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
       col1, col2,col3 = st.columns([0.2,0.6,0.2])
       with col1:
           st.write("")
       with col2:
           st.image(w1,width=1100)
       with col3:
           st.write("")



if choose == "Upload":
    col11, col22,col33= st.columns([0.3,0.2,0.5])
    with col11:
       main_bg_ext1=r"Picture5.png"

       st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext1};base64,{base64.b64encode(open(main_bg_ext1, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     ) #Uploading dataset
       st.markdown("<h4 style=' color:white;'> Upload the file here:</h4><br>", unsafe_allow_html=True)
       path1=st.file_uploader("")
       if path1:
           df=pd.read_csv(path1)
           df
       st.write("")
       st.write("")
       st.markdown("<h4 style=' color:white;'> Or enter  below  the  CSV  file  path :</h4><br>", unsafe_allow_html=True)
       path = st.text_input('',placeholder="Here")
       if path:
           df = pd.read_csv(path)
           df
           
    
    with col22:
        st.write("")
        st.write("")
    with col33:
     st.write("")
     st.write("")
     st.markdown("<h3 style=' color:white;'> How large is the data? </h3><br>", unsafe_allow_html=True)
      # Checking the data
     if st.button("Shape"):
             st.write(df.shape)
     st.write("")
     st.write("")
     f1 = st.checkbox('Confirm if this is your dataset')
     if f1:
        st.markdown("<h5 style=' color:#FF4B4B;'>Confirmed! 👍 Now continue to explore your data !</h5><br>", unsafe_allow_html=True)


# Third page 
if choose == "Data Exploration":
    col111, col222,col333,col444,col555= st.columns([0.05,0.3,0.1,0.3,0.1])
    with col111:
        st.write("")
    with col222:
       df = pd.read_csv(r"hotel_bookings.csv")
       main_bg_ext2=r"Picture5.png"

       st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext2};base64,{base64.b64encode(open(main_bg_ext2, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
        )

       #First visual ( Cancelation rate)
       df2 = df['is_canceled1'].value_counts()
       df2 = pd.DataFrame({'is_canceled1':df2.index, 'freq':df2.values})
       bar2=px.bar(df2, x="is_canceled1", y="freq",color="is_canceled1",labels={'freq':'Number of Cancellations','is_canceled1':'Canceled'}, height = 250)
       st.write(bar2)

       #Second visual (Cancellation rate per Hotel)
       df4 = df.loc[df.is_canceled==1, 'hotel'].value_counts()
       df4 = pd.DataFrame({'hotel':df4.index, 'freq':df4.values})
       bar4=px.bar(df4, x="hotel", y="freq", color="hotel",labels={'freq':'Number of Cancellations'}, height = 250) 
       st.write(bar4)  
       st.write("")
       st.write("")
       st.write("")
       st.write("")
       # Checkbox for correlations
       r = st.checkbox('Check some correlations on the target variable based on the mean')
       if r:
         st.write(df.groupby("is_canceled").mean())
         if st.button("Click to see 4 insights"):
            st.markdown("<h5 style=' color:#FF4B4B;'> 1.People who cancels usually have more previous cancellations</h5><br>", unsafe_allow_html=True)
            st.markdown("<h5 style=' color:#FF4B4B;'> 2.People who cancels are not customers(repeated_guests)</h5><br>", unsafe_allow_html=True)
            st.markdown("<h5 style=' color:#FF4B4B;'> 3.When no deposits are paid, more cancelation are made </h5><br>", unsafe_allow_html=True)
            st.markdown("<h5 style=' color:#FF4B4B;'> 4.The more a client changes the booking and put special requests, the less you should worry! </h5><br>", unsafe_allow_html=True)

       st.write("")
       st.write("")
       #Checkbox2 for correlations
       f = st.checkbox('Check some correlations between types of hotels based on the mean')
       if f:
         st.write(df.groupby("hotel").mean())
         if st.button("Click to see another 2 insights"):
            st.markdown("<h5 style=' color:#FF4B4B;'> 1.City Hotel witnessed more cancellations </h5><br>", unsafe_allow_html=True)
            st.markdown("<h5 style=' color:#FF4B4B;'> 2.City Hotel has very few parking spots compared to Resort hotel</h5><br>", unsafe_allow_html=True)       
    with col333:
        st.write("")
    with col444:
       st.write("")
       #third visual (bookings per year)
       df3 = df['arrival_date_year'].value_counts(normalize = True)
       df3 = pd.DataFrame({'arrival_date_year':df3.index, 'freq':df3.values})
       bar3=px.bar(df3, x="arrival_date_year", y="freq", color="arrival_date_year", labels={'freq':'Number of Bookings','arrival_date_year':'Year'}, height = 250)
       st.write(bar3) 
       

       month_ndx = ['January', 'February','March', 'April', 'May',
          'June', 'July', 'August', 'September', 'October', 'November', 'December']
       #forth visual (bookings per month)
       df1 = df.loc[df.is_canceled==1, 'arrival_date_month'].value_counts(normalize = True).reindex(month_ndx)
       df1 = pd.DataFrame({'month':df1.index, 'freq':df1.values})
       bar=px.bar(df1, x="month", y="freq",color="month",labels={'freq':'Number of Bookings'}, height = 250)
       st.write(bar)     
    with col555:
       st.write("")

#Last page
if choose == "Predictions & Insights":

  col1111, col2222= st.columns([0.5,0.5])
  with col1111:
      st.title("City Hotel")
      st.write("")
      st.write('')
      st.write("")
      st.markdown("<h4 style=' color:Grey;'> Choose the model : </h4><br>", unsafe_allow_html=True)
      #Applying the models on City Hotel
      m1=st.selectbox("🔻",["None","Logistic Regression","K Nearest Neighbor","Random Forest (In valid)"])
      if m1=="None":
          st.write("")
      if m1=="Logistic Regression":
            # Let us seperate our numerical and categroical variables 
          df2 = pd.read_csv(r"hotel_bookings1.csv")  
          X1 = df2.drop(["is_canceled",'previous_bookings_not_canceled','children','babies'],axis=1)
          y1=df2["is_canceled"]   
          X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle = True)            
          model = LogisticRegression(solver='liblinear', random_state=0).fit(X1_train, y1_train)
          scores4 = cross_val_score(model, X1_train, y1_train, scoring='roc_auc', cv=5)
          if st.button("Check Accuracy"):
                st.write(scores4.mean())
          
          if st.button("Predict if the booking will get canceled"):
              st.write(model.predict(X1_test[-10:]))
      if m1=="K Nearest Neighbor":
            # Let us seperate our numerical and categroical variables
          
          df3 = pd.read_csv(r"hotel_bookings1.csv")  
          X2 = df3.drop(["is_canceled",'previous_bookings_not_canceled','children','babies'],axis=1)
          y2=df3["is_canceled"]   
          X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1, shuffle = True)          
          knn = KNeighborsClassifier(n_neighbors=7).fit(X2_train,y2_train)
          scores5 = cross_val_score(knn, X2_train, y2_train, scoring='roc_auc', cv=5)
          if st.button("Check Accuracy"):
                st.write(scores5.mean())
          
          if st.button("Predict if the booking will get canceled"):
              st.write(knn.predict(X2_test[-10:]))
     # if m1=="Random Forest": (99%)
            # Let us seperate our numerical and categroical variables  
         # df4 = pd.read_csv(r"C:\Users\haith\OneDrive\Desktop\hotel_bookings1.csv")  
         # X4 = df4.drop(["is_canceled",'previous_bookings_not_canceled','children','babies'],axis=1)
          #y4=df4["is_canceled"]   
        #  X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=1, shuffle = True)          
        #  RF = RandomForestClassifier().fit(X4_train,y4_train)
       #   scores6 = cross_val_score(RF, X4_train, y4_train, scoring='roc_auc', cv=5)
        #  if st.button("Check Accuracy"):
              #  st.write(scores6.mean())
        #  if st.button("Predict if the booking will get canceled"):
              #st.write(RF.predict(X4_test[-10:]))



  with col2222:
      st.title("Resort Hotel")
      df3=pd.read_csv(r"hotel_bookings2.csv")
      st.write("")
      st.write('')
      st.write("")
      st.markdown("<h4 style=' color:Grey;'> Choose the model : </h4><br>", unsafe_allow_html=True)
      #Applying the models on Resort Hotel
      m2=st.selectbox("🔹 ",["None","Logistic Regression","K Nearest Neighbor","Random Forest (Invalid)"])
      if m2=="Logistic Regression":
          df2 = pd.read_csv(r"hotel_bookings2.csv")  
          X1 = df2.drop(["is_canceled",'previous_bookings_not_canceled','children','babies'],axis=1)
          y1=df2["is_canceled"]   
          X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=1, shuffle = True)            
          model = LogisticRegression(solver='liblinear', random_state=0).fit(X1_train, y1_train)
          scores4 = cross_val_score(model, X1_train, y1_train, scoring='roc_auc', cv=5)
          if st.button("Check Accuracy :"):
                st.write(scores4.mean())          
          if st.button("Predict if the booking will get canceled :"):
              st.write(model.predict(X1_test[-10:]))
      if m2=="K Nearest Neighbor":
          # Let us seperate our numerical and categroical variables  
          df3 = pd.read_csv(r"hotel_bookings2.csv")  
          X2 = df3.drop(["is_canceled",'previous_bookings_not_canceled','children','babies'],axis=1)
          y2=df3["is_canceled"]   
          X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=1, shuffle = True)          
          knn = KNeighborsClassifier(n_neighbors=7).fit(X2_train,y2_train)
          scores5 = cross_val_score(knn, X2_train, y2_train, scoring='roc_auc', cv=5)
          if st.button("Check Accuracy :"):
                st.write(scores5.mean())
          if st.button("Predict if the booking will get canceled :"):
              st.write(knn.predict(X2_test[-10:]))
      #if m2=="Random Forest":
            # Let us seperate our numerical and categroical variables  (89%)
        #  df4 = pd.read_csv(r"C:\Users\haith\OneDrive\Desktop\hotel_bookings2.csv")  
         # X4 = df4.drop(["is_canceled",'previous_bookings_not_canceled','children','babies'],axis=1)
         # y4=df4["is_canceled"]   
          #X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=1, shuffle = True)          
         # RF = RandomForestClassifier().fit(X4_train,y4_train)
         # scores6 = cross_val_score(RF, X4_train, y4_train, scoring='roc_auc', cv=5)
         # if st.button("Check Accuracy :"):
               # st.write(scores6.mean())
         # if st.button("Predict if the booking will get canceled :"):
             # st.write(RF.predict(X4_test[-10:]))   
