import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import BytesIO

st.set_page_config(page_title='MysteryLens AI', page_icon='🔍', layout='wide')
st.title('🔍 MysteryLens AI')
st.caption('ALG-DATA-01 • Data Investigation & Predictive Analysis')

@st.cache_data
def load_demo():
    return pd.read_csv('data/mystery_demo.csv')

def clean_data(df):
    df = df.copy().dropna(axis=1, how='all').drop_duplicates()
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude=np.number).columns:
        if df[c].isna().any():
            m = df[c].mode()
            df[c] = df[c].fillna(m.iloc[0] if len(m) else 'Unknown')
    return df

uploaded = st.file_uploader('📂 Upload your mystery CSV (optional)', type=['csv'])
raw = pd.read_csv(uploaded) if uploaded else load_demo()
df = clean_data(raw)
st.success('Using: ' + ('Uploaded dataset' if uploaded else 'Built-in demo dataset'))

c1,c2,c3,c4=st.columns(4)
c1.metric('Rows', f'{len(df):,}')
c2.metric('Columns', f'{df.shape[1]:,}')
c3.metric('Missing Values', f'{int(df.isna().sum().sum()):,}')
c4.metric('Duplicate Rows Removed', f'{len(raw)-len(df):,}')

t1,t2,t3,t4,t5=st.tabs(['📊 Profile','🔗 Relationships','🚨 Anomalies','🤖 Prediction','💡 Findings'])

with t1:
    st.subheader('Dataset Preview')
    st.dataframe(df.head(15), use_container_width=True)
    profile=pd.DataFrame({'Column':df.columns,'Data Type':[str(df[c].dtype) for c in df.columns],'Missing':[int(df[c].isna().sum()) for c in df.columns],'Unique Values':[int(df[c].nunique()) for c in df.columns]})
    st.subheader('Column Profile'); st.dataframe(profile,use_container_width=True)
    nums=df.select_dtypes(include=np.number).columns.tolist()
    if nums:
        col=st.selectbox('Choose a numeric column',nums)
        st.plotly_chart(px.histogram(df,x=col,title=f'Distribution of {col}',marginal='box'),use_container_width=True)

with t2:
    st.subheader('Correlation Discovery')
    nums=df.select_dtypes(include=np.number)
    if nums.shape[1]>=2:
        corr=nums.corr()
        st.plotly_chart(px.imshow(corr,text_auto='.2f',aspect='auto',title='Numeric Feature Correlation Matrix'),use_container_width=True)
        pairs=[]
        cols=corr.columns
        for i in range(len(cols)):
            for j in range(i+1,len(cols)): pairs.append((cols[i],cols[j],corr.iloc[i,j]))
        for a,b,r in sorted(pairs,key=lambda x:abs(x[2]),reverse=True)[:5]: st.write(f'• **{a} ↔ {b}** — correlation **{r:.2f}**')
    else: st.info('At least two numeric columns are needed.')

with t3:
    st.subheader('🚨 Isolation Forest Anomaly Detection')
    nums=df.select_dtypes(include=np.number).columns.tolist()
    if len(nums)>=2:
        X=df[nums].replace([np.inf,-np.inf],np.nan).fillna(df[nums].median())
        model=IsolationForest(contamination=0.08,random_state=42); labels=model.fit_predict(X)
        result=df.copy(); result['Anomaly']=np.where(labels==-1,'🚨 Anomaly','Normal'); result['Anomaly Score']=model.decision_function(X)
        an=result[result.Anomaly=='🚨 Anomaly']
        a,b=st.columns(2); a.metric('Records Investigated',len(result)); b.metric('Anomalies Detected',len(an))
        st.plotly_chart(px.scatter(result,x=nums[0],y=nums[1],color='Anomaly',title='Normal vs Anomalous Records'),use_container_width=True)
        st.dataframe(an,use_container_width=True)
    else: st.info('At least two numeric columns are needed.')

with t4:
    st.subheader('🤖 Random Forest Predictive Analysis')
    target=st.selectbox('Select a target column',df.columns)
    X=df.drop(columns=[target]).select_dtypes(include=np.number)
    y=df[target]
    valid=y.notna()
    if X.shape[1]>=1 and y[valid].nunique()>=2 and valid.sum()>=20:
        X=X.loc[valid].replace([np.inf,-np.inf],np.nan).fillna(X.median()); y2=y.loc[valid]
        y_model=y2.astype(str) if (not pd.api.types.is_numeric_dtype(y2) or y2.nunique()<=10) else pd.qcut(y2,q=3,labels=['Low','Medium','High'],duplicates='drop')
        if y_model.nunique()>=2:
            Xtr,Xte,ytr,yte=train_test_split(X,y_model,test_size=.25,random_state=42,stratify=y_model)
            clf=RandomForestClassifier(n_estimators=150,random_state=42); clf.fit(Xtr,ytr); pred=clf.predict(Xte)
            st.metric('Random Forest Accuracy',f'{accuracy_score(yte,pred):.1%}')
            imp=pd.DataFrame({'Feature':X.columns,'Importance':clf.feature_importances_}).sort_values('Importance',ascending=False)
            st.plotly_chart(px.bar(imp,x='Importance',y='Feature',orientation='h',title='Feature Importance'),use_container_width=True)
            st.dataframe(imp,use_container_width=True)
        else: st.warning('The target does not contain enough distinct classes.')
    else: st.info('Choose a target with at least 2 classes and enough rows.')

with t5:
    st.subheader('💡 Automatic Evidence-Backed Findings')
    st.info(f'💡 The cleaned dataset contains **{len(df):,} records** across **{df.shape[1]} columns**.')
    nums=df.select_dtypes(include=np.number)
    if nums.shape[1]>=2:
        corr=nums.corr().abs(); np.fill_diagonal(corr.values,0); s=corr.stack().sort_values(ascending=False)
        if len(s):
            (a,b),v=s.index[0],s.iloc[0]; st.info(f'💡 The strongest numeric relationship is **{a} ↔ {b}**, with absolute correlation **{v:.2f}**.')
        X=nums.replace([np.inf,-np.inf],np.nan).fillna(nums.median()); labels=IsolationForest(contamination=.08,random_state=42).fit_predict(X); n=int((labels==-1).sum())
        st.info(f'💡 Isolation Forest flags **{n} records ({n/len(df):.1%})** as potential anomalies.')
    buf=BytesIO(); df.to_csv(buf,index=False)
    st.download_button('⬇️ Download Cleaned CSV',buf.getvalue(),'mysterylens_cleaned.csv','text/csv')

st.divider(); st.caption('MysteryLens AI • ALG-DATA-01 Data Investigation App')
