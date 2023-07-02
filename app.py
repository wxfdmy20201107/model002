import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

@st.cache_data  # 添加缓存器
def load_data():
    df = pd.read_excel('textdata.xls')
    return df

@st.cache_data  # 添加缓存器
def train_model():
    df = load_data()
    X = df[["age", "drug", "otherdisease"]]
    y = df["flag"]
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

# 加载数据
df = load_data()
# 建立模型
clf = train_model()

# Title
st.header("肺克感染病人死亡风险评估")

# Input bar 2
age = st.number_input("输入年龄")

# Dropdown input
drug = st.selectbox("是否多次使用抗生素", ("0", "1"))
otherdisease = st.selectbox("是否存在基础疾病",("0","1"))

# If button is pressed
if st.button("点击预测"):
    # Store inputs into dataframe
    X_text = pd.DataFrame([[age, drug,otherdisease]],
                     columns=["age", "drug","otherdisease"])

    # Get prediction1
    Y1 = clf.predict(X_text)[0]
    if Y1 == 0:
        outcome1 = "存活"
    elif Y1 == 1:
        outcome1 = "死亡"
    else:
        outcome1 = "未知"
   
    # Get prediction2
    X_text2 = pd.DataFrame([[age, drug]],
                     columns=["age", "drug"])
    Y2 = 3*X_text2[age]+126*X_text2[drug]
    if Y2 < 226.5:
        outcome2 = "存活"
    elif Y2 >= 226.5:
        outcome2 = "死亡"
    else:
        outcome2 = "未知"
   
    # Output prediction
    st.text(f"该肺克感染病人是否死亡？机器学习的答案是： {outcome1}")
    st.text(f"该肺克感染病人是否死亡？传统方法的答案是： {outcome2}")



