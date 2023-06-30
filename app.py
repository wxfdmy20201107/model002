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
    X = df[["gender", "age", "drug", "otherdisease"]]
    y = df["flag"]
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

# 加载数据
df = load_data()
# 建立模型
clf = train_model()

# Title
st.header("耐药肺克病人死亡风险评估")

# Dropdown input
gender = st.selectbox("选择性别",("0","1"))

# Input bar 2
age = st.number_input("输入年龄")

# Dropdown input
drug = st.selectbox("是否多次使用抗生素", ("0", "1"))
otherdisease = st.selectbox("是否存在基础疾病",("0","1"))

# If button is pressed
if st.button("点击预测"):
    # Store inputs into dataframe
    X_text = pd.DataFrame([[gender, age, drug,otherdisease]],
                     columns=["gender", "age", "drug","otherdisease"])

    # Get prediction
    Y = clf.predict(X_text)[0]
    if Y == 0:
        outcome = "存活"
    elif Y == 1:
        outcome = "死亡"
    else:
        outcome = "未知"

    # Output prediction
    st.text(f"该耐药肺克病人是否死亡？答案是： {outcome}")



