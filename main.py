import os
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.anomaly import AnomalyExperiment
from PIL import Image
import google.generativeai as genai

# Configuring the Streamlit app
st.set_page_config(layout="wide", page_title="benjiTable DS", page_icon="ðŸ¤–")

# Apply custom CSS for enhanced background and lighter sidebar color
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,pink, #f5f7fa, skyblue, #f5f7fa);
    color: #333;
}
[data-testid="stSidebar"] {
    background: linear-gradient(skyblue, blue) !important;
    color: white;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}
body {
    font-family: "Source Sans Pro", sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #2c3e50;
}
.stMarkdown p {
    color: #333;
}
.stDataFrame {
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.stProgress > div > div > div {
    background-color: #4CAF50;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Google API Key Configuration for benGPT
GOOGLE_API_KEY = 'AIzaSyCJzha8fEyQg-0F6jxHnswpEreMzxisyQw'  # Replace with your Google API Key
genai.configure(api_key=GOOGLE_API_KEY)
geminiModel = genai.GenerativeModel("gemini-1.5-flash")

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "creativity" not in st.session_state:
    st.session_state.creativity = 0

def replace_key_in_dict_list(dict_list):
    prefix1 = '**Me**:'
    prefix1_len = len(prefix1)
    prefix2 = '**benji**:'
    prefix2_len = len(prefix2)
    
    new_list = []
    for d in dict_list:
        new_dict = {}
        for key, value in d.items():
            if value == 'assistant':
                new_dict['role'] = "model"
            elif key == 'contents':
                if value.startswith(prefix1):
                    value = value[prefix1_len:].strip()
                elif value.startswith(prefix2):
                    value = value[prefix2_len:].strip()
                new_dict['parts'] = value
            else:
                new_dict[key] = value
        new_list.append(new_dict)
    return new_list

def ChatBot() -> None:
    ''' Chatbot Page '''
    st.header("benGPT", divider="orange")
    st.markdown(
        """
        **WELCOME TO THE WORLD OF PRO-HUMANOID INTELLIGENCE**

        benGPT will provide you with **accurate**, **informative**, and **insightful** responses to your queries.
        """
    )

    image1 = Image.open("image3.png")
    st.image(image1, width=300)
    st.subheader("Ask benGPT anything")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["contents"])

    prompt: str = st.chat_input("Message benGPT...")
    if prompt:
        prompt = f"**You**: {prompt}"
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "contents": prompt})

        chat = geminiModel.start_chat(history=replace_key_in_dict_list(st.session_state.messages))
        response = chat.send_message(prompt, 
                                    generation_config=genai.types.GenerationConfig(
                                    candidate_count=1,
                                    temperature=st.session_state.creativity,
                                    ),)
  
        response = f"**benGPT**: \n{response.text}"

        with st.chat_message("assistant", avatar="image5.png"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "contents": response})

        st.session_state.history.extend([{"role": "user", "contents": prompt}, {"role": "assistant", "contents": response}])

# Navigation bar
st.markdown("<h1 style='text-align: center;'>benjiTable DS</h1>", unsafe_allow_html=True)
image1 = Image.open("mypiclogo.png")
st.image(image1)
st.markdown("<p style='text-align: center; color: #666;'><em>Your AI-powered ML Solutions</em></p>", unsafe_allow_html=True)
choices = st.radio("**Navigation Menu**", ["Upload Dataset", "Explore Data", "Build Models", "benGPT Chatbot"], horizontal=True)




# Handle page routing
if choices == 'Upload Dataset':
    
    st.title('Upload Dataset For Modeling & Analysis')
    st.write("Begin exploring and building machine learning models by uploading your dataset.")
    file = st.file_uploader("**Upload a CSV file**", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=None)
        st.success('Dataset uploaded successfully!')
        st.write("### Dataset Preview:")
        st.dataframe(df)
    else:
        st.warning("Please upload a CSV file to continue.")

elif choices == 'Explore Data':
    st.title('Exploratory Data Analysis (EDA)')
    st.write("Generate in-depth exploratory reports to understand the data before building any models.")
    
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
        st.write("### Dataset Overview:")
        st.dataframe(df)

        if st.button("Generate Report", key="generate_report"):
            with st.spinner("Generating detailed report... This may take a moment."):
                profile = ProfileReport(df)
                st_profile_report(profile)
        else:
            st.info("Click the 'Generate Report' button above to create an exploratory data analysis report.")
    else:
        st.error("No dataset found. Please upload a dataset first.")

elif choices == 'Build Models':
    st.title('Machine Learning Modeling')
    st.write("Select a machine learning task to proceed with automated model training and evaluation.")
    
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')

        exp_choice = st.radio("Choose a Machine Learning Task", ["Classification", "Regression", "Clustering", "Anomaly Detection"])

        if exp_choice == "Classification":
            st.header('Classification Task')
            st.write("In this section, you'll train classification models to predict categorical outcomes based on your dataset.")
            X = st.multiselect("**Select Features for Training**", df.columns)
            y = st.selectbox("**Select Target Variable**", df.columns)

            if X and y:
                data = df[X + [y]]
                if st.button('Start Training'):
                    st.write("Running Classification Experiment... â³")
                    cls = ClassificationExperiment()
                    cls.setup(data, target=y)
                    setup_df = cls.pull()
                    st.write("### Model Setup Summary:")
                    st.dataframe(setup_df)

                    st.write("### Comparing Models... ðŸ“ˆ")
                    best_model = cls.compare_models()
                    st.write("### Best Model:")
                    st.dataframe(cls.pull())

        elif exp_choice == "Regression":
            st.header('Regression Task')
            st.write("In this section, you'll train regression models to predict continuous outcomes based on your dataset.")
            X = st.multiselect("**Select Features for Training**", df.columns)
            y = st.selectbox("**Select Target Variable**", df.columns)

            if X and y:
                data = df[X + [y]]
                if st.button('Start Training'):
                    st.write("Running Regression Experiment... â³")
                    reg = RegressionExperiment()
                    reg.setup(data, target=y)
                    setup_df = reg.pull()

                    st.write("### Model Setup Summary:")
                    st.dataframe(setup_df)

                    st.write("### Comparing Models... ðŸ“ˆ")
                    best_model = reg.compare_models()
                    st.write("### Best Model:")
                    st.dataframe(reg.pull())

        elif exp_choice == "Clustering":
            st.header('Clustering Task')
            st.write("In this section, you'll perform clustering to find patterns or groupings in your dataset.")
            X = st.multiselect("**Select Features for Clustering**", df.columns)

            if X:
                data = df[X]
                if st.button('Start Clustering'):
                    st.write("Running Clustering Experiment... â³")
                    clus = ClusteringExperiment()
                    clus.setup(data)
                    setup_df = clus.pull()

                    st.write("### Clustering Setup Summary:")
                    st.dataframe(setup_df)

                    st.write("### Comparing Clustering Models... ðŸ“ˆ")
                    best_model = clus.compare_models()
                    st.write("### Best Clustering Model:")
                    st.dataframe(clus.pull())

        elif exp_choice == "Anomaly Detection":
            st.header('Anomaly Detection Task')
            st.write("In this section, you'll perform anomaly detection to identify outliers in your dataset.")
            X = st.multiselect("**Select Features for Anomaly Detection**", df.columns)

            if X:
                data = df[X]
                if st.button('Start Anomaly Detection'):
                    st.write("Running Anomaly Detection Experiment... â³")
                    anom = AnomalyExperiment()
                    anom.setup(data)
                    setup_df = anom.pull()

                    st.write("### Anomaly Detection Setup Summary:")
                    st.dataframe(setup_df)

                    st.write("### Comparing Anomaly Detection Models... ðŸ“ˆ")
                    best_model = anom.compare_models()
                    st.write("### Best Anomaly Detection Model:")
                    st.dataframe(anom.pull())

    else:
        st.error("No dataset found. Please upload a dataset first.")

elif choices == 'benGPT Chatbot':
    ChatBot() 
