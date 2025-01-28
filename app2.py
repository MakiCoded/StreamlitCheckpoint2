import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title = "ML Model Prediction App 🈸",
    page_icon = "🔮",
    layout = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "This is a header. This is an extremely cool app!"
    }
)

# Sidebar for input fields
st.sidebar.title("ML Model Prediction App")
st.sidebar.write("Enter the feature values in the dropdowns on the main page.")

# Contact Info
st.sidebar.subheader("Contact Info ℹ")
st.sidebar.write("Phone ☎: +234 70 4001 3181")
st.sidebar.write("Email 📧: maccyosakwe@gmail.com")

# Tips
st.sidebar.subheader("Tips ➡")
st.sidebar.write("1. Ensure your data is clean and well-prepared.")
st.sidebar.write("2. Normalize feature values for better model performance.")
st.sidebar.write("3. Validate model performance with a test set.")

# Company Details
st.sidebar.subheader("About Our Company 🎯")
st.sidebar.write("We are a pioneering company specializing in cutting-edge AI solutions, dedicated to leveraging artificial intelligence to address real-world challenges.")

# About
st.sidebar.subheader("About This App 🆎")
st.sidebar.write("This app enables users to enter feature values and receive predictions from a trained machine learning model.")

# Reasons to Own a Bank Account
st.sidebar.subheader("Why Should You Have a Bank Account? 🏦")
st.sidebar.write("1. A safe and reliable way to keep your money.")
st.sidebar.write("2. Convenient access to your money.")
st.sidebar.write("3. Chance to grow your money with interest.")
st.sidebar.write("4. Availability of financial services like loans and credit.")

# Add a colorful header and instructions
st.markdown('<h1 style="color: darkblue;">ML Model Prediction App</h1>', unsafe_allow_html=True)
st.write("This is a basic ML prediction app that estimates the target variable using the provided input features.")
st.write("The model is trained using the Financial Inclusion in Africa dataset.")
st.write('The target variable, "bank_account," is a binary indicator that shows whether a respondent has a bank account.')
st.write("Select the feature values from the dropdown menus and click the predict button to receive the prediction.")

@st.cache_resource
def load_data():
    RF = pickle.load(open("model.pkl", "rb"))
    return RF

# Load the model
model = load_data()


gender_of_respondent = {
    'Female': 0,
    'Male': 1
}

relationship_with_head = {
    'Child': 0,
    'Head of Household': 1,
    'Other non-relatives': 2,
    'Other relative': 3,
    'Parent': 4,
    'Spouse': 5
}

marital_status = {
    'Divorced/Seperated': 0,
    'Dont know': 1,
    'Married/Living together': 2,
    'Single/Never Married': 3,
    'Widowed': 4
}

education_level = {
    'No formal education': 0,
    'Other/Dont know/RTA': 1,
    'Primary education': 2,
    'Secondary education': 3,
    'Tertiary education': 4,
    'Vocational/Specialised training': 5
}

job_type = {
    'Dont Know/Refuse to answer': 0,
    'Farming and Fishing': 1,
    'Formally employed Government': 2,
    'Formally employed Private': 3,
    'Government Dependent': 4,
    'Informally employed': 5,
    'No Income': 6,
    'Other Income': 7,
    'Remittance Dependent': 8,
    'Self employed': 9
}

country = {
    'Kenya': 0,
    'Rwanda': 1,
    'Tanzania': 2,
    'Uganda': 3
}
location_type = {
    'Rural': 0,
    'Urban': 1
}

cellphone_access = {
    'No': 0,
    'Yes': 1
}

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    user_country = country[st.selectbox("Country", list(country.keys()))]
    year = st.number_input("Year", min_value=2016, max_value=2018, value=2016)
    user_location = location_type[st.selectbox("Location Type", list(location_type.keys()))]
    user_cellphone = cellphone_access[st.selectbox("Cellphone Access", list(cellphone_access.keys()))]
    
    
with col2:
    user_house_hold_size = st.number_input("Household Size", min_value=1, max_value=21, value=5)
    user_age_of_respondent = st.number_input("Age of Respondent", min_value=16, max_value=100, value=30)
    user_gender =gender_of_respondent[st.selectbox("Gender of Respondent", list(gender_of_respondent.keys()))]
  
with col3:
    user_relationship_with_head =relationship_with_head[st.selectbox("Relationship with Head", list(relationship_with_head.keys()))]
    user_marital_status = marital_status[st.selectbox("Marital Status", list(marital_status.keys()))]
    user_education_level = education_level[st.selectbox("Education Level", list(education_level.keys()))]
    user_job_type = job_type[st.selectbox("Job Type", list(job_type.keys()))]

# Validation button on the main page
if st.button("Predict"):
    # features = np.array([user_country, year, user_location, user_cellphone, user_house_hold_size, user_age_of_respondent, user_gender, user_relationship_with_head, user_marital_status,user_education_level,user_job_type])
    
    input_data = {
        "country": user_country,
        "year": year,
        "location_type": user_location,
        "cellphone_access": user_cellphone,
        "household_size": user_house_hold_size,
        "age_of_respondent": user_age_of_respondent,
        "gender_of_respondent": user_gender,
        "relationship_with_head": user_relationship_with_head,
        "marital_status": user_marital_status,
        "education_level": user_education_level,
        "job_type": user_job_type

    }
   #     # Create input data
    input_features = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_features)[0]

    if prediction == 1:
        st.success("The respondent is likely to have a bank account.")
    else:
        st.warning("The respondent is unlikely to have a bank account.")
   

# Add a footer with additional information        
st.markdown("---")
st.markdown("### Additional Information")
st.write("This app is powered by a machine learning model trained on the Financial Inclusion in Africa dataset.")
st.write("For more information about the dataset, visit the [dataset page](https://www.kaggle.com/competitions/financial-inclusion-in-africa).")
st.write("For any inquiries or support, please contact us at the provided contact information in the sidebar.")

# Footer with additional links and resources
st.markdown("---")
st.markdown("### Useful Links")
st.write("[Streamlit Documentation](https://docs.streamlit.io/)")
st.write("[Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)")
st.write("[NumPy Documentation](https://numpy.org/doc/)")
st.write("[Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)")

# Footer with copyright information
st.markdown("---")
st.markdown("© 2023 Macdonald's Library. All rights reserved.")