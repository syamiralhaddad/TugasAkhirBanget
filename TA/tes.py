import streamlit as st
import pandas as pd
import plotly.express as px

# Create sample data
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [15, 25, 35, 25]
})

data['Remaining'] = 100 - data['Value']

# Create a doughnut chart with Plotly
fig = px.pie(data, values='Value', names='Category', 
             hole=0.4, title='Doughnut Chart Example (Plotly)', 
             labels={'Value': 'Original Percentage', 'Remaining': 'Remaining Percentage'})

# Display the chart in Streamlit
st.title("Doughnut Chart Example (Plotly)")
st.plotly_chart(fig)

# Define the values for the boxes
box_values = {
    "Box 1": 10,
    "Box 2": 20,
    "Box 3": 30,
    "Box 4": 40,
    "Box 5": 50
}

# Display the boxes with titles
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.subheader("Box 1")
    st.write(box_values["Box 1"])
with col2:
    st.subheader("Box 2")
    st.write(box_values["Box 2"])
with col3:
    st.subheader("Box 3")
    st.write(box_values["Box 3"])
with col4:
    st.subheader("Box 4")
    st.write(box_values["Box 4"])
with col5:
    st.subheader("Box 5")
    st.write(box_values["Box 5"])

