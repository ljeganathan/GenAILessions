import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Book Store", layout="wide")

st.title("📚 Books Scraped Using Playwright")

# Flask API URL
API_URL = "http://127.0.0.1:5000/books"

if st.button("Load Books"):

    with st.spinner("Fetching books..."):

        response = requests.get(API_URL)

        if response.status_code == 200:

            data = response.json()

            # Convert to dataframe
            df = pd.DataFrame(data)

            # Show table
            st.dataframe(df, use_container_width=True)

            # Metrics
            st.success(f"Total Books Loaded: {len(df)}")

        else:
            st.error("Failed to fetch data from API")