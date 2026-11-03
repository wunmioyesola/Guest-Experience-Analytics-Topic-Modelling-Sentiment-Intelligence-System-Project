import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Guest Experience Sentiment Analysis Dashboard", layout="wide")

st.title("Guest Experience & Sentiment Analysis Dashboard")
st.write("Upload your customer review CSV file to analyze sentiment in batch.")

# file uploader widget
uploaded_file = st.file_uploader("choose a CSV file", type="csv")

if uploaded_file is not None:
    # Display preview of raw data
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_raw.head())

    if st.button("Run Sentiment Analysis"):
        try:
            # Reset file pointer and send to FastAPI backend
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}

            # fastAPI endpoint URL
            response = requests.post("http://localhost:8000/predict/batch", files=files)

            if response.status_code == 200:
                result_data = response.json()
                
                # check if the backend returned an error dictionary
                if isinstance(result_data, dict) and "error" in result_data:
                    st.error(f"Backend Error: {result_data['error']}")
                else:
                    df_results = pd.DataFrame(result_data)
                    st.success("Batch prediction completed successfully!")
                    st.dataframe(df_results)

                # Display full results table
                # st.subheader("Detailed Results")
                # st.dataframe(df_results)
            
                # Download button results
                    csv_export = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_export,
                        file_name="sentiment_result.csv",
                        mime="text/csv",
                    )
            else:
                st.error(f"Server error: Status code {response.status_code}")


            #else:
                #st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI backend: {str(e)}")

