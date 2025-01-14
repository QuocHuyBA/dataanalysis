import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import plotly.express as px  # pip install plotly-express
import plotly.graph_objects as go  # pip install plotly
import base64  # Standard Python Module
from io import StringIO, BytesIO  # Standard Python Module
import openai  # pip install openai
from prophet import Prophet  # pip install prophet

# Set page configuration
st.set_page_config(page_title='Excel Analysis with AI')

# Define CSS for light and dark modes
light_mode_css = """
<style>
body {
    background-color: white;
    color: black;
}
</style>
"""

dark_mode_css = """
<style>
body {
    background-color: #2E2E2E;
    color: white;
}
</style>
"""


# Set your OpenAI API key
openai.api_key = 'your_actual_openai_api_key'  # Replace with your actual OpenAI API key

def ask_openai(question, context):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def forecast(df, date_col, value_col):
    df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast

st.title('Excel Plotter with AI 📈')
st.subheader('Send me your Excel file')

uploaded_file = st.file_uploader('Choose a XLSX file', type='xlsx')
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # Check if 'CDATE' column exists
    if 'CDATE' in df.columns:
        # Convert CDATE to datetime format
        df['CDATE'] = pd.to_datetime(df['CDATE'], format='%Y%m%d')
    else:
        st.error("The 'CDATE' column does not exist in the uploaded file.")
    
    tab1, tab2 = st.tabs(["Data Analysis", "Ask AI"])

    with tab1:
        st.subheader('Data Preview')
        st.dataframe(df)
        st.markdown('---')
        st.subheader('Data Analysis')
        
        report_type = st.selectbox('What kind of report you want?', ['Data Visual', 'Comparison'])
        
        groupby_columns = [col for col in df.columns if 'Cd' in col or 'Logistic' in col or 'Cn' in col]
        groupby_column = st.multiselect('What would you like to analyse?', groupby_columns)

        output_columns_amt = [col for col in df.columns if 'Amt' in col]
        output_columns_qty = [col for col in df.columns if 'Qty' in col]
        
        if report_type == 'Data Visual':
            date_filter = st.selectbox('View by', ['Day', 'Month', 'Year'])
            
            if date_filter == 'Day':
                df['DateFilter'] = df['CDATE'].dt.date
            elif date_filter == 'Month':
                df['DateFilter'] = df['CDATE'].dt.to_period('M').astype(str)
            elif date_filter == 'Year':
                df['DateFilter'] = df['CDATE'].dt.to_period('Y').astype(str)
            
            filter_values = {}
            for col in groupby_column:
                filter_values[col] = st.selectbox(f'Select {col} value to filter chart', df[col].unique().tolist())
        else:
            start_date = st.date_input('Start date', df['CDATE'].min())
            end_date = st.date_input('End date', df['CDATE'].max())
            df = df[(df['CDATE'] >= pd.to_datetime(start_date)) & (df['CDATE'] <= pd.to_datetime(end_date))]
            
            filter_values = {}
            for col in groupby_column:
                filter_values[col] = st.multiselect(f'Select {col} values to filter chart', df[col].unique().tolist())
        
        tab3, tab4 = st.tabs(["Data Visual", "Data Forecast"])

        with tab3:
            if report_type == 'Data Visual':
                if not groupby_column:
                    df_grouped = df.groupby(by=['DateFilter'], as_index=False).sum(numeric_only=True)
                elif groupby_column and (output_columns_amt or output_columns_qty):
                    df_grouped = df.groupby(by=groupby_column + ['DateFilter'], as_index=False).sum(numeric_only=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(df_grouped)
                        
                        filtered_df = df_grouped
                        for col, value in filter_values.items():
                            if value:
                                filtered_df = filtered_df[filtered_df[col] == value]
                    
                    with col2:
                        if any(filter_values.values()):
                            fig = go.Figure()
                            
                            if output_columns_amt:
                                for col in output_columns_amt:
                                    fig.add_trace(go.Scatter(
                                        x=filtered_df['DateFilter'],
                                        y=filtered_df[col],
                                        mode='lines',
                                        name=col,
                                        yaxis='y1'
                                    ))
                            
                            if output_columns_qty:
                                for col in output_columns_qty:
                                    fig.add_trace(go.Scatter(
                                        x=filtered_df['DateFilter'],
                                        y=filtered_df[col],
                                        mode='lines',
                                        name=col,
                                        yaxis='y2'
                                    ))
                            
                            fig.update_layout(
                                title=f'Values by {", ".join(groupby_column)} and Date',
                                xaxis_title='Date',
                                yaxis=dict(
                                    title='Amt',
                                    titlefont=dict(
                                        color='blue'
                                    ),
                                    tickfont=dict(
                                        color='blue'
                                    )
                                ),
                                yaxis2=dict(
                                    title='Qty',
                                    titlefont=dict(
                                        color='red'
                                    ),
                                    tickfont=dict(
                                        color='red'
                                    ),
                                    overlaying='y',
                                    side='right'
                                ),
                                legend=dict(
                                    x=0,
                                    y=1.0,
                                    bgcolor='rgba(255, 255, 255, 0)',
                                    bordercolor='rgba(255, 255, 255, 0)'
                                )
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            st.write("Please select values to filter the chart.")
                else:
                    st.error("The required columns do not exist in the uploaded file.")
            
            elif report_type == 'Comparison':
                if not groupby_column:
                    df_grouped = df.groupby(by=['DateFilter'], as_index=False).sum(numeric_only=True)
                elif groupby_column and (output_columns_amt or output_columns_qty):
                    df_grouped = df.groupby(by=groupby_column, as_index=False).sum(numeric_only=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(df_grouped)
                        
                        filtered_df = df_grouped
                        for col, values in filter_values.items():
                            if values:
                                filtered_df = filtered_df[filtered_df[col].isin(values)]
                    
                    with col2:
                        if any(filter_values.values()):
                            fig = go.Figure()
                            
                            for col in output_columns_amt + output_columns_qty:
                                fig.add_trace(go.Bar(
                                    x=filtered_df[groupby_column[0]],
                                    y=filtered_df[col],
                                    name=col
                                ))
                            
                            fig.update_layout(
                                title=f'Comparison by {", ".join(groupby_column)}',
                                xaxis_title=groupby_column[0],
                                yaxis_title='Values',
                                legend=dict(
                                    x=0,
                                    y=1.0,
                                    bgcolor='rgba(255, 255, 255, 0)',
                                    bordercolor='rgba(255, 255, 255, 0)'
                                ),
                                barmode='group'
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            st.write("Please select values to filter the chart.")
                else:
                    st.error("The required columns do not exist in the uploaded file.")

        with tab4:
            st.subheader('Forecast:')
            date_col = st.selectbox('Select the date column', df.columns)
            value_col = st.selectbox('Select the value column to forecast', df.columns)
            if st.button('Generate Forecast'):
                forecast_df = forecast(df, date_col, value_col)
                st.dataframe(forecast_df)
                fig_forecast = px.line(forecast_df, x='ds', y='yhat', title='Forecast')
                st.plotly_chart(fig_forecast)

    with tab2:
        st.subheader('Ask AI:')
        st.markdown('<div class="chat-container" id="chat-container"></div>', unsafe_allow_html=True)
        user_input = st.text_area('Type your message:', key='user_input')
        if st.button('Send'):
            context = df.to_string()
            answer = ask_openai(user_input, context)
            
            st.markdown(f"""
            <script>
            var chatContainer = document.getElementById('chat-container');
            chatContainer.innerHTML += '<div class="chat-bubble user">{user_input}</div>';
            chatContainer.innerHTML += '<div class="chat-bubble ai'>{answer}</div>';
            chatContainer.scrollTop = chatContainer.scrollHeight;
            </script>
            """, unsafe_allow_html=True)
