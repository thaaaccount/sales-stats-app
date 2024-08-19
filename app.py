import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Streamlit app
st.title("Sales Data Analysis")

# Use an expander for the file upload widget
with st.expander("Upload your CSV file", expanded=True):
    uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data preprocessing
    df = df[df['UnitPrice'] >= 0]
    df = df[df['UnitPrice'] != 315.93]
    
    df['UnitPrice'] = df['UnitPrice'].astype(float)
    df['Quantity'] = df['Quantity'].astype(float)
    df['UnitCost'] = df['UnitCost'].astype(float)
    df['Revenue'] = df['UnitPrice'] * df['Quantity']
    df['OrderDate_Year'] = df['OrderDate_Year'].astype(str)

    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['UnitCost'] = pd.to_numeric(df['UnitCost'], errors='coerce')

    df = df.dropna(subset=['UnitPrice', 'Quantity'])

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Raw Data", "Summary Statistics", "Quantity Distribution", "Quantity VS Unit Price", "Monthly Revenue Trend", "Python Price vs Revenue Analysis"])

    with tab1:
        st.subheader("Raw Data")
        st.dataframe(df)

    with tab2:
        # Summary statistics
        summary_stats = df.groupby('LongItem').agg({
            'Quantity': ['sum', 'mean', 'median', 'std', 'min', 'max'],
            'UnitPrice': ['mean', 'median', 'std', 'min', 'max'],
            'Revenue': ['sum', 'mean', 'median', 'std', 'min', 'max']
        }).reset_index()

        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats.rename(columns={'LongItem_': 'LongItem'}, inplace=True)

        currency_columns = ['UnitPrice_mean', 'UnitPrice_min', 'UnitPrice_max', 'Revenue_median',
                            'Revenue_sum', 'Revenue_mean', 'Revenue_min', 'Revenue_max', 'UnitPrice_median']

        quantity_columns = ['Quantity_sum', 'Quantity_mean', 'Quantity_std', 'Quantity_min',
                            'Quantity_max', 'UnitPrice_std', 'Revenue_std', 'Quantity_median']

        # Apply formatting
        summary_stats[currency_columns] = summary_stats[currency_columns].applymap(lambda x: "${:,.2f}".format(x))
        summary_stats[quantity_columns] = summary_stats[quantity_columns].applymap(lambda x: "{:,.2f}".format(x))

        st.subheader("Summary Statistics")
        st.dataframe(summary_stats)

    with tab3:
        st.subheader("Quantity Distribution per LongItem")
        fig = px.box(df, x='LongItem', y='Quantity', title='Quantity Distribution per LongItem')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Quantity VS Unit Price")
        fig = px.scatter(df, x='UnitPrice', y='Quantity', color='LongItem', title='Quantity vs UnitPrice')
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Monthly Revenue Trend")
        df['OrderDate'] = pd.to_datetime(df['OrderDate_Year'].astype(str) + '-' + df['OrderDate_Month'].astype(str) + '-01')
        monthly_revenue = df.groupby(['OrderDate', 'LongItem']).agg({'Revenue': 'sum'}).reset_index()
        fig = px.line(monthly_revenue, x='OrderDate', y='Revenue', color='LongItem', title='Monthly Revenue Trend per LongItem')
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("Python Price vs Revenue Analysis")

        # Use the model you provided
        pd.options.display.float_format = '{:.2f}'.format

        df_agg = df.groupby(['LongItem', 'UnitPrice']).agg({
            'Quantity': 'sum',
            'Revenue': 'sum'
        }).reset_index()

        # Creating a Revenue column again after aggregation
        df_agg['Revenue'] = df_agg['UnitPrice'] * df_agg['Quantity']

        gams = {}
        for item in df_agg['LongItem'].unique():
            try:
                # Load pre-trained model from .pkl file
                with open(f'{item}_gam_model.pkl', 'rb') as f:
                    gams[item] = pickle.load(f)
            except FileNotFoundError:
                st.warning(f"No pre-trained model found for {item}. Skipping...")
                continue

        df_agg['Predicted_Revenue'] = np.nan
        for item in gams:
            mask = df_agg['LongItem'] == item
            df_agg.loc[mask, 'Predicted_Revenue'] = gams[item].predict(df_agg[mask][['UnitPrice']])

        fig = px.scatter(
            df_agg,
            x='UnitPrice',
            y='Predicted_Revenue',
            color='LongItem',
            opacity=0.6,
            template="none",
            title='GAM Predicted Revenue: Price vs Revenue Analysis',
            width=800,
            height=600,
        ).update_traces(
            marker=dict(size=7),
            hoverlabel=dict(font=dict(size=10)),
        ).update_layout(
            legend_title_text='LongItem',
            title_font=dict(size=16),
            legend_font=dict(size=10),
        ).update_xaxes(
            title_text='UnitPrice',
            title_font=dict(size=10),
            tickfont=dict(size=10),
        ).update_yaxes(
            title_text='Predicted Revenue',
            title_font=dict(size=10),
            tickfont=dict(size=10),
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Please upload a CSV file to view the data and summary statistics.")
