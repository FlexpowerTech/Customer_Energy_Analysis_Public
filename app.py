import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------------------
# Layout and Header Elements
# ---------------------------
st.markdown(
    """
    <style>
    .title-container {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow effect */
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .title-container img {
        width: 100px; /* Adjust the logo size */
    }
    .title-container h1 {
        margin: 10px 0 0 0;
        font-size: 38px;
        color: #333333; /* Title color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("LogoFlexpower.png", width=100)

st.markdown(
    """
    <div class="title-container">
        <h1>Calculate energy savings based on solar curtailment</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "Did you feed in power in the past period?  \n"
    "You might have provided power to the grid while the day-ahead price was negative, this results in you paying for delevering power.  \n"
    "Upload your hourly energy usage and run the calculations to find out!  \n"
    "Explanation: This tool uses the day-ahead prices from the ENTSOE platform API for the Netherlands.  \n"
    "The day-ahead prices are updated monthly; the current dataset is 01-01-2023 till 17-02-2025.  \n"
    "The template file has an hourly format, so upload your meter data using the template.  \n"
    "Either match the time stamps to those of the template or use the same format in dd-mm-yyyy HH:SS."
)

st.subheader("Let's analyse your energy profile and potential savings by curtailing your solar system")

# ----------------------------
# Download Template CSV
# ----------------------------
st.write(
    "Download the template file below and paste your energy profile in. "
    "Make sure to match the time formatting."
)
with open("MeterDataKlant.csv", "rb") as file:
    st.download_button(
        label="Download the Template CSV",
        data=file,
        file_name="MeterDataKlant.csv",
        mime="text/csv"
    )

# ----------------------------
# File Upload and Configuration
# ----------------------------
uploaded_file = st.file_uploader("Upload your Hourly MeterDataKlant.csv", type=["csv"])
afslag_VHP = st.number_input(
    "Afslag adjustment (€/kWh)",
    min_value=-0.05, max_value=0.05,
    value=-0.0185, step=0.0005, format="%.4f"
)

# Use session_state to store the processed DataFrame
if "merged_df" not in st.session_state:
    st.session_state.merged_df = None

# ----------------------------
# Calculation and Data Processing
# ----------------------------
if uploaded_file is not None:
    # Run Calculation button OR if the data is already processed
    if st.button("Run Calculation") or st.session_state.merged_df is not None:
        # Process the data only if it hasn't been done already
        if st.session_state.merged_df is None:
            df_Epex = pd.read_csv("Epex_NL_API_data.csv")
            df_klantdata = pd.read_csv(uploaded_file)
            df_klantdata.rename(
                columns={'Timestamp': 'DateTime', 'Powerconsumption (kWh)': 'Energy_Consumption_kWh'},
                inplace=True
            )

            # Convert DateTime columns to datetime objects
            df_Epex['DateTime'] = pd.to_datetime(df_Epex['DateTime'], utc=True).dt.tz_localize(None)
            df_klantdata['DateTime'] = pd.to_datetime(df_klantdata['DateTime'], dayfirst=True, utc=True).dt.tz_localize(None)

            # Merge the data
            merged_df = df_klantdata.merge(df_Epex, on="DateTime", how="inner")

            # Data transformations
            merged_df['Epex incl afslag Price_eur/kwh'] = merged_df['Epex Price_eur/kwh'] + afslag_VHP
            merged_df['Result eur of grid input incl afslag'] = (
                merged_df['Epex incl afslag Price_eur/kwh'] * merged_df['Energy_Consumption_kWh']
            )
            merged_df['Kwh of negative grid input'] = merged_df.apply(
                lambda row: 0 if row['Energy_Consumption_kWh'] > 0 else -row['Energy_Consumption_kWh'],
                axis=1
            )
            merged_df['NEG_Value_grid_input_customer'] = merged_df.apply(
                lambda row: row['Epex incl afslag Price_eur/kwh'] * row['Energy_Consumption_kWh']
                if row['Epex incl afslag Price_eur/kwh'] < 0 and row['Energy_Consumption_kWh'] < 0 else 0,
                axis=1
            )

            # Store processed DataFrame in session_state
            st.session_state.merged_df = merged_df
        else:
            merged_df = st.session_state.merged_df

        # ----------------------------
        # Summary Calculations
        # ----------------------------
        kwh_neg_grid_input_customer = int(
            merged_df.loc[merged_df['Energy_Consumption_kWh'] < 0, 'Energy_Consumption_kWh'].sum()
        )
        NEG_Value_grid_input_customer = merged_df.loc[
            (merged_df['Epex incl afslag Price_eur/kwh'] < 0) & (merged_df['Energy_Consumption_kWh'] < 0),
            ['Epex incl afslag Price_eur/kwh', 'Energy_Consumption_kWh']
        ].prod(axis=1).sum()

        # Aggregate Data by Year-Month for the monthly bar graph
        merged_df['Year-Month'] = merged_df['DateTime'].dt.to_period('M').astype(str)
        monthly_data = merged_df.groupby('Year-Month').agg({
            'NEG_Value_grid_input_customer': 'sum',
            'Kwh of negative grid input': 'sum'
        }).reset_index()

        st.markdown(f"""
        <div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px; text-align: center;'>
            <h3>Your results of your energy provided back into the grid</h3>
            <p><strong>Total kWh provided to the grid:</strong> {kwh_neg_grid_input_customer:,} kWh</p>
            <p><strong>Total paid for negative grid input based on EPEX price (incl afslag):</strong> €{NEG_Value_grid_input_customer:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------
        # Static Graph: Monthly Negative Grid Feed Impact (Styled)
        # ----------------------------
        st.markdown("""
        <h4 style='text-align: center;'>Visualization of the energy provided to the grid and the cost impact</h4>
        """, unsafe_allow_html=True)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')  # Overall figure background
        ax1.set_facecolor('white')        # Axis background

        x = np.arange(len(monthly_data['Year-Month']))
        bar_width = 0.4

        # Custom colors: techy blue for kWh and grey for euros
        color_kwh = '#3498db'
        color_euro = '#95a5a6'

        # Plot kWh data on left y-axis
        ax1.bar(x - bar_width/2, monthly_data['Kwh of negative grid input'], 
                width=bar_width, color=color_kwh, label='kWh of Negative Grid Input')
        ax1.set_ylabel("Energy grid input [kWh]", color=color_kwh, size=12)
        ax1.tick_params(axis='y', labelcolor=color_kwh)
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_data['Year-Month'], rotation=45, size=10)
        ax1.set_xlabel("Year-Month", size=12)

        # Create a twin axis for the euro data
        ax2 = ax1.twinx()
        ax2.bar(x + bar_width/2, monthly_data['NEG_Value_grid_input_customer'], 
                width=bar_width, color=color_euro, label='€ Paid for Negative Grid Input')
        ax2.set_ylabel("€ Paid for energy grid input", color=color_euro, size=12)
        ax2.tick_params(axis='y', labelcolor=color_euro, size=10)

        # Remove legends so the bars have maximum space
        ax1.legend().set_visible(False)
        ax2.legend().set_visible(False)

        # Style gridlines
        ax1.grid(True, linestyle='--', color='lightgrey', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

        # ----------------------------
        # Interactive Time Series Analysis (Plotly) - Styled
        # ----------------------------
        st.subheader("Interactive Time Series Analysis")

        st.markdown(
            """
            <style>
            /* This CSS targets slider labels rendered by Streamlit's baseweb component */
            div[data-baseweb="slider"] label {
                color: blue !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Convert pandas Timestamps to native Python datetimes for the slider
        min_date = merged_df['DateTime'].min().to_pydatetime()
        max_date = merged_df['DateTime'].max().to_pydatetime()

        selected_range = st.slider(
            "Select date range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD HH:mm"
        )

        # Filter data based on the selected date range
        filtered_data = merged_df[
            (merged_df['DateTime'] >= selected_range[0]) &
            (merged_df['DateTime'] <= selected_range[1])
        ]

        # Create a Plotly figure with three vertically stacked subplots
        fig2 = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Energy consumption in [kwh]", 
                "EPEX Spot Price €/Kwh (incl afslag)",
                "Result € financial impact of grid input (incl afslag)"
            )
        )

        # Top subplot: Negative Power Production Analysis (techy blue)
        fig2.add_trace(
            go.Scatter(
                x=filtered_data['DateTime'],
                y=filtered_data['Energy_Consumption_kWh'],
                mode='lines+markers',
                name='Energy Consumption (kWh) per hour',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1
        )

        # Middle subplot: EPEX Spot Price Analysis (red)
        fig2.add_trace(
            go.Scatter(
                x=filtered_data['DateTime'],
                y=filtered_data['Epex incl afslag Price_eur/kwh'],
                mode='lines+markers',
                name='EPEX Price (incl afslag)',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )

        # Bottom subplot: Result EUR of Grid Input (incl afslag) (grey)
        fig2.add_trace(
            go.Scatter(
                x=filtered_data['DateTime'],
                y=filtered_data['NEG_Value_grid_input_customer'],
                mode='lines+markers',
                name='Energy cost for grid input EUR (incl afslag)',
                line=dict(color='#95a5a6', width=2)
            ),
            row=3, col=1
        )

        # Update layout with custom styling (no legend, white background, custom margins)
        fig2.update_layout(
            height=850,
            title={
                'text': "Interactive Time Series Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=20)
            },
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Update x-axis and y-axis styling for all subplots
        fig2.update_xaxes(
            title_text="Date and Time", 
            row=3, col=1,
            gridcolor="lightgrey",
            linecolor="black",
            tickfont=dict(color="black")
        )
        fig2.update_yaxes(
            title_text="Energy Consumption (kWh)", 
            row=1, col=1,
            gridcolor="lightgrey",
            linecolor="black",
            tickfont=dict(color="black")
        )
        fig2.update_yaxes(
            title_text="EPEX Price (incl afslag) (€/kWh)", 
            row=2, col=1,
            gridcolor="lightgrey",
            linecolor="black",
            tickfont=dict(color="black")
        )
        fig2.update_yaxes(
            title_text="Result EUR (incl afslag)", 
            row=3, col=1,
            gridcolor="lightgrey",
            linecolor="black",
            tickfont=dict(color="black")
        )

        st.plotly_chart(fig2, use_container_width=True)


        # ----------------------------
        # Summary Calculations
        # ----------------------------
        filtered_data_kwh_neg_grid_input_customer = int(
            filtered_data.loc[filtered_data['Energy_Consumption_kWh'] < 0, 'Energy_Consumption_kWh'].sum()
        )
        filtered_data_NEG_Value_grid_input_customer = filtered_data.loc[
            (filtered_data['Epex incl afslag Price_eur/kwh'] < 0) & (filtered_data['Energy_Consumption_kWh'] < 0),
            ['Epex incl afslag Price_eur/kwh', 'Energy_Consumption_kWh']
        ].prod(axis=1).sum()
        
        st.markdown(f"""
        <div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px; text-align: center;'>
            <h5>Interactive results of negative energy grid input</h5>
            <p><strong>Total kWh provided to grid: </strong> {filtered_data_kwh_neg_grid_input_customer:,} kWh</p>
            <p><strong>Total paid for negative grid input based on EPEX price:</strong> €{filtered_data_NEG_Value_grid_input_customer:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------
        # Data Preview
        # ----------------------------
        st.subheader("Processed Data Preview")
        st.dataframe(merged_df.head(40).style.format({
            'Epex incl afslag Price_eur/kwh': '{:.6f}',
            'Epex Price_eur/kwh': '{:.6f}'
        }))
    else:
        st.info("Click 'Run Calculation' to process the data.")
