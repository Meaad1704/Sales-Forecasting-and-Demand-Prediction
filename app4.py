import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
# Path to your dataset. Update this path if needed.
DATA_PATH = r'D:\Courses\Data Science - Depi\Final Project\rossmann-store-sales\rossman.csv'

def load_data():
    # Load the dataset with parsed dates and rename columns for Prophet.
    data = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    data = data.rename(columns={'Date': 'ds', 'Sales': 'y'})
    # Keep only the needed columns.
    data = data[['ds', 'y', 'Customers', 'Promo', 'Open', 'StateHoliday', 'Store', 'DayOfWeek']]
    return data

# Load the data once for the application.
data = load_data()

# Instantiate the Dash app.
app = Dash(__name__)
server = app.server  # Expose the server if needed for deployment.

# Create the app layout.
app.layout = html.Div([
    html.H1(
        "Prophet Model: Sales Forecasting for Rossmann Stores",
        style={
            'margin-bottom': '50px',
            'fontWeight': 'bold',
            'color': 'white',
            'textAlign': 'center'
        }
    ),
    html.Img(
        src="/assets/Model_Deployment/rossmann2254.jpg",
        style={
            'width': '50%',
            'display': 'block',
            'margin': '0 auto'
        }
    ),
    html.Label(
        "Select Store Number:",
        style={
            'fontSize': '20px',
            'fontWeight': 'bold',
            'color': 'white'
        }
    ),
    dcc.Dropdown(
        id='store-dropdown',
        options=[{'label': str(store), 'value': store} for store in sorted(data['Store'].unique())],
        value=1,
        style={
            'width': '300px',
            'fontSize': '20px',
            'fontWeight': 'bold'
        }
    ),
    html.Br(),
    html.Label(
        "Enter Number of Forecast Days:",
        style={
            'fontSize': '20px',
            'fontWeight': 'bold',
            'color': 'white'
        }
    ),
    html.Br(),
    dcc.Input(
        id='periods-input',
        type='number',
        value=90,
        min=1,
        style={
            'width': '300px',
            'fontSize': '20px',
            'fontWeight': 'bold'
        }
    ),
    html.Br(), html.Br(),
    html.Button(
        "Generate Forecast",
        id="submit-button",
        n_clicks=0,
        style={
            'width': '300px',
            'fontSize': '20px',
            'fontWeight': 'bold'
        }
    ),
    html.Br(), html.Br(),

    # Download button and hidden download component for the forecast data.
    html.Div(
        children=[
            html.Button(
                "Download Forecast Data",
                id="download-forecast-button",
                style={
                    'width': '300px',
                    'fontSize': '20px',
                    'fontWeight': 'bold',
                    'marginTop': '10px'
                }
            ),
            dcc.Download(id="download-forecast")
        ],
        style={
            'width': '50%',
            'margin': '0 auto',
            'textAlign': 'center'
        }
    ),

    # Group each model's graph (without messages now).
    html.Div(
        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start'},
        children=[
            html.Div([
                dcc.Graph(id='forecast-graph')
            ], style={'width': '50%'}),
            html.Div([
                dcc.Graph(id='baseline-graph')
            ], style={'width': '50%'})
        ]
    ),
      
    # Hidden store to hold the forecast data as JSON.
    dcc.Store(id="forecast-data-store")
],style={'backgroundColor': 'orange'})

# Main callback: updates both graphs and stores the forecast output.
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('baseline-graph', 'figure'),
     Output('forecast-data-store', 'data')],
    [Input("submit-button", "n_clicks")],
    [State("store-dropdown", "value"),
     State("periods-input", "value")]
)
def update_forecast(n_clicks, store_val, periods):
    if n_clicks == 0:
        return go.Figure(), go.Figure(), None

    # Filter the dataset for the selected store.
    filtered_data = data[data['Store'] == store_val].sort_values('ds')
    if filtered_data.empty:
        return go.Figure(), go.Figure(), None

    # Split the data into training and test sets.
    split_date_recent = filtered_data['ds'].max() - pd.DateOffset(months=6)
    train = filtered_data[filtered_data['ds'] < split_date_recent].copy()
    test = filtered_data[filtered_data['ds'] >= split_date_recent].copy()

    # Create a holiday dataframe for the forecasting model.
    holiday_df = filtered_data[filtered_data['StateHoliday'] == 1][['ds']].drop_duplicates().copy()
    holiday_df['holiday'] = 'state_holiday'

    ##########################################
    # Forecast Model
    ##########################################
    model = Prophet(daily_seasonality=True, holidays=holiday_df)
    model.fit(train)
    Future_Predict = model.make_future_dataframe(periods=int(periods + len(test)))
    forecast = model.predict(Future_Predict)

    fig = plot_plotly(model, forecast)
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(visible=True),
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    # Add test data as red markers.
    fig.add_trace(go.Scatter(
        x=test['ds'],
        y=test['y'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Test Data'
    ))
    split_date_train = train['ds'].max().to_pydatetime()
    split_end_date = filtered_data['ds'].max().to_pydatetime()
    fig.add_vline(x=split_date_train, line=dict(color='red', dash='dash', width=2))
    fig.add_vline(x=split_end_date, line=dict(color='violet', dash='dash', width=2))
    max_y = filtered_data['y'].max()
    fig.add_annotation(x=split_date_train, y=max_y, text="Train/Test Split",
                       showarrow=True, arrowhead=1, ax=0, ay=-40)
    fig.add_annotation(x=split_end_date, y=max_y, text="Future Predict Split",
                       showarrow=True, arrowhead=1, ax=0, ay=-40)
    Performance = pd.merge(
        test,
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-int(periods + len(test)):],
        on='ds'
    )
    mae_forecast = mean_absolute_error(Performance['y'], Performance['yhat'])
    fig.update_layout(
        title={
            'text': f"Forecasting Model <br> MAE {mae_forecast:.2f}",
            'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': 'blue'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Date",
        yaxis_title="Sales",
        width=800,
        height=600
    )

    ##########################################
    # Baseline Model
    ##########################################
    model_baseline = Prophet()
    model_baseline.fit(train)
    future_baseline = model_baseline.make_future_dataframe(periods=len(test))
    forecast_baseline = model_baseline.predict(future_baseline)

    baseline_fig = plot_plotly(model_baseline, forecast_baseline)
    baseline_fig.add_vline(x=split_date_train, line=dict(color='red', dash='dash', width=2))
    baseline_fig.add_trace(go.Scatter(
        x=test['ds'],
        y=test['y'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Test Data'
    ))
    baseline_fig.add_annotation(x=split_date_train, y=max_y, text="Train/Test Split",
                                showarrow=True, arrowhead=1, ax=0, ay=-40)
    performance_baseline = pd.merge(
        test,
        forecast_baseline[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-len(test):],
        on='ds'
    )
    mae_baseline = mean_absolute_error(performance_baseline['y'], performance_baseline['yhat'])
    baseline_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(visible=True),
            rangeslider=dict(visible=False),
            type="date"
        ),
        title={
            'text': f"Baseline Model <br> MAE {mae_baseline:.2f}",
            'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': 'blue'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Date",
        yaxis_title="Sales",
        width=800,
        height=600
    )

    # Process the forecast data for download:
    forecast['Day'] = forecast['ds'].dt.day_name()
    forecast_data = forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted Future Sales'})
    forecast_data = forecast_data[['Date', 'Day', 'Predicted Future Sales']][-int(periods):]
    forecast_data.loc[forecast_data['Day'] == 'Sunday', 'Predicted Future Sales'] = 0
    forecast_json = forecast_data.to_json(date_format='iso', orient='split')

    return fig, baseline_fig, forecast_json

# Callback for download: uses the stored forecast data to export an Excel file.
@app.callback(
    Output("download-forecast", "data"),
    Input("download-forecast-button", "n_clicks"),
    State("forecast-data-store", "data"),
    prevent_initial_call=True
)
def download_forecast(n_clicks, forecast_json):
    if not forecast_json:
        return no_update
    df_forecast = pd.read_json(forecast_json, orient='split')
    return dcc.send_data_frame(df_forecast.to_excel, "Predicted Future Sales.xlsx", sheet_name="Forecast", index=False)

if __name__ == '__main__':
    app.run(debug=True)
