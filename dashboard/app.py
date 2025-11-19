"""
Hospital Readmission Risk Prediction Dashboard
Interactive Plotly Dash application for visualizing readmission predictions
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import (
    load_data_and_model, get_kpi_metrics, get_top_risk_factors,
    filter_dataframe, calculate_cost_savings
)

# Initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Hospital Readmission Dashboard"
)

# Load data
print("Loading data and model...")
df, model, model_metadata, feature_cols = load_data_and_model()
kpis = get_kpi_metrics(df)
risk_factors = get_top_risk_factors(df, feature_cols)
df['risk_factors'] = df.index.map(risk_factors)

print(f"Loaded {len(df)} patient records")
print(f"Model: {model_metadata['model_name']}")
print(f"Model Accuracy: {model_metadata['metrics']['accuracy']:.2%}")

# Calculate cost savings
cost_info = calculate_cost_savings(df)

# Color scheme
COLORS = {
    'NO': '#28a745',       # Green
    '<30': '#dc3545',      # Red
    '>30': '#ffc107',      # Yellow/Orange
    'background': '#f8f9fa',
    'card': '#ffffff'
}

# App Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🏥 Hospital Readmission Risk Prediction Dashboard", 
                   className="text-center mb-2 mt-4"),
            html.P("Predicting 30-day hospital readmission risk using machine learning",
                  className="text-center text-muted mb-4")
        ])
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Patients", className="card-title text-muted"),
                    html.H2(f"{kpis['total_patients']:,}", className="text-primary"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("High Risk (<30 days)", className="card-title text-muted"),
                    html.H2(f"{kpis['high_risk']:,}", className="text-danger"),
                    html.P(f"{kpis['high_risk_pct']:.1f}%", className="text-muted")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("No Readmission", className="card-title text-muted"),
                    html.H2(f"{kpis['low_risk']:,}", className="text-success"),
                    html.P(f"{kpis['low_risk_pct']:.1f}%", className="text-muted")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Model Accuracy", className="card-title text-muted"),
                    html.H2(f"{kpis['accuracy']:.1f}%", className="text-info"),
                    html.P(model_metadata['model_name'], className="text-muted small")
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Cost Savings Card
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("💰 Potential Impact", className="card-title"),
                    html.P(f"Predicted Readmissions: {cost_info['predicted_readmissions']:,}", 
                          className="mb-2"),
                    html.P(f"Preventable (25%): {cost_info['prevented_readmissions']:,}", 
                          className="mb-2"),
                    html.H5(f"Est. Cost Savings: ${cost_info['cost_savings']:,}", 
                           className="text-success")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Filters", className="card-title mb-3"),
                    
                    html.Label("Gender:"),
                    dcc.Dropdown(
                        id='gender-filter',
                        options=[
                            {'label': 'All', 'value': 'All'},
                            {'label': 'Male', 'value': 'Male'},
                            {'label': 'Female', 'value': 'Female'}
                        ],
                        value='All',
                        className="mb-3"
                    ),
                    
                    html.Label("Risk Score Threshold (%):"),
                    dcc.Slider(
                        id='risk-slider',
                        min=0,
                        max=100,
                        step=5,
                        value=0,
                        marks={i: f'{i}%' for i in range(0, 101, 25)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Visualizations Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Readmission Risk Distribution", className="card-title"),
                    dcc.Graph(id='risk-distribution-bar')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Distribution (Pie)", className="card-title"),
                    dcc.Graph(id='risk-distribution-pie')
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    # Visualizations Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Score Distribution", className="card-title"),
                    dcc.Graph(id='risk-score-histogram')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Confusion Matrix", className="card-title"),
                    dcc.Graph(id='confusion-matrix')
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    # Patient Risk Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Patient Risk Table", className="card-title mb-3"),
                    html.P(id='table-info', className="text-muted"),
                    dash_table.DataTable(
                        id='patient-table',
                        columns=[
                            {'name': 'Patient ID', 'id': 'patient_id'},
                            {'name': 'Age', 'id': 'age_numeric'},
                            {'name': 'Gender', 'id': 'gender'},
                            {'name': 'Predicted Risk', 'id': 'predicted_label'},
                            {'name': 'Risk Score (%)', 'id': 'risk_score'},
                            {'name': 'True Outcome', 'id': 'true_label'},
                            {'name': 'Top Risk Factors', 'id': 'risk_factors'},
                        ],
                        page_size=15,
                        sort_action='native',
                        filter_action='native',
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontSize': '12px'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{predicted_label} = "<30"'},
                                'backgroundColor': '#ffebee'
                            },
                            {
                                'if': {'filter_query': '{predicted_label} = ">30"'},
                                'backgroundColor': '#fff8e1'
                            },
                            {
                                'if': {'filter_query': '{predicted_label} = "NO"'},
                                'backgroundColor': '#e8f5e9'
                            },
                        ]
                    )
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Hospital Readmission Risk Prediction System | Powered by Machine Learning",
                  className="text-center text-muted small")
        ])
    ])
    
], fluid=True, style={'backgroundColor': COLORS['background']})


# Callbacks
@app.callback(
    [Output('risk-distribution-bar', 'figure'),
     Output('risk-distribution-pie', 'figure'),
     Output('risk-score-histogram', 'figure'),
     Output('confusion-matrix', 'figure'),
     Output('patient-table', 'data'),
     Output('table-info', 'children')],
    [Input('gender-filter', 'value'),
     Input('risk-slider', 'value')]
)
def update_dashboard(gender, risk_threshold):
    """Update all dashboard components based on filters"""
    
    # Filter data
    filtered_df = filter_dataframe(df, gender=gender, risk_threshold=risk_threshold)
    
    # Bar chart - Risk Distribution
    risk_counts = filtered_df['predicted_label'].value_counts()
    fig_bar = go.Figure(data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[COLORS['NO'], COLORS['<30'], COLORS['>30']],
            text=risk_counts.values,
            textposition='auto'
        )
    ])
    fig_bar.update_layout(
        xaxis_title="Predicted Risk Category",
        yaxis_title="Number of Patients",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Pie chart - Risk Distribution
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker_colors=[COLORS['NO'], COLORS['<30'], COLORS['>30']],
        hole=0.3
    )])
    fig_pie.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Histogram - Risk Score Distribution
    fig_hist = px.histogram(
        filtered_df,
        x='risk_score',
        nbins=50,
        color='predicted_label',
        color_discrete_map={'NO': COLORS['NO'], '<30': COLORS['<30'], '>30': COLORS['>30']},
        labels={'risk_score': 'Risk Score (%)'},
        height=350
    )
    fig_hist.update_layout(
        xaxis_title="Risk Score (%)",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(filtered_df['readmitted_encoded'], filtered_df['predicted_class'])
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['NO', '<30', '>30'],
        y=['NO', '<30', '>30'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Table data
    table_cols = ['patient_id', 'age_numeric', 'gender', 'predicted_label', 
                  'risk_score', 'true_label', 'risk_factors']
    table_data = filtered_df[table_cols].round(1).to_dict('records')
    
    # Table info
    table_info = f"Showing {len(filtered_df)} patients (filtered from {len(df)} total)"
    
    return fig_bar, fig_pie, fig_hist, fig_cm, table_data, table_info


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting Hospital Readmission Dashboard...")
    print("="*70)
    print(f"\n📊 Dashboard will be available at: http://127.0.0.1:8050")
    print(f"\nPress Ctrl+C to stop the server\n")
    
    app.run_server(debug=False, host='127.0.0.1', port=8050)

