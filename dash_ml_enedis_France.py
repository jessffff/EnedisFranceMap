
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json
import plotly.graph_objects as go
import plotly.express as px

from urllib.request import urlopen
import json
with urlopen("https://raw.githubusercontent.com/jessffff/bdd/main/geojson_regions.json") as f:
    data = json.load(f)


df_final_20_22 = pd.read_csv("df_final_France_20_22.csv")

y = df_final_20_22["Moyenne_consommation"]
X = df_final_20_22.select_dtypes(include='number').drop(
    ["Moyenne_consommation", "Nb points soutirage", "Total énergie soutirée (Wh)"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=False)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(max_depth=None, max_features=1.0,
                              min_samples_leaf=1, min_samples_split=2, n_estimators=100)
model.fit(X_train_scaled, y_train)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

external_stylesheets = ['style.css']

# LAYOUT

app.layout = html.Div(
    className='container',
    children=[
        html.Img(
            src="assets/logo_enedis.PNG",
            className='logo'
        ),
        html.Br(),
        html.Br(),
        dcc.Tabs(
            id="tabs-with-classes",
            value='tab-1',
            parent_className='custom-tabs',
            className='custom-tabs-container',
            children=[
                dcc.Tab(
                    label='Présentation',
                    value='tab-1',
                    className='custom-tab',
                    selected_className='custom-tab--selected'
                ),
                dcc.Tab(
                    label='Prédictions',
                    value='tab-2',
                    className='custom-tab',
                    selected_className='custom-tab--selected'
                ),
            ]),
        html.Div(id='tabs-content-classes')
    ]
)

# CALLBACKS


@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Iframe(
                src="https://docs.google.com/presentation/d/e/2PACX-1vQsi9y2cjNVBKLE1QMejRiTPrBWPngvogn3IKggLFvFcitMTW9X68XcQHQ3r7zTTUsf-7dirlwv-d1q/embed?start=false&loop=false&delayms=3000",
                style={"width": "100%",  "height": "500px", "border": "none", "margin": "0 auto"})
        ]),
    elif tab == 'tab-2':
        return html.Div([
            html.H1("Analyse et prédiction de la consommation",
                    className='title'),
            html.P("Toutes les variables sont à compléter"),
            html.Br(),
            html.Br(),
            html.Br(),
            html.P("Choisissez une température (°C): "),
            dcc.Slider(
                id='temperature-slider',
                min=-20,
                max=40,
                step=0.1,
                value=20,
                marks={-20: '-20°C', -15: '-15°C', -10: '-10°C', -5: '-5°C', 0: '0°C', 5: '5°C', 10: '10°C',
                       15: '15°C', 20: '20°C', 25: '25°C', 30: '30°C', 35: '35°C', 40: '40°C'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False
            ),
            html.P("Quantité de pluie (mm)"),
            dcc.Slider(
                id='rainfall-slider',
                min=0,
                max=30,
                step=0.1,
                value=10,
                marks={0: '0mm', 5: '5mm', 10: '10mm',
                       15: '15mm', 20: '20mm', 25: '25mm', 30: '30mm'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False
            ),
            html.Br(),
            html.Br(),
            html.Div([
                html.P("Statut du jour de la semaine"),
                dcc.Dropdown(
                    id='style-jour-dropdown',
                    options=[
                        {'label': 'Ouvré', 'value': 'Ouvré'},
                        {'label': 'Week-end', 'value': 'Week-end'},
                        {'label': 'Férié', 'value': 'Férié'},
                    ],
                    value='',
                    className='dropdown'
                ),
            ], className='dropdown-container1'),
            html.Div([
                html.P("Description"),
                dcc.Dropdown(
                    id='description-dropdown',
                    options=[
                        {'label': "Sans objet", 'value': 'Aucune'},
                        {'label': 'Vacances', 'value': 'Vacances'},
                    ],
                    value='',
                    className='dropdown'
                ),
            ], className='dropdown-container2'),

            html.Br(),
            html.Br(),
            html.Br(),

            html.Br(),
            html.Div([
                html.P("Mois"),
                dcc.Dropdown(
                    id='date-input',
                    options=[
                        {'label': 'Janvier', 'value': '1'},
                        {'label': 'Février', 'value': '2'},
                        {'label': 'Mars', 'value': '3'},
                        {'label': 'Avril', 'value': '4'},
                        {'label': 'Mai', 'value': '5'},
                        {'label': 'Juin', 'value': '6'},
                        {'label': 'Juillet', 'value': '7'},
                        {'label': 'Aout', 'value': '8'},
                        {'label': 'Septembre', 'value': '9'},
                        {'label': 'Octobre', 'value': '10'},
                        {'label': 'Novembre', 'value': '11'},
                        {'label': 'Décembre', 'value': '12'}
                    ],
                    value='',
                    className='dropdown'
                ),
            ], className='dropdown-container1'),
            html.Div([
                html.P("Profil consommateur"),
                dcc.Dropdown(
                    id='profil-consommateur-dropdown',
                    options=[
                        {'label': 'Professionnel', 'value': 'Professionnel'},
                        {'label': 'Résidentiel', 'value': 'Résidentiel'}
                    ],
                    value='',
                    className='dropdown'
                ),
            ], className='dropdown-container2'),
            html.Br(),
            html.Br(),
            html.Button('Prédire', id='predict-button',
                        n_clicks=0, className='prediction-button'),
            html.Br(),
            html.H3("Consommation électrique prédite par région"),
            html.Div(id='prediction-output'),


        ])


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("temperature-slider", "value"),
    State("rainfall-slider", "value"),
    State("style-jour-dropdown", "value"),
    State("description-dropdown", "value"),
    State("date-input", "value"),
    State("profil-consommateur-dropdown", "value")
)
def update_prediction_output(n_clicks, temperature, rainfall, style_jour, description, date, profil_consommateur):
    if n_clicks > 0 and date == '' or style_jour == '' or description == '' or profil_consommateur == '':
        return html.P('Veuillez remplir les champs nécessaires', style={'color': 'red', 'font-size': '20px'})
    if n_clicks > 0 and date != '':

        input_data = pd.DataFrame(columns=X_train.columns)

        input_data.loc[0, 'Température (°C)'] = temperature
        input_data.loc[0,
                       'Précipitations dans les 3 dernières heures'] = rainfall
        input_data['Statut_férié'] = 0
        input_data['Statut_ouvré'] = 0
        input_data['Statut_week-end'] = 0

        if style_jour == 'Férié':
            input_data['Statut_férié'] = 1
        elif style_jour == 'Ouvré':
            input_data['Statut_ouvré'] = 1
        elif style_jour == 'Week-end':
            input_data['Statut_week-end'] = 1

        input_data['Profil_consommateur_Professionnel'] = 0
        input_data['Profil_consommateur_Résident'] = 0

        if profil_consommateur == 'Professionnel':
            input_data['Profil_consommateur_Professionnel'] = 1
        elif profil_consommateur == 'Résidentiel':
            input_data['Profil_consommateur_Résident'] = 1

        input_data['Mois'] = date

        input_data['Vacances_Vacances'] = 0

        if description == 'Vacances':
            input_data['Vacances_Vacances'] = 1

        input_data = input_data.fillna(X_train[(X_train['Mois'] == int(
            date))].mean())

        input_data_scaled = scaler.transform(input_data)

        regions = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de Loire',
                   'Grand-Est', 'Hauts-de-France', 'Île-de-France', 'Normandie', 'Nouvelle Aquitaine',
                   'Occitanie', 'Pays de la Loire', "Provence-Alpes-Côte d'Azur"]

        predictions = []
        for region in regions:
            input_data_region = input_data_scaled.copy()
            input_data_region[0, X_train.columns.str.startswith('Région_')] = 0
            input_data_region[0, X_train.columns == f"Région_{region}"] = 1
            predicted_consumption = model.predict(input_data_region)
            predictions.append(predicted_consumption[0] / 1000)
        df_predictions = pd.DataFrame(
            {'Région': regions, 'Prédict. conso (kWh)': predictions})
        df_predictions["Prédict. conso (kWh)"] = df_predictions["Prédict. conso (kWh)"].round(
            3)

        fig = px.choropleth_mapbox(df_predictions,
                                   geojson=data,
                                   locations='Région',
                                   color="Prédict. conso (kWh)",
                                   color_continuous_scale="haline_r",
                                   labels={"Prédict. conso (kWh)": 'kWh'},
                                   featureidkey='properties.nom',
                                   mapbox_style="carto-positron",
                                   zoom=4, center={"lat": 46.2276, "lon": 2.2137},
                                   opacity=0.5,
                                   )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return html.Div(
            style={'display': 'flex', 'flex-direction': 'row'},
            children=[
                html.Div(
                    style={'width': '40%'},
                    children=[
                        html.Table([
                            html.Thead(html.Tr([html.Th(col)
                                       for col in df_predictions.columns])),
                            html.Tbody([
                                html.Tr([html.Td(df_predictions.iloc[i][col])
                                        for col in df_predictions.columns])
                                for i in range(len(df_predictions))
                            ])
                        ])
                    ]
                ),
                html.Div(
                    style={'width': '60%'},
                    children=[
                        dcc.Graph(figure=fig)
                    ]
                )
            ]
        )

    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
