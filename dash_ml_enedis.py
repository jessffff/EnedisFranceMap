import pandas as pd
import dash

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Load the CSV data
df_final_20_22 = pd.read_csv("df_final_20_22.csv")


# Define X and y variables
y = df_final_20_22["Consommation_moyenne"]
X = df_final_20_22.select_dtypes(include='number').drop(
    ["Consommation_moyenne", "Nb points soutirage", "Total énergie soutirée (Wh)"], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=False)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(max_depth=None, max_features=1.0,
                              min_samples_leaf=1, min_samples_split=2, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Create the Dash application
app = dash.Dash(__name__)
server = app.server 

# External CSS stylesheets
external_stylesheets = ['assets/style.css']

# Define the application layout
app.layout = html.Div(
    className='container',
    children=[
        html.Br(),
        html.Img(
            src="assets/logo_enedis.PNG",
            className='logo'
        ),
        html.Br(),
        html.Br(),
        html.Br(),
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
            tooltip={"placement": "bottom", "always_visible": True}
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
            tooltip={"placement": "bottom", "always_visible": True}
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
                    {'label': 'Confinement', 'value': 'Confinement'},
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
        html.P("Région"),
        dcc.Dropdown(
            id='region-dropdown',
            options=[
                {'label': 'Centre Val de Loire', 'value': 'Centre-Val de Loire'},
                {'label': 'Hauts de France', 'value': 'Hauts-de-France'}
            ],
            value='',
            className='dropdown'
        ),
        html.Br(),
        html.Br(),
        html.Button('Prédire', id='predict-button',
                    n_clicks=0, className='prediction-button'),
        html.Br(),
        html.H3("Consommation électrique prédite"),
        html.Div(id='prediction-output')
    ],
)

# Callback function


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("temperature-slider", "value"),
    State("rainfall-slider", "value"),
    State("style-jour-dropdown", "value"),
    State("description-dropdown", "value"),
    State("date-input", "value"),
    State("profil-consommateur-dropdown", "value"),
    State("region-dropdown", "value")
)
def update_prediction_output(n_clicks, temperature, rainfall, style_jour, description, date, profil_consommateur, region):
    if n_clicks > 0 and date == '' or style_jour == '' or description == '' or profil_consommateur == '' or region == '':
        return html.P('Veuillez remplir les champs nécessaires', style={'color': 'red', 'font-size': '20px'})
    if n_clicks > 0 and date != '':
        noms_mois = ['janvier', 'février', 'mars', 'avril', 'mai',
                     'juin', 'septembre', 'octrobre', 'novembre', 'décembre']
        input_data = pd.DataFrame(columns=X_train.columns)

        input_data.loc[0, 'Moyenne_temperature'] = temperature
        input_data.loc[0, 'PRECIP_TOTAL_DAY_MM'] = rainfall
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

        input_data['Région_Centre-Val de Loire'] = 0
        input_data['Région_Hauts-de-France'] = 0

        if region == 'Centre-Val de Loire':
            input_data['Région_Centre-Val de Loire'] = 1
        elif region == 'Hauts-de-France':
            input_data['Région_Hauts-de-France'] = 1

        input_data['Mois'] = date

        input_data['Description_y_Confinement'] = 0
        input_data['Description_y_Vacances'] = 0

        if description == 'Vacances':
            input_data['Description_y_Vacances'] = 1
        elif description == 'Confinement':
            input_data['Description_y_Confinement'] = 1

        input_data = input_data.fillna(X_train[(X_train['Mois'] == int(
            date)) & (X_train['Région_' + region] == 1)].mean())

        temp_moy = X_train[(X_train['Mois'] == int(date)) & (
            X_train['Région_' + region] == 1)]['Moyenne_temperature'].mean()
        pluie_moy = X_train[(X_train['Mois'] == int(date)) & (
            X_train['Région_' + region] == 1)]['PRECIP_TOTAL_DAY_MM'].mean()
        if profil_consommateur == 'Professionnel':
            conso_moy = df_final_20_22[(df_final_20_22['Mois'] == int(date)) & (df_final_20_22['Région_' + region] == 1) & (
                df_final_20_22['Profil_consommateur_Professionnel'] == 1)]['Consommation_moyenne'].mean()
        else:
            conso_moy = df_final_20_22[(df_final_20_22['Mois'] == int(date)) & (df_final_20_22['Région_' + region] == 1) & (
                df_final_20_22['Profil_consommateur_Résident'] == 1)]['Consommation_moyenne'].mean()

        input_data_scaled = scaler.transform(input_data)

        predicted_consumption = model.predict(input_data_scaled)

        if predicted_consumption[0] > conso_moy:
            variation = 'hausse'
        else:
            variation = 'baisse'

        return html.P([html.Strong(f"{predicted_consumption[0]/1000:.2f} kWh."),
                       html.Br(),
                       f"Sélection:  {noms_mois[int(date)-1]}   température :  {temperature}°C    pluie: {rainfall}mm",
                       html.Br(),
                       f"Sur le mois {noms_mois[int(date)-1]} il fait en moyenne {temp_moy: .1f}°C et il tombe {pluie_moy: .0f} mm/jour"
                       f"la consommation électrique moyenne est alors de {conso_moy/1000: .2f} kWh",
                       html.Br(),
                       html.Strong(f"cela représente une {variation} de {100*(predicted_consumption[0]-conso_moy)/conso_moy: .2f}%.")])
# =============================================================================
#
#         input_data_scaled = scaler.transform(input_data)
#
#         predicted_consumption = model.predict(input_data_scaled)
#
#         return f"Prédiction de la consommation électrique : {predicted_consumption[0]:.2f} Wh"
# =============================================================================

    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
