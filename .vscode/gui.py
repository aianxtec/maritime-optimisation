import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from torch import float64


filevessel = 'vesseldata.csv'
fileenv = 'envdata.csv'
filefuel = 'fueldata.csv'

vesseldf = pd.read_csv(filevessel)
envdf = pd.read_csv(fileenv)
fueldf = pd.read_csv(filefuel)

vesseldf["Date"] = pd.to_datetime(vesseldf["Date"])
vesseldf["Time"] = pd.to_datetime(vesseldf["Time"])

envdf["Date"] = pd.to_datetime(envdf["Date"])
envdf["Time"] = pd.to_datetime(envdf["Time"])

# fueldf["Date"] = pd.to_datetime(fueldf["Date"])
# fueldf["Time"] = pd.to_datetime(fueldf["Time"])


###################################
# D  A  T  A  ***  A  C  Q  U  I  S  I  T  I  O  N

CombinedMetrics = pd.DataFrame()
# joined all datasources into one DF for easier manipulation
CombinedMetrics = pd.concat([vesseldf, envdf, fueldf], axis=1)

# Setting Output Variables
yVar = pd.DataFrame()
yVar = (CombinedMetrics[['SOX', 'NOX', 'Viscosity_cst']])

# Setting Input Variables
Xvar = CombinedMetrics.drop(
    columns=['SOX', 'NOX', 'Date', 'Time', 'Viscosity_cst'], axis=1)

# print(Xvar, yVar)

XandY = pd.concat([Xvar, yVar], axis=1)

pt = PowerTransformer(method='yeo-johnson')
dataTransform = pt.fit_transform(XandY.iloc[: , 0:14])
# convert the array back to a dataframe
datasetYeo = pd.DataFrame(dataTransform, columns=['Load_pct', 'Engine-RPM', 'SOG', 'STW', 'DFT', 'L.O.Main_Pressure_kgm3',
                                                  'L.O.Main_Temp_Celcius', 'RWS', 'RWD', 'Wave_height', 'Air_Temp', 'F.O_pressure', 'F.O_Temp_Celcius', 'Viscosity_cst'])


model = pickle.load(open('saved_model.pkl', 'rb'))
# print(model.predict([[12,32,32,4,5,4]]))

L_O_Main_Pressure_kgm3 = None
SOG = None
Wave_height = None
F_O_pressure = None
RWD = None
F_O_Temp_Celcius = None

app = dash.Dash(__name__)

# ALLOWED_TYPES = (
#     "number",
# )


INPUT_NAME = (
    "L_O_Main_Pressure_kgm3","SOG", "Wave_height", "F_O_pressure", "RWD", "F_O_Temp_Celcius"
)


app.layout = html.Div(children=
    [   

        
        dcc.Input(
            id="input_{}".format(_),
            type="number",
            placeholder="{}".format(_),
        
        )
        for _ in INPUT_NAME
    ]
    + [html.Div(id="out-all-types")]
    # + [html.Div(id="out_{}".format(_)) for _ in INPUT_NAME]
    +[html.Button('Submit', id='submit-val', n_clicks=0)]
    
)
dcc.Graph(id="graph") 



@app.callback(
    Output('out-all-types', 'children'),
    Input('submit-val', 'n_clicks'),
    [State("input_{}".format(_), "value") for _ in INPUT_NAME],
)
def predict_v(n_clicks, *vals):
    if len(vals) == 6 and all(vals):
        model_predict = model.predict([list(vals)])
        
        # dfx = pd.DataFrame(np.array(vals), model_predict,)

        vallist = list(vals)
        vallist = vallist.append(model_predict[0][0])
        # modellist = list(model_predict)
        df = pd.DataFrame(vallist ,columns = ["L_O_Main_Pressure_kgm3","SOG", "Wave_height", "F_O_pressure", "RWD", "F_O_Temp_Celcius"])
        
        # df = px.data.gapminder()
        # fig = px.scatter(datasetYeo, x=datasetYeo.L_O_Main_Pressure_kgm3, y = datasetYeo['Viscosity_cst'],
        # log_x=False, size_max=60)
        
        #  fig = px.scatter(datasetYeo, x=["L_O_Main_Pressure_kgm3","SOG", "Wave_height", "F_O_pressure", "RWD", "F_O_Temp_Celcius"], y = [[model_predict]],
        # log_x=False, size_max=60)
        
        
        fig = px.parallel_coordinates(datasetYeo, color=datasetYeo.index, labels={"L_O_Main_Pressure_kgm3": "L_O_Main_Pressure_kgm3",
                  "SOG": "SOG", "Wave_height": "Wave_height",
                  "F_O_pressure": "F_O_pressure", "RWD": "RWD","F_O_Temp_Celcius": "F_O_Temp_Celcius" },
                     color_continuous_midpoint=2)
        # fig = px.area(df, x = vallist, y = [[model_predict]])
        #plot
        

        fig.write_html('first_figure.html', auto_open=True)
        
        
        return "Prediction: {}".format(str(model_predict))
    else:
        return "please enter all values"


if __name__ == "__main__":
    app.run_server(debug=True)
    
