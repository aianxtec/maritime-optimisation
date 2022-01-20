import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pickle



fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
fig.write_html('first_figure.html', auto_open=True)

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


app.layout = html.Div(
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

@app.callback(
    Output('out-all-types', 'children'),
    Input('submit-val', 'n_clicks'),
    [State("input_{}".format(_), "value") for _ in INPUT_NAME],
)
def predict_v(n_clicks, *vals):
    if len(vals) == 6 and all(vals):
        model_predict = model.predict([list(vals)])
        
        
        #plot
        
        return "Prediction: {}".format(str(model_predict))
    else:
        return "please enter all values"



if __name__ == "__main__":
    app.run_server(debug=True)