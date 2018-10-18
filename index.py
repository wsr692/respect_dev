'''
Guided PCA plotter (main page)
 |
 |--apps
      |--app_ucm.py
      |--app_cm.py

2018-09-02

ref:
- https://community.plot.ly/t/deploying-multi-page-app-to-heroku-not-deploying-as-set-up/7877/3
'''

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import soundfile as sf

import os
import pandas as pd
import numpy as np
import pickle

# from app import app
from apps import app_ucm, app_cm
from app import app

### from spectrogram
from spectrogram.src.reassign_spec import data_process
import argparse

####


param = {}
param['win_size'] = 0.025
param['win_step'] = 0.01


def predict(param, wave, low, high, fftn, type, clip, window='kaiser'):
    # import processing class.
    param["data"] = "./sound_data/{}.wav".format(wave)
    param['window'] = window
    param['delay'] = 1
    param['freqdelay'] = 1

    param['low_cut'] = int(low)
    param['high_cut'] = int(high)
    param['fftn'] = int(fftn)
    param['value_type'] = type
    param['clip'] = -int(clip)

    a_wav = data_process(param)
    # apply window
    a_wav.apply_window()
    # apply fft
    a_wav.apply_fft()
    # extract specific frequency.
    a_wav.extract_postive()
    # critical part: calculate different angles.
    a_wav.compute_angles()
    # normalization
    a_wav.get_magnitude()

    if type == 'both':
        STFTplot, CIFplot, tremap = a_wav.retrieve_values('both')
    elif type == 'time':
        STFTplot, CIFplot, tremap = a_wav.retrieve_values('time')
    elif type == 'frequency':
        STFTplot, CIFplot, tremap = a_wav.retrieve_values('frequency')
    elif type == 'spectrogram':
        STFTplot, CIFplot, tremap = a_wav.retrieve_values('spectrogram')

    return STFTplot, CIFplot, 1000 * tremap


server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')
app.config.suppress_callback_exceptions = True

app.css.append_css(
    {"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

# -------------------- Data processing -------------------------- #


slider_text_style = {
    'padding': '0px 10px 10px 0px',
    'margin-top' : '20'
}

param_style = {
    'display': 'block',
    'padding': '10px 10px 10px 10px',
    'margin-bottom':'15'
}

col1_style = {
    'width': '25%',
    'display': 'inline-block',
}
col2_style = {
    'width': '70%',
    'display': 'inline-block',
}


row_style = {
    'display': 'inline-block',
    'width' : '70%',
    'padding': '0px 0px 0px 0px',
}

margins = {'l': 40, 'r': 40, 'b': 40, 't': 40}
fig_width1 = 1000
fig_height1 = 300
fig_width2 = 1000
fig_height2 = 1000

footer_style = {
    'fontSize': '15px',
    'display': 'block',
    'textAlign': 'center',
}




# -------------------- Figures -------------------------- #
fig_spec = go.Scatter(
     x=[0],
     y=[0],
     mode='lines',
     name='spectrum',
     line=dict(
         color=('rgb(0,0,0)')
     ),
     hoverinfo='none',
 )

fig_specgram1 = go.Scatter3d(
    x=[1],
    y=[2],
    z=[3],
    mode='markers',
    visible=False,
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)

# -------------------- Traces & Layouts -------------------------- #


traces_1 = [fig_spec]
layout_1 = go.Layout(
     height=fig_height1,
     width=fig_width1,
     margin=margins,
     showlegend=False,
     # Axis
     xaxis=dict(
         linecolor='black',
         mirror=True,
         title='Time <br />Waveform',
         range=[0, 100],
         dtick=20,
         tickangle=0,
         zeroline=False,
         tickfont=dict(
             size=10,
         ),
         fixedrange=True,
     ),
     yaxis=dict(
         title='Amplitude',
         linecolor='black',
         mirror=True,
         range=[-0.3, 0.3],
         dtick=10,
         zeroline=False,
         tickfont=dict(
             size=10,
         ),
         fixedrange=True,
     ),
 )


traces_2 = [fig_specgram1]
layout_2 = go.Layout(
    height=fig_height2,
    width=fig_width2,
    margin={'l': 20, 'r': 20, 'b': 20, 't': 20},
    showlegend=True,
    # Axis
    scene=dict(
        xaxis=dict(
            title='Time',
            mirror=True,
            range=[0, 400],
            tickfont=dict(
                size=10,
            ),
        ),
        yaxis=dict(
            title='Frequency',
            mirror=True,
            range=[0, 5000],
            tickfont=dict(
                size=10,
            ),
        ),
        zaxis=dict(
            title='Amplitude',
            mirror=True,
            range=[-90, 0],
            tickfont=dict(
                size=10,

            ),
        ),
        camera=dict(
            eye=dict(x=2.5, y=2.5, z=0.8)
        ),
        aspectmode='manual',
        aspectratio={'x': 1.2, 'y': 1.2, 'z': 1.2}
    )
)


# -------------------- Page layout -------------------------- #
# Set index_page layout
index_page = html.Div([
    html.H1('Reassigned Spectrogram',
            style={'textAlign': 'center'}),
    html.H3('Reassigning Time & Frequency', style={
            'textAlign': 'center'}),

    html.Div(className='row', children=[

        # ---------- First column ---------- #
        html.Div(children=[
            html.H4('Import Wav File', style={'textAlign': 'center'}),

            html.Div([dcc.Dropdown(
                id = "wave",
                options=[
                    {'label': 'a', 'value': 'a'},
                    {'label': 'e', 'value': 'e'},
                    {'label': 'i', 'value': 'i'},
                    {'label': 'o', 'value': 'o'},
                    {'label': 'u', 'value': 'u'}
                ],
                value="a"
                )
            ], style=param_style,)
        ]),


        html.Div(children=[
            html.H4('Parameters', style={'textAlign': 'center'}),

            html.Div([
                    html.P('Plot type', style=slider_text_style),
                dcc.Checklist(
                    id='checklist',
                    options=[
                        {'label': 'Normal (Red)', 'value': 'Normal'},
                        {'label': 'Time-reassigned (Green)', 'value': 'Time'},
                        {'label': 'Frequency-reassigned (Blue)', 'value': 'Freq'},
                        {'label': 'Time&Freq-reassigned (Gray)', 'value': 'Both'}
                    ],
                    values=['Normal']
                )]
                )], style=param_style),

            # fftn
            html.Div([
                html.P('fftn tick', style=slider_text_style),
                dcc.Dropdown(
                    id='fftn',
                    options=[
                        {'label': '128', 'value': '128'},
                        {'label': '256', 'value': '256'},
                        {'label': '512', 'value': '512'},
                        {'label': '1024', 'value': '1024'},
                        {'label': '2048', 'value': '2048'}
                    ],
                    value='1024'
                ),
            ], style=param_style),


                ### low cut
            html.Div([
                html.P('low cut', style=slider_text_style),
                dcc.Slider(
                    id='low',
                    min=10,
                    max=100,
                    marks={i: i for i in range(10, 100, 10)},
                    value=10,
                ),
            ], style=param_style),

                ### high cut
            html.Div([
                    html.P('high cut', style=slider_text_style),
                    dcc.Slider(
                    id='high',
                    min=5000,
                    max=8000,
                    marks={i: i for i in range(5000, 8000, 1000)},
                    value=5000,
                ),
                ], style=param_style),

            html.Div([
                    html.P('clip range (-)', style=slider_text_style),
                    dcc.Slider(
                    id='clip',
                    min=0,
                    max=100,
                    marks={i: i for i in range(0, 100, 10)},
                    value=0,
                ),
                ], style=param_style),

            html.Div([
                    html.P('windowing method', style=slider_text_style),
                    dcc.RadioItems(
                    id="window",
                    options=[
                        {'label': 'Kaiser', 'value': 'kaiser'},
                        {'label': 'Rectangular', 'value': 'rectangular'},
                        {'label': 'Hamming', 'value': 'hamming'},
                        {'label': 'Hanning', 'value': 'hanning'},
                        {'label': 'Blackman', 'value': 'blackman'}
                    ],
                    value='kaiser'
                ),
                ], style=param_style),


        ], style=col1_style),


        # ---------- Second column ---------- #
        html.Div(children=[
             # wave plot
             html.Div([
                 dcc.Graph(
                     id='1',
                     figure={
                         'data': traces_1,
                         'layout': layout_1,
                     },
                 ),
             ], style=col2_style),

            # spectrogram 3D plot
            html.Div([
                dcc.Graph(
                    id='2',
                    figure={
                        'data': traces_2,
                        'layout': layout_2,
                    },
                ),
                html.Br(),
            ], style=col2_style),
        ], className='two columns', style=row_style),
    ])


@app.callback(
    # Set callbacks:
    Output('1', 'figure'),
    [Input('wave', 'value')]
)

def update_wave(wave):

    print(wave)
    sound_data = "./sound_data/{}.wav".format(wave)

    signal, Fs = sf.read(sound_data)
    trace1 = go.Scatter(
        x=[i/len(signal)*100 for i in range(len(signal))],
        y=signal,
        mode='lines',
        visible=True,
        line = dict(
                color = ('rgb(0, 0, 0)'),
                width = 0.5
        ),
    )
    return {'data': traces_1 + [trace1],
            'layout': layout_1}


@app.callback(
    # Set callbacks:
    Output('2', 'figure'),
    [Input('wave', 'value'),
    Input('low', 'value'),
    Input('high', 'value'),
    Input('fftn', 'value'),
     Input('checklist', 'values'),
     Input('clip', 'value'),
     Input('window', 'value')]
)

def update_graph(wave, low, high, fftn, checkbox_values, clip, window):

    print(checkbox_values)

    if 'Normal' in checkbox_values:
        STFTmag, CIFpos, tremap = predict(param, wave, low, high, fftn, 'spectrogram',clip,window)
        trace1 = go.Scatter3d(
            x=tremap[:],
            y=CIFpos[:],
            z=STFTmag[:],
            mode='markers',
            visible=True,
            marker = dict(size=1,
                symbol='circle',
                color='#FF0000',
                opacity=0.8),
            name="Normal"
        )

    else:
        trace1 = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            visible=False,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(0,0,255)',
                opacity=0.8
            ),
        )



    if 'Both' in checkbox_values:
        STFTmag, CIFpos, tremap = predict(param, wave, low,high,fftn, 'both',clip,window)
        trace2 = go.Scatter3d(
            x=tremap[:],
            y=CIFpos[:],
            z=STFTmag[:],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='#585858',
                opacity=0.8
            ),
            name="Time & Freq"
        )

    else:
        trace2 = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            visible=False,
            marker=dict(
                size=1,
                symbol='circle',
                color='#FBFBEF',
                opacity=0.8
            ),
        )

    if 'Time' in checkbox_values:
        STFTmag, CIFpos, tremap = predict(param, wave, low,high,fftn, 'time',clip,window)
        trace3 = go.Scatter3d(
            x=tremap[:],
            y=CIFpos[:],
            z=STFTmag[:],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='#01DF01',
                opacity=0.8
            ),
            name="Time"
        )

    else:
        trace3 = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            visible=False,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(0,255,0)',
                opacity=0.8
            ),
        )

    if 'Freq' in checkbox_values:
        STFTmag, CIFpos, tremap = predict(param, wave, low,high,fftn, 'frequency',clip,window)
        trace4 = go.Scatter3d(
            x=tremap[:],
            y=CIFpos[:],
            z=STFTmag[:],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='#013ADF',
                opacity=0.8
            ),
            name="Freq"
        )

    else:
        trace4 = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            showlegend=False,
            visible=False,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(0,255,255)',
                opacity=0.8
            ),
        )

    return {'data': traces_2 + [trace1]+[trace2]+[trace3]+[trace4],
            'layout': layout_2}



@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)

def display_page(pathname):
    if (pathname == '') | (pathname == '/'):
        return index_page
    else:
        '404'


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
