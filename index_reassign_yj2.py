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
from dash.dependencies import Input, Output, Event
import plotly.graph_objs as go

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
# app = dash.Dash(__name__)
server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')
app.config.suppress_callback_exceptions = True

app.css.append_css(
    {"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])


slider_style = {
    'display': 'flex',
    'flexDirection': 'row',
    'padding': '0px 0px 10px 0px'
}
slider_text_style = {
    'padding': '0px 10px 10px 0px',
    'margin-top' : '20'
}
slider_dec_style = {
    'fontSize': '10px',
    'padding': '0px 10px 10px 0px',
    'width': '55px',
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
    'width': '75%',
    'display': 'inline-block',
}

# col3_style = {
#     'width': '25%',
#     'display': 'inline-block',
#}

row_style = {
    'display': 'inline-block',
    'width' : '70%',
    'padding': '0px 0px 0px 0px',
}

margins = {'l': 50, 'r': 50, 'b': 50, 't': 50}
fig_width1 = 1000
fig_height1 = 300
fig_width2 = 1000
fig_height2 = 500 # 350

footer_style = {
    'fontSize': '15px',
    'display': 'block',
    'textAlign': 'center',
}


# def add_slider(id, xmax, xmin, values, xstep):
#     return dcc.Slider(
#         id=id,
#         marks={int(i): f'{i:.1f}' for i in values
#                if (i == xmin) | (i == xmax) | (i == 0.0)},
#         min=xmin,
#         max=xmax,
#         step=xstep,
#         value=0.0,
#         updatemode='drag',
#     )


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
    visible=True,
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)

fig_specgram2 = go.Scatter3d(
    x=[1],
    y=[2],
    z=[3],
    mode='markers',
    visible=True,
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(0,0,240)',
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
        range=[-90, 40],
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
        range=[-30, 30],
        dtick=10,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
)


traces_2 = [fig_specgram1, fig_specgram2]
layout_2 = go.Layout(
    height=fig_height2,
    width=fig_width2,
    margin=margins,
    showlegend=False,
    # Axis
    scene=dict(
        xaxis=dict(
            title='F1',
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        yaxis=dict(
            title='F2',
            dtick=100,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        zaxis=dict(
            title='F3',
            mirror=True,
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
                id="wave",
                options=[
                    {'label': 'a', 'value': 'a'},
                    {'label': 'e', 'value': 'e'},
                    {'label': 'i', 'value': 'i'},
                    {'label': 'o', 'value': 'o'},
                    {'label': 'u', 'value': 'u'}
                ])
            ], style=param_style)
        ]),


        html.Div(children=[
            html.H4('Parameters', style={'textAlign': 'center'}),

            html.Div([
                    html.P('Value type', style=slider_text_style),
                dcc.Checklist(
                    id='types',
                    options=[
                        {'label': 'Normal', 'value': 'spectrogram'},
                        {'label': 'Time-reassigned', 'value': 'time'},
                        {'label': 'Frequency-reassigned', 'value': 'frequency'},
                        {'label': 'Time&Freq-reassigned', 'value': 'both'}
                    ],
                    values=['']
                )]
                )], style=param_style),

            # fftn
            html.Div([
                html.P('fftn tick', style=slider_text_style),
                dcc.Dropdown(
                    id="fftn",
                    options=[
                        {'label': '128', 'value': '128'},
                        {'label': '256', 'value': '256'},
                        {'label': '512', 'value': '512'},
                        {'label': '1024', 'value': '1024'}
                    ],
                    value='128'
                ),
            ], style=param_style),


                ### low cut
            html.Div([
                html.P('low cut', style=slider_text_style),
                dcc.Slider(
                    id="low",
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
                    id="high",
                    min=5000,
                    max=8000,
                    marks={i: i for i in range(5000, 8000, 1000)},
                    value=5000,
                ),
                ], style=param_style),

            html.Div([
                    html.P('clip range', style=slider_text_style),
                    dcc.Slider(
                    id="clip",
                    min=-100,
                    max=0,
                    marks={i: i for i in range(-100, 0, 10)},
                    value=0,
                ),
                ], style=param_style),

            html.Div([
                    html.P('Windowing', style=slider_text_style),
                    dcc.RadioItems(
                    id="window",
                    options=[
                        {'label': 'Rectangular', 'value': 'rectangular'},
                        {'label': 'Hamming', 'value': 'hamming'},
                        {'label': 'Hanning', 'value': 'hanning'},
                        {'label': 'Blackman', 'value': 'blackman'},
                        {'label': 'Kaiser', 'value': 'kaiser'}
                    ],
                    value='kaiser'
                ),
                ], style=param_style),


        ], style=col1_style),


        # ---------- Second column ---------- #
        html.Div(children=[
            # spectrum plot
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
                html.P('For UCM/CM plots, '),
                dcc.Link('Go to UCM plots', href='/apps/app_ucm'),
                html.Br(),
                dcc.Link('Go to CM plots', href='/apps/app_cm'),
            ], style=col2_style),
        ], className='two columns', style=row_style),
    ])
@app.callback(
    # Set callbacks:
    Output('1', 'figure'),
    [Input('wave', 'value')]
)

def update_wave(wave_type):

    return None
    #return {'data': traces_2 + [trace1]+[trace2],
    #        'layout': layout_2}

@app.callback(
    # Set callbacks:
    Output('2', 'figure'),
    [Input('fftn', 'value'),
     Input('types', 'values')])

    # [Input('wave', 'value'),
     # Input('low', 'value'),
     # Input('high', 'value'),
     # Input('clip', 'value'),
     # Input('window', 'value')])

def update_graph(fftn, types):
    #wave='./'

    low=10
    high=5000
    clip=0
    window='kaiser'

    param = {}
    param['data'] = './spectrogram/a.wav'
    param['win_size'] = 0.025
    param['win_step'] = 0.01
    param['window'] = window  # kaiser, rectangular, hamming, hanning, blackman
    param['fftn'] = int(fftn)
    param['low_cut'] = int(low)
    param['high_cut'] = int(high)
    param['clip'] = int(clip)
    param['delay'] = 1
    param['freqdelay'] = 1
    param['value_type'] = types  # both, time, frequency, spectrogram

    print(param)

    # import processing class.
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


    print(types)

    if 'both' in types:

        STFTplot1, CIFplot1, tremap1 = a_wav.retrieve_values('both')

        print(len(STFTplot1))
        print(len(CIFplot1))
        print(len(tremap1))
        # STFTmag, CIFpos, tremap = predict(param)
        tremap1 = 1000 * tremap1
        trace1 = go.Scatter3d(
            x=STFTplot1,
            y=CIFplot1,
            z=tremap1,
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(255,0,0)',
                opacity=0.8
            ),
        )

    else:
        trace1 = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(255,0,0)',
                opacity=0.5
            ),
        )

    if 'time' in types:

        STFTplot, CIFplot, tremap = a_wav.retrieve_values('time')
        tremap = 1000 * tremap
        trace2 = go.Scatter3d(
            x=STFTplot[:],
            y=CIFplot[:],
            z=tremap[:],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(0,255,0)',
                opacity=0.5
            ),
        )

    else:
        trace2 = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(0,255,0)',
                opacity=0
            ),
        )

    return {'data': traces_2 + [trace1]+[trace2],
            'layout': layout_2}

    # # if 'both' in types:
    # #     STFTplot, CIFplot, tremap = a_wav.retrieve_values('both')
    # #     tremap1000 = 1000 * tremap
    # #     trace1 = go.Scatter3d(
    # #         x=tremap1000[:],
    # #         y=CIFplot[:],
    # #         z=STFTplot[:],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(255,0,0)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # # else:
    # #     trace1 = go.Scatter3d(
    # #         x=[0],
    # #         y=[1],
    # #         z=[2],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(255, 0, 0)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # #
    # # if 'time' in types:
    # #     STFTplot, CIFplot, tremap = a_wav.retrieve_values('time')
    # #     tremap1000 = 1000 * tremap
    # #     trace2 = go.Scatter3d(
    # #         x=tremap1000[:],
    # #         y=CIFplot[:],
    # #         z=STFTplot[:],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(0,255,0)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # #
    # # else:
    # #     trace2 = go.Scatter3d(
    # #         x=[1],
    # #         y=[0],
    # #         z=[0],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(0, 255, 0)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # #
    # # if 'frequency' in types:
    # #     STFTplot, CIFplot, tremap = a_wav.retrieve_values('frequency')
    # #     tremap1000 = 1000 * tremap
    # #     trace3 = go.Scatter3d(
    # #         x=tremap1000[:],
    # #         y=CIFplot[:],
    # #         z=STFTplot[:],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(0,0,255)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # # else:
    # #     trace3 = go.Scatter3d(
    # #         x=[3],
    # #         y=[0],
    # #         z=[0],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(0, 0, 255)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # #
    # #
    # # if 'spectrogram' in types:
    # #     STFTplot, CIFplot, tremap = a_wav.retrieve_values('spectrogram')
    # #     tremap1000 = 1000 * tremap
    # #     trace4 = go.Scatter3d(
    # #         x=tremap1000[:],
    # #         y=CIFplot[:],
    # #         z=STFTplot[:],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(0,255,255)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # #
    # # else:
    # #     trace4 = go.Scatter3d(
    # #         x=[4],
    # #         y=[0],
    # #         z=[0],
    # #         mode='markers',
    # #         visible=True,
    # #         marker=dict(
    # #             size=3,
    # #             symbol='circle',
    # #             color='rgb(0, 255, 255)',
    # #             opacity=0.5
    # #         ),
    # #     )
    # return {'data': traces_2 +[trace1]+[trace2]+[trace3]+[trace4],
    #         'layout': layout_2}


# @app.callback(
#     # Set callbacks: Midsagittal view
#     Output('221', 'figure'),
#     [Input('slider-pc1', 'value'),
#      Input('slider-pc2', 'value'),
#      Input('slider-pc3', 'value'),
#      Input('slider-pc4', 'value'),
#      Input('slider-pc5', 'value')]
# )
# def update_221(pc1, pc2, pc3, pc4, pc5):
#     pellets, _ = predict(pc1, pc2, pc3, pc4, pc5)
#     T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy = pellets[0]
#     trace = go.Scatter(
#         x=[T1x, T2x, T3x, T4x, None, ULx, None, LLx, None, JAWx],
#         y=[T1y, T2y, T3y, T4y, None, ULy, None, LLy, None, JAWy],
#         name='',
#         mode='lines+markers',
#         line={
#             'shape': 'spline',
#         },
#         marker={
#             'size': 5,
#             'color': 'rgb(0,0,0)',
#         },
#     )
#     return {'data': traces_221 + [trace],
#             'layout': layout_221}
#
#
# @app.callback(
#     # Set callbacks: 223
#     Output('223', 'figure'),
#     [Input('slider-pc1', 'value'),
#      Input('slider-pc2', 'value'),
#      Input('slider-pc3', 'value'),
#      Input('slider-pc4', 'value'),
#      Input('slider-pc5', 'value')]
# )
# def update_223(pc1, pc2, pc3, pc4, pc5):
#     _, formants = predict(pc1, pc2, pc3, pc4, pc5)
#     F1, F2, _ = formants[0]
#
#     trace = go.Scatter(
#         x=[F2],
#         y=[F1],
#         name='',
#         mode='markers',
#         marker={
#             'size': 5,
#             'color': 'rgb(255,0,0)'
#         },
#     )
#     return {'data': traces_223 + [trace],
#             'layout': layout_223}
#
#
# @app.callback(
#     Output('222', 'figure'),
#     [Input('slider-pc1', 'value'),
#      Input('slider-pc2', 'value'),
#      Input('slider-pc3', 'value'),
#      Input('slider-pc4', 'value'),
#      Input('slider-pc5', 'value')]
# )
# def update_222(pc1, pc2, pc3, pc4, pc5):
#     _, formants = predict(pc1, pc2, pc3, pc4, pc5)
#     F1, F2, F3 = formants[0]
#
#     trace = go.Scatter3d(
#         x=[F1],
#         y=[F2],
#         z=[F3],
#         mode='markers',
#         visible=True,
#         marker=dict(
#             size=5,
#             symbol='circle',
#             color='rgb(255,0,0)',
#             opacity=0.8
#         ),
#     )
#     return {'data': traces_222 + [trace],
#             'layout': layout_222}
#
#
# @app.callback(
#     Output('224', 'figure'),
#     [Input('slider-pc1', 'value'),
#      Input('slider-pc2', 'value'),
#      Input('slider-pc3', 'value'),
#      Input('slider-pc4', 'value'),
#      Input('slider-pc5', 'value')]
# )
# def update_224(pc1, pc2, pc3, pc4, pc5):
#     _, formants = predict(pc1, pc2, pc3, pc4, pc5)
#     _, F2, F3 = formants[0]
#
#     trace = go.Scatter(
#         x=[F2],
#         y=[F3],
#         name='',
#         mode='markers',
#         marker={
#             'size': 5,
#             'color': 'rgb(255,0,0)'
#         },
#     )
#     return {'data': traces_224 + [trace],
#             'layout': layout_224}
#
#
wakemydyno_page = html.Div([
    html.H1('Test')
])
#
#
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)

def display_page(pathname):
    if pathname == '/apps/app_ucm':
        return app_ucm.layout
    elif pathname == '/apps/app_cm':
        return app_cm.layout
    # elif pathname == '/wakemydyno.txt':
    #     return wakemydyno_page
    elif (pathname == '') | (pathname == '/'):
        return index_page
    else:
        '404'


# if __name__ == '__main__':
#     app.run_server(debug=True)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
