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

# data and parameter
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-D", "--data", type=str, default="a.wav", help="data path")

parser.add_argument("-wsi", "--win_size", type=int, default=0.025, help="window size")
parser.add_argument("-wst", "--win_step", type=int, default=0.01, help="window step")
parser.add_argument("-win", "--window", type=str, default='kaiser', help="window method")
parser.add_argument("-fn", "--fftn", type=int, default=1024, help="the number of fft bins")
parser.add_argument("-lc", "--low_cut", type=int, default=10, help="low frequency limit")
parser.add_argument("-hc", "--high_cut", type=int, default=5000, help="higt frequency limit")
parser.add_argument("-c", "--clip", type=int, default=-30, help="Db under the clip will not be plotted")
parser.add_argument("-d", "--delay", type=int, default=1, help="audio sample delay")
parser.add_argument("-fd", "--freqdelay", type=int, default=1, help="data path")

param = parser.parse_args()
print(param)
#### param.data
def predict(param):

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
    # plot reassign spectrogram
    #a_wav.plot_Rspectrogram()
    # retrieve reassigned values.
    STFTmag, CIFpos, tremap = a_wav.retrieve_values()

    ### for plotting
    STFTplot = np.reshape(STFTmag, [STFTmag.shape[0] * STFTmag.shape[1]])
    CIFplot = np.reshape(CIFpos, [a_wav.CIFpos.shape[0] * a_wav.CIFpos.shape[1]])
    tremap = np.reshape(a_wav.tremap, [a_wav.tremap.shape[0] * a_wav.tremap.shape[1]])

    plot_these = np.where(STFTplot >= a_wav.clip) and np.where(a_wav.low_cut <= CIFplot) and np.where(
        CIFplot <= a_wav.high_cut) and \
                 np.where(a_wav.t[0] <= tremap) and np.where(tremap <= a_wav.t[-1])

    if len(STFTplot) != len(plot_these[0]):
        STFTplot = STFTplot[plot_these]
        CIFplot = CIFplot[plot_these]
        tremap = tremap[plot_these]

    f = a_wav.Fs * np.arange(a_wav.lowindex - 1, a_wav.highindex - 1) / a_wav.fftn
    
    return STFTplot, CIFplot, 1000 * tremap



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

# -------------------- Data processing -------------------------- #
# Load data
# R = pd.read_pickle('data/ref_vowel.pckl')  # 4x(14+3)
# F1_med, F2_med, F3_med = R['F1'].median(), R['F2'].median(), R['F3'].median()
# with open('data/pal_pha.pckl', 'rb') as pckl:
#     pal, pha = pickle.load(pckl)
# artic_col = ['T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y',
#              'T4x', 'T4y', 'ULx', 'ULy', 'LLx', 'LLy', 'JAWx', 'JAWy']
# acous_col = ['F1', 'F2', 'F3']
# vowel_list = ['IY1', 'AE1', 'AA1', 'UW1']
# # Load params
# X_scaler, Y_scaler, G, W = np.load('data/params.npy')
#
#
# def predict(pc1, pc2, pc3, pc4, pc5):
#     x_reduced = np.array([[pc1, pc2, pc3, pc4, pc5]])
#     x_recon_scaled = G.inverse_transform(x_reduced)
#     pellets = X_scaler.inverse_transform(x_recon_scaled)
#
#     y_scaled = np.dot(x_reduced, W)
#     formants = Y_scaler.inverse_transform(y_scaled)
#     # Returns:
#     #  T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy
#     #  F1, F2, F3
#     return pellets, formants


# -------------------- Style -------------------------- #
# Slider settings
# xmin = -5.0
# xmax = 5.0
# xstep = 0.5
# values = np.arange(xmin, xmax + xstep, xstep)

slider_style = {
    'display': 'flex',
    'flexDirection': 'row',
    'padding': '0px 0px 10px 0px'
}
slider_text_style = {
    'padding': '0px 10px 10px 0px',
}
slider_dec_style = {
    'fontSize': '10px',
    'padding': '0px 10px 10px 0px',
    'width': '55px',
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
fig_height2 = 500# 350

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
### testing
# STFTmag, CIFpos, tremap = predict(param)

fig_specgram = go.Scatter3d(
    x=[0],
    y=[0],
    z=[0],
    mode='markers',
    visible=True,
    marker=dict(
        size=1,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=1
    ),
)

# fig_specgram = go.Scatter3d(
#     x=0,
#     y=0,
#     z=0,
#     mode='lines', ### input 받게 하자
#     name='specgram',
#     line=dict(
#         color=('rgb(0,0,0)')
#     ),
#     hoverinfo='none',
# )
#
# fig_formant3d = go.Scatter3d(
#     x=R['F1'],
#     y=R['F2'],
#     z=R['F3'],
#     mode='markers',
#     visible=True,
#     marker=dict(
#         size=3,
#         symbol='circle',
#         color='rgb(112,112,112)',
#         opacity=0.8
#     ),
# )
# fig_f2f1 = go.Scatter(
#     x=R['F2'],
#     y=R['F1'],
#     name='',
#     mode='markers',
#     marker=dict(
#         size=3,
#         symbol='circle',
#         color='rgb(112,112,112)',
#         opacity=0.8
#     ),
# )
# fig_f2f3 = go.Scatter(
#     x=R['F2'],
#     y=R['F3'],
#     name='',
#     mode='markers',
#     marker=dict(
#         size=3,
#         symbol='circle',
#         color='rgb(112,112,112)',
#         opacity=0.8
#     ),
# )

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
        title='Time (sec)',
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
        title='Amplitude (decibel)',
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


traces_2 = [fig_specgram]
layout_2 = go.Layout(
    height=fig_height2,
    width=fig_width2,
    margin=margins,
    showlegend=False,
    # Axis
    xaxis=dict(
        linecolor='black',
        mirror=True,
        title='Spectrogram',
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
        title='Y axis',
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

# layout_2 = go.Layout(
#     height=fig_height,
#     width=fig_width,
#     margin={'l': 10, 'r': 10, 'b': 10, 't': 10},
#     showlegend=False,
#     scene=dict(
#         xaxis=dict(
#             title='F1',
#             range=[200, 800],
#             dtick=100,
#             mirror=True,
#             tickfont=dict(
#                 size=10,
#             ),
#         ),
#         yaxis=dict(
#             title='F2',
#             range=[1000, 2500],
#             dtick=100,
#             mirror=True,
#             tickfont=dict(
#                 size=10,
#             ),
#         ),
#         zaxis=dict(
#             title='F3',
#             range=[2400, 3000],
#             dtick=100,
#             mirror=True,
#             tickfont=dict(
#                 size=10,
#             ),
#         ),
#         camera=dict(
#             eye=dict(x=2.5, y=2.5, z=0.8)
#         ),
#         aspectmode='manual',
#         aspectratio={'x': 1.2, 'y': 1.2, 'z': 1.2},
#         annotations=[{
#             'text': v,
#             'showarrow': False,
#             'x': R.loc[R.Vowel == v, 'F1'].values[0] - 20,
#             'y': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
#             'z': R.loc[R.Vowel == v, 'F3'].values[0] - 20,
#         } for v in vowel_list],
#     )
# )


# traces_221 = [fig_palate, fig_pharynx]
# layout_221 = go.Layout(
#     height=fig_height,
#     width=fig_width,
#     margin=margins,
#     showlegend=False,
#     # Axis
#     xaxis=dict(
#         linecolor='black',
#         mirror=True,
#         title='Back <--> Front (mm)<br />Midsagittal view',
#         range=[-90, 40],
#         dtick=20,
#         tickangle=0,
#         zeroline=False,
#         tickfont=dict(
#             size=10,
#         ),
#         fixedrange=True,
#     ),
#     yaxis=dict(
#         title='Low <--> High (mm)',
#         linecolor='black',
#         mirror=True,
#         range=[-30, 30],
#         dtick=10,
#         zeroline=False,
#         tickfont=dict(
#             size=10,
#         ),
#         fixedrange=True,
#     ),
# )
#
# traces_223 = [fig_f2f1]
# layout_223 = go.Layout(
#     height=fig_height,
#     width=fig_width,
#     margin=margins,
#     showlegend=False,
#     xaxis=dict(
#         title='F2 (Hz)<br />Formant space (F2-F1)',
#         linecolor='black',
#         mirror=True,
#         range=[2200, 800],
#         dtick=250,
#         zeroline=False,
#         tickfont=dict(
#             size=10,
#         ),
#         fixedrange=True,
#     ),
#     yaxis=dict(
#         title='F1 (Hz)',
#         linecolor='black',
#         mirror=True,
#         range=[800, 200],
#         dtick=100,
#         zeroline=False,
#         tickfont=dict(
#             size=10,
#         ),
#         fixedrange=True,
#     ),
#     annotations=[{
#         'text': v,
#         'showarrow': False,
#         'xref': 'x',
#         'yref': 'y',
#         'x': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
#         'y': R.loc[R.Vowel == v, 'F1'].values[0] - 20,
#     } for v in vowel_list])
#
# traces_222 = [fig_formant3d]
# layout_222 = go.Layout(
#     height=fig_height,
#     width=fig_width,
#     margin={'l': 10, 'r': 10, 'b': 10, 't': 10},
#     showlegend=False,
#     scene=dict(
#         xaxis=dict(
#             title='F1',
#             range=[200, 800],
#             dtick=100,
#             mirror=True,
#             tickfont=dict(
#                 size=10,
#             ),
#         ),
#         yaxis=dict(
#             title='F2',
#             range=[1000, 2500],
#             dtick=100,
#             mirror=True,
#             tickfont=dict(
#                 size=10,
#             ),
#         ),
#         zaxis=dict(
#             title='F3',
#             range=[2400, 3000],
#             dtick=100,
#             mirror=True,
#             tickfont=dict(
#                 size=10,
#             ),
#         ),
#         camera=dict(
#             eye=dict(x=2.5, y=2.5, z=0.8)
#         ),
#         aspectmode='manual',
#         aspectratio={'x': 1.2, 'y': 1.2, 'z': 1.2},
#         annotations=[{
#             'text': v,
#             'showarrow': False,
#             'x': R.loc[R.Vowel == v, 'F1'].values[0] - 20,
#             'y': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
#             'z': R.loc[R.Vowel == v, 'F3'].values[0] - 20,
#         } for v in vowel_list],
#     )
# )
#
# traces_224 = [fig_f2f3]
# layout_224 = go.Layout(
#     height=fig_height,
#     width=fig_width,
#     margin=margins,
#     showlegend=False,
#     xaxis=dict(
#         title='F2 (Hz)<br />Formant space (F2-F3)',
#         linecolor='black',
#         mirror=True,
#         range=[2500, 750],
#         dtick=250,
#         zeroline=False,
#         tickfont=dict(
#             size=10,
#         ),
#         fixedrange=True,
#     ),
#     yaxis=dict(
#         title='F3 (Hz)',
#         linecolor='black',
#         mirror=True,
#         anchor='x3',
#         range=[3000, 2300],
#         dtick=100,
#         zeroline=False,
#         tickfont=dict(
#             size=10,
#         ),
#         fixedrange=True,
#     ),
#     annotations=[{
#         'text': v,
#         'showarrow': False,
#         'xref': 'x',
#         'yref': 'y',
#         'x': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
#         'y': R.loc[R.Vowel == v, 'F3'].values[0] - 20,
#     } for v in vowel_list],
# )

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
            html.H4('Parameters', style={'textAlign': 'center'}),

            # fftn
            html.Div([
                dcc.Dropdown(
                    options=[
                        {'label': '128', 'value': '128'},
                        {'label': '256', 'value': '256'},
                        {'label': '512', 'value': '512'},
                        {'label': '1024', 'value': '1024'}
                    ],
                    value='128'
                ),
            ], style={'display': 'block', 'padding': '10px 10px 10px 10px'}),
                ### low cut
            html.Div([dcc.Slider(
                    min=10,
                    max=100,
                    marks={i: i for i in [10,20,30,40,50,60,70,80,90,100]},
                    value=10,

                ),
            ], style={'display': 'block', 'padding': '10px 10px 10px 10px'}),

                ### high cut
            html.Div([
                    html.P('high cut', style=slider_text_style),
                    dcc.Slider(
                    min=5000,
                    max=8000,
                    marks={i: i for i in [5000,6000,7000,8000]},
                    value=5000,
                ),

                ##### 수정하기
                dcc.Checklist(
                    id='checklist',
                    options=[
                        {'label': 'freq', 'value': 'F'},
                        {'label': 'time', 'value': 'T'},
                        {'label': 'Freq+time change', 'value': 'FT'}
                    ],
                    values=['']
                ),

                dcc.Input(
                    placeholder='Enter a value...',
                    type='text',
                    value=''
                )

            ], style={'display': 'block', 'padding': '10px 10px 10px 10px'}),
        ], className='two columns', style=col1_style),


        # ---------- Second column ---------- #
        html.Div(children=[
            # (Center-Top) Midsagittal view
            html.Div([
                dcc.Graph(
                    id='1',
                    figure={
                        'data': traces_1,
                        'layout': layout_1,
                    },
                ),
            ], style=col2_style),

            # 2nd figure
            html.Div([
                dcc.Graph(
                    id='2',
                    figure={
                        'data': traces_2,
                        'layout': layout_2,
                    },
                ),
            ], style=col2_style),
        ], className='two columns', style=row_style),
    ]),
])


@app.callback(
    # Set callbacks:
    Output('2', 'figure'),
    [Input('checklist', 'values')]
)

def update_graph(checkbox_values):

    print(checkbox_values)
    if 'FT' in checkbox_values:
        STFTmag, CIFpos, tremap = predict(param)
        trace = go.Scatter3d(
            x=STFTmag[:],
            y=CIFpos[:],
            z=tremap[:],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(255,0,0)',
                opacity=0
            ),
        )

    else:
        trace = go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            visible=True,
            marker=dict(
                size=1,
                symbol='circle',
                color='rgb(255,0,0)',
                opacity=0
            ),
        )

    return {'data': traces_2 + [trace],
            'layout': layout_2}


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
