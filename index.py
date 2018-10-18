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
R = pd.read_pickle('data/ref_vowel.pckl')  # 4x(14+3)
F1_med, F2_med, F3_med = R['F1'].median(), R['F2'].median(), R['F3'].median()
with open('data/pal_pha.pckl', 'rb') as pckl:
    pal, pha = pickle.load(pckl)
artic_col = ['T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y',
             'T4x', 'T4y', 'ULx', 'ULy', 'LLx', 'LLy', 'JAWx', 'JAWy']
acous_col = ['F1', 'F2', 'F3']
vowel_list = ['IY1', 'AE1', 'AA1', 'UW1']
# Load params
X_scaler, Y_scaler, G, W = np.load('data/params.npy')


def predict(pc1, pc2, pc3, pc4, pc5):
    x_reduced = np.array([[pc1, pc2, pc3, pc4, pc5]])
    x_recon_scaled = G.inverse_transform(x_reduced)
    pellets = X_scaler.inverse_transform(x_recon_scaled)

    y_scaled = np.dot(x_reduced, W)
    formants = Y_scaler.inverse_transform(y_scaled)
    # Returns:
    #  T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy
    #  F1, F2, F3
    return pellets, formants


# -------------------- Style -------------------------- #
# Slider settings
xmin = -5.0
xmax = 5.0
xstep = 0.5
values = np.arange(xmin, xmax + xstep, xstep)

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
    'width': '25%',
    'display': 'inline-block',
}
col3_style = {
    'width': '25%',
    'display': 'inline-block',
}
row_style = {
    'display': 'inline-block',
    'padding': '0px 0px 0px 0px',
}
margins = {'l': 50, 'r': 50, 'b': 50, 't': 50}
fig_width = 350
fig_height = 300  # 350

footer_style = {
    'fontSize': '15px',
    'display': 'block',
    'textAlign': 'center',
}


def add_slider(id):
    return dcc.Slider(
        id=id,
        marks={int(i): f'{i:.1f}' for i in values
               if (i == xmin) | (i == xmax) | (i == 0.0)},
        min=xmin,
        max=xmax,
        step=xstep,
        value=0.0,
        updatemode='drag',
    )


# -------------------- Figures -------------------------- #
fig_pharynx = go.Scatter(
    x=pha[:, 0],
    y=pha[:, 1],
    mode='lines',
    name='Pharynx',
    line=dict(
        color=('rgb(0,0,0)')
    ),
    hoverinfo='none',
)
fig_palate = go.Scatter(
    x=pal[:, 0],
    y=pal[:, 1],
    mode='lines',
    name='Palate',
    line=dict(
        color=('rgb(0,0,0)')
    ),
    hoverinfo='none',
)
fig_formant3d = go.Scatter3d(
    x=R['F1'],
    y=R['F2'],
    z=R['F3'],
    mode='markers',
    visible=True,
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)
fig_f2f1 = go.Scatter(
    x=R['F2'],
    y=R['F1'],
    name='',
    mode='markers',
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)
fig_f2f3 = go.Scatter(
    x=R['F2'],
    y=R['F3'],
    name='',
    mode='markers',
    marker=dict(
        size=3,
        symbol='circle',
        color='rgb(112,112,112)',
        opacity=0.8
    ),
)

# -------------------- Traces & Layouts -------------------------- #
traces_221 = [fig_palate, fig_pharynx]
layout_221 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin=margins,
    showlegend=False,
    # Axis
    xaxis=dict(
        linecolor='black',
        mirror=True,
        title='Back <--> Front (mm)<br />Midsagittal view',
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
        title='Low <--> High (mm)',
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

traces_223 = [fig_f2f1]
layout_223 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin=margins,
    showlegend=False,
    xaxis=dict(
        title='F2 (Hz)<br />Formant space (F2-F1)',
        linecolor='black',
        mirror=True,
        range=[2200, 800],
        dtick=250,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    yaxis=dict(
        title='F1 (Hz)',
        linecolor='black',
        mirror=True,
        range=[800, 200],
        dtick=100,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    annotations=[{
        'text': v,
        'showarrow': False,
        'xref': 'x',
        'yref': 'y',
        'x': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
        'y': R.loc[R.Vowel == v, 'F1'].values[0] - 20,
    } for v in vowel_list])

traces_222 = [fig_formant3d]
layout_222 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin={'l': 10, 'r': 10, 'b': 10, 't': 10},
    showlegend=False,
    scene=dict(
        xaxis=dict(
            title='F1',
            range=[200, 800],
            dtick=100,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        yaxis=dict(
            title='F2',
            range=[1000, 2500],
            dtick=100,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        zaxis=dict(
            title='F3',
            range=[2400, 3000],
            dtick=100,
            mirror=True,
            tickfont=dict(
                size=10,
            ),
        ),
        camera=dict(
            eye=dict(x=2.5, y=2.5, z=0.8)
        ),
        aspectmode='manual',
        aspectratio={'x': 1.2, 'y': 1.2, 'z': 1.2},
        annotations=[{
            'text': v,
            'showarrow': False,
            'x': R.loc[R.Vowel == v, 'F1'].values[0] - 20,
            'y': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
            'z': R.loc[R.Vowel == v, 'F3'].values[0] - 20,
        } for v in vowel_list],
    )
)

traces_224 = [fig_f2f3]
layout_224 = go.Layout(
    height=fig_height,
    width=fig_width,
    margin=margins,
    showlegend=False,
    xaxis=dict(
        title='F2 (Hz)<br />Formant space (F2-F3)',
        linecolor='black',
        mirror=True,
        range=[2500, 750],
        dtick=250,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    yaxis=dict(
        title='F3 (Hz)',
        linecolor='black',
        mirror=True,
        anchor='x3',
        range=[3000, 2300],
        dtick=100,
        zeroline=False,
        tickfont=dict(
            size=10,
        ),
        fixedrange=True,
    ),
    annotations=[{
        'text': v,
        'showarrow': False,
        'xref': 'x',
        'yref': 'y',
        'x': R.loc[R.Vowel == v, 'F2'].values[0] - 20,
        'y': R.loc[R.Vowel == v, 'F3'].values[0] - 20,
    } for v in vowel_list],
)

# -------------------- Page layout -------------------------- #
# Set index_page layout
index_page = html.Div([
    html.H1('From Articulation to Acoustics',
            style={'textAlign': 'center'}),
    html.H3('Move sliders and see how articulation and acoustics are related!', style={
            'textAlign': 'center'}),

    html.Div(className='row', children=[

        # ---------- First column ---------- #
        html.Div(children=[
            html.H4('PCA slider', style={'textAlign': 'center'}),
            # (Top-Left) Sliders
            html.Div([
                # PC1
                html.Div([
                    html.P('PC1', style=slider_text_style),
                    html.P('(Jaw)', style=slider_dec_style),
                    add_slider('slider-pc1'),
                ], style=slider_style),
                # PC2
                html.Div([
                    html.P('PC2', style=slider_text_style),
                    html.P('(Tongue body)', style=slider_dec_style),
                    add_slider('slider-pc2'),
                ], style=slider_style),
                # PC3
                html.Div([
                    html.P('PC3', style=slider_text_style),
                    html.P('(Tongue dorsum)', style=slider_dec_style),
                    add_slider('slider-pc3'),
                ], style=slider_style),
                # PC4
                html.Div([
                    html.P('PC4', style=slider_text_style),
                    html.P('(Lips)', style=slider_dec_style),
                    add_slider('slider-pc4'),
                ], style=slider_style),
                # PC5
                html.Div([
                    html.P('PC5', style=slider_text_style),
                    html.P('(Residual)', style=slider_dec_style),
                    add_slider('slider-pc5'),
                ], style=slider_style),
            ], style={'display': 'block'}),

            # (Btm-Left) Information (TODO)
            html.Div([
                html.P('''
                This app demostrates the relationship between articulation (7 sensors: tongue tip, tongue dorsum, tongue body, tongue back, lower incisor, and lips) and acoustics (i.e., F1, F2, F3). By moving sliders, you can check how movement along the given principal component shapes the articulators and generates formant frequencies concurrently. 
                '''),
                html.P('This work was based on using the uncontrolled manifold method on speech production to roughly answer, "How can we decompose variability in speech proudction and which variability is good or bad?".'),
                html.P('UCM and CM visualization will be added soon.'),
                html.P(
                    'For more details, see:'),
                html.A("Link", href='https://goo.gl/WdUSG9', target='_blank'),
                html.Br(),
            ], style=row_style),
        ], className='three columns', style=col1_style),


        # ---------- Second column ---------- #
        html.Div(children=[
            # (Center-Top) Midsagittal view
            html.Div([
                dcc.Graph(
                    id='221',
                    figure={
                        'data': traces_221,
                        'layout': layout_221,
                    },
                ),
            ], style=row_style),

            # (Center-Btm) F2-F1 formant space
            html.Div([
                dcc.Graph(
                    id='223',
                    figure={
                        'data': traces_223,
                        'layout': layout_223,
                    },
                ),
                html.Br(),
                html.P('For UCM/CM plots, '),
                dcc.Link('Go to UCM plots', href='/apps/app_ucm'),
                html.Br(),
                dcc.Link('Go to CM plots', href='/apps/app_cm'),
            ], style=row_style),
        ], className='three columns', style=col2_style),

        # ---------- Third column --------- #
        html.Div(children=[
            # (Right-Top)
            html.Div([
                dcc.Graph(
                    id='222',
                    figure={
                        'data': traces_222,
                        'layout': layout_222
                    },
                ),
            ], style=row_style),
            # (Right-Btm)
            html.Div([
                dcc.Graph(
                    id='224',
                    figure={
                        'data': traces_224,
                        'layout': layout_224,
                    },
                ),
            ], style=row_style),
        ], className='three columns', style=col3_style),

    ]),

    # ---- Bottom of the page ---- #
    html.Div(className='row', children=[
        html.Hr(),
        html.P('Kang, Chen & Nam (2018)',
               style=footer_style),
        html.A('GitHub',
               href="https://github.com/jaekookang", target='_blank',
               style={'textAlign': 'center', 'display': 'block'}),
    ]),
])


@app.callback(
    # Set callbacks: Midsagittal view
    Output('221', 'figure'),
    [Input('slider-pc1', 'value'),
     Input('slider-pc2', 'value'),
     Input('slider-pc3', 'value'),
     Input('slider-pc4', 'value'),
     Input('slider-pc5', 'value')]
)
def update_221(pc1, pc2, pc3, pc4, pc5):
    pellets, _ = predict(pc1, pc2, pc3, pc4, pc5)
    T1x, T1y, T2x, T2y, T3x, T3y, T4x, T4y, ULx, ULy, LLx, LLy, JAWx, JAWy = pellets[0]
    trace = go.Scatter(
        x=[T1x, T2x, T3x, T4x, None, ULx, None, LLx, None, JAWx],
        y=[T1y, T2y, T3y, T4y, None, ULy, None, LLy, None, JAWy],
        name='',
        mode='lines+markers',
        line={
            'shape': 'spline',
        },
        marker={
            'size': 5,
            'color': 'rgb(0,0,0)',
        },
    )
    return {'data': traces_221 + [trace],
            'layout': layout_221}


@app.callback(
    # Set callbacks: 223
    Output('223', 'figure'),
    [Input('slider-pc1', 'value'),
     Input('slider-pc2', 'value'),
     Input('slider-pc3', 'value'),
     Input('slider-pc4', 'value'),
     Input('slider-pc5', 'value')]
)
def update_223(pc1, pc2, pc3, pc4, pc5):
    _, formants = predict(pc1, pc2, pc3, pc4, pc5)
    F1, F2, _ = formants[0]

    trace = go.Scatter(
        x=[F2],
        y=[F1],
        name='',
        mode='markers',
        marker={
            'size': 5,
            'color': 'rgb(255,0,0)'
        },
    )
    return {'data': traces_223 + [trace],
            'layout': layout_223}


@app.callback(
    Output('222', 'figure'),
    [Input('slider-pc1', 'value'),
     Input('slider-pc2', 'value'),
     Input('slider-pc3', 'value'),
     Input('slider-pc4', 'value'),
     Input('slider-pc5', 'value')]
)
def update_222(pc1, pc2, pc3, pc4, pc5):
    _, formants = predict(pc1, pc2, pc3, pc4, pc5)
    F1, F2, F3 = formants[0]

    trace = go.Scatter3d(
        x=[F1],
        y=[F2],
        z=[F3],
        mode='markers',
        visible=True,
        marker=dict(
            size=5,
            symbol='circle',
            color='rgb(255,0,0)',
            opacity=0.8
        ),
    )
    return {'data': traces_222 + [trace],
            'layout': layout_222}


@app.callback(
    Output('224', 'figure'),
    [Input('slider-pc1', 'value'),
     Input('slider-pc2', 'value'),
     Input('slider-pc3', 'value'),
     Input('slider-pc4', 'value'),
     Input('slider-pc5', 'value')]
)
def update_224(pc1, pc2, pc3, pc4, pc5):
    _, formants = predict(pc1, pc2, pc3, pc4, pc5)
    _, F2, F3 = formants[0]

    trace = go.Scatter(
        x=[F2],
        y=[F3],
        name='',
        mode='markers',
        marker={
            'size': 5,
            'color': 'rgb(255,0,0)'
        },
    )
    return {'data': traces_224 + [trace],
            'layout': layout_224}


wakemydyno_page = html.Div([
    html.H1('Test')
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/apps/app_ucm':
        return app_ucm.layout
    elif pathname == '/apps/app_cm':
        return app_cm.layout
    elif pathname == '/wakemydyno.txt':
        return wakemydyno_page
    elif (pathname == '') | (pathname == '/'):
        return index_page
    else:
        '404'


# if __name__ == '__main__':
#     app.run_server(debug=True)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
