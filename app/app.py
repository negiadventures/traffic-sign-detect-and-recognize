import base64
import glob
import hashlib
import os

import dash
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html as dngr
import numpy as np
import pandas as pd
from cv2 import cv2
from dash import dcc, Output, Input, State
from dash import html
from flask import Flask, render_template
from flask_caching import Cache
from waitress import serve

from yolov5 import detect

app = Flask(__name__)
dash_header = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    <script src="../static/js/script.js"></script>
    <script src="../static/js/lib/jquery-ui.js'"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <link rel="stylesheet" type="text/css" href="../static/css/lib/jquery-ui.css"/>
    <link rel="stylesheet" type="text/css" href="../static/css/style.css"/>
</head>
<body>
<div class="header">
    <h2>Road Sign Detection and Recognition</h2>
</div>
<div id="navbar">
    <a class="$home" href="/">Home</a>
    <a class="$analytics" href="/model">Models</a>
</div>
<div class="content">
    <div class="container">

"""

dash_footer = """

</div>
<script>
window.onscroll = function() {myFunction()};

var navbar = document.getElementById("navbar");
var sticky = navbar.offsetTop;

function myFunction() {
  if (window.pageYOffset >= sticky) {
    navbar.classList.add("sticky")
  } else {
    navbar.classList.remove("sticky");
  }
}

</script>
</body>

</html>
<!doctype html>

"""

CONTENT_STYLE = {
    'position': 'relative',
    'float': 'left',
    'margin-left': '25px',
    'margin-right': '3%',
    'padding': '10px 10px',
    'width': '70%',
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}
SIDEBAR_STYLE = {
    'position': 'relative',
    'float': 'left',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '25%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa',
    'padding-top': '0px',
    'padding-left': '20px'
}


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message='ERR_PAGE_NOT_FOUND'), 404


TIMEOUT = 60
dash_app_home = dash.Dash(__name__, server=app, url_base_pathname='/home/')
dash_app_home.title = 'Home'
dash_app_analytics = dash.Dash(__name__, server=app, url_base_pathname='/model/')
dash_app_analytics.title = 'Analytics'

cache_home = Cache(dash_app_home.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

cache_analytics = Cache(dash_app_analytics.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})


@cache_analytics.memoize(timeout=TIMEOUT)
@cache_home.memoize(timeout=TIMEOUT)
def get_cnn_accuracy_data():
    df = pd.read_csv('cnn_model_accuracy.csv')
    return df


@cache_analytics.memoize(timeout=TIMEOUT)
@cache_home.memoize(timeout=TIMEOUT)
def get_yolo_accuracy_data():
    df = pd.read_csv('yolo_model_accuracy.csv')
    return df


dash_app_home.layout = html.Div([
    # div for static header
    dngr.DangerouslySetInnerHTML(
        dash_header.replace('$home', 'active').replace('$analytics', 'inactive')),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '98%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'background-color': '#f8f9fa'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='output-data-upload', children=[
            html.Div(
                [html.H4('Please upload the file above to view some stats.', style={'textAlign': 'center'})
                 ])

        ]),
    ], style={'background-color': '#f8f9fa'}),
    dngr.DangerouslySetInnerHTML(
        dash_footer)
])

dash_app_analytics.layout = html.Div([
    # div for static header
    dngr.DangerouslySetInnerHTML(
        dash_header.replace('$home', 'inactive').replace('$analytics', 'active')),
    html.Div(
        children=[
            html.Div(
                [
                    html.H1('Models Summary', style=TEXT_STYLE),

                    html.H2('Model 1: Sign Detection (YOLO)'),
                    html.Br(),
                    html.Img(width=800, src='assets/results_yolo.png'),
                    html.H4('Epochs vs Metrics'),
                    html.Br(),
                    html.Br(),
                    html.Img(width=900, src='assets/yolo_val_batch2_pred.jpg'),
                    html.H4('Sample Batch Prediction'),
                    html.Br(),
                    html.Br(),

                    html.H2('Model 2: Sign Recognition (CNN)'),
                    html.Br(),
                    html.Img(width=700, src='assets/cnn_model summary.png'),
                    html.H4('CNN Model Summary'),
                    html.Br(),
                    html.Br(),
                    html.Img(width=800, src='assets/cnn-model.png'),
                    html.H4('CNN Model'),
                    html.Br(),
                    html.Br(),
                    html.Img(width=800, src='assets/results_cnn.png'),
                    html.H4('Epochs vs Accuracy(left) & Epochs vs Loss(left)')

                ]),
        ], style={
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background-color': '#f8f9fa'
        }),
    dngr.DangerouslySetInnerHTML(
        dash_footer)
])


def cleanup():
    lis = glob.glob('temp/*.jpg')
    lis.extend(glob.glob('temp/*.csv'))
    lis.extend(glob.glob('static/video/*.*'))
    lis.extend(glob.glob('yolov5/out/*.jpg'))
    for im in lis:
        os.remove(im)


content_image = html.Div([
    html.Hr(),
    dcc.Loading(
        id="loading",
        children=[dbc.Row(
            [
                dbc.Col(
                    html.Div([html.Center(html.Img(id="image_org", width=580))],
                             style={'width': '600px', 'position': 'relative', 'float': 'left',
                                    'margin-left': '10px', 'margin-bottom': '50px'}),
                ),
                dbc.Col(
                    html.Div([html.Center(html.Img(id="image_updated", width=580))],
                             style={'width': '600px', 'position': 'relative', 'float': 'left',
                                    'margin-left': '10px', 'margin-bottom': '50px'}),
                ),
            ]
        )]),
],
    style=CONTENT_STYLE)

content_video = html.Div([
    html.Hr(),
    dcc.Loading(
        id="loading",
        children=[dbc.Row(
            [
                dbc.Col(
                    html.Div([html.Center(html.Video(id="video_org", style={'width': '580px'}, controls=True))],
                             style={'width': '600px', 'position': 'relative', 'float': 'left',
                                    'margin-left': '10px', 'margin-bottom': '50px'}),
                ),
                dbc.Col(
                    html.Div([html.Center(html.Video(id="video_updated", style={'width': '580px'}, controls=True))],
                             style={'width': '600px', 'position': 'relative', 'float': 'left',
                                    'margin-left': '10px', 'margin-bottom': '50px'})),

            ]
        )]),
],
    style=CONTENT_STYLE)

sidebar = html.Div(
    [
        html.Div(id='sidebar_data'),
        html.Div(id='sidebar_data_vid')
    ],
    style=SIDEBAR_STYLE,
)


@dash_app_home.callback(Output("loading-data", "children"),

                        Input('upload-data', 'contents'),
                        Input('loading', "value"),
                        State('upload-data', 'filename'),
                        State('upload-data', 'last_modified')
                        )
def loading(list_of_contents, loading, list_of_names, list_of_dates):
    return list_of_names


# upload image
@dash_app_home.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    # cleanup()
    if list_of_contents is not None and len(list_of_contents) > 0:
        if list_of_contents.startswith('data:image'):
            children = [sidebar, content_image]
            return children
        else:
            children = [sidebar, content_video]
            return children

    else:
        return [
            html.Div(
                [
                    html.Br(),
                    html.H6('Please upload the file above to view some stats.', style={'textAlign': 'center'})
                ],
            )

        ]


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)


@dash_app_home.callback(
    Output('image_org', 'src'),
    Input('upload-data', 'contents'),
    Input('loading', "value"))
def view_org(content, loading):
    if content.startswith('data:image'):
        return content
    else:
        pass


def get_summary(type):
    df = pd.read_csv('yolov5/out/detections.csv')
    img_array = None
    with open('yolov5/out/detections.npy', 'rb') as f:
        img_array = np.load(f, allow_pickle=True)
    children = [html.H3('Detection Summary', style=TEXT_STYLE),
                html.Hr(), ]
    if type == 'img':
        for i in range(len(img_array)):
            img = img_array[i]
            # img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
            _, buffer = cv2.imencode('.png', img)
            children.extend(
                [
                    html.Div([
                        html.Div(
                            [html.Img(id="image_updated", width=40, src='data:image/png;base64,' + str(base64.b64encode(buffer).decode("utf-8")))],
                            style={'width': '45px', 'position': 'relative', 'float': 'left',
                                   'margin-left': '5px', 'margin-top': '15px'}),
                        html.Div([html.B(html.H5('Detection Confidence(Model 1): ' + str(df['model_1_sign_confidence'].iloc[i])))],
                                 style={'width': '250px', 'position': 'relative', 'float': 'left',
                                        'margin-left': '15px', 'margin-top': '5px'}),
                        html.Div([html.H5('Sign Recognized(Model 2): ' + str(df['model_2_detected_sign'].iloc[i]))],
                                 style={'width': '250px', 'position': 'relative', 'float': 'left', 'margin-left': '15px'
                                        })]
                        , style={'position': 'relative', 'float': 'left'}
                    )
                ])
        return children
    else:
        data = {}
        for i in range(len(df)):
            key = hashlib.md5((str(df['model_2_detected_sign'].iloc[i])).encode()).hexdigest()
            if key in data:
                data[key]['count'] += 1
                if data[key]['model_1_sign_confidence'] < df['model_1_sign_confidence'].iloc[i]:
                    data[key]['image'] = img_array[i]
                    data[key]['model_1_sign_confidence'] = df['model_1_sign_confidence'].iloc[i]
                    data[key]['model_2_detected_sign'] = str(df['model_2_detected_sign'].iloc[i])
            else:
                data[key] = {}
                data[key]['image'] = img_array[i]
                data[key]['model_1_sign_confidence'] = df['model_1_sign_confidence'].iloc[i]
                data[key]['model_2_detected_sign'] = str(df['model_2_detected_sign'].iloc[i])
                data[key]['count'] = 1
        data = sorted(data.items(), key=lambda x: x[1]['count'], reverse=True)
        for d in data:
            img = d[1]['image']
            # img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
            _, buffer = cv2.imencode('.png', img)
            children.extend(
                [
                    html.Div([
                        html.Div(
                            [html.Img(id="image_updated", width=40, src='data:image/png;base64,' + str(base64.b64encode(buffer).decode("utf-8")))],
                            style={'width': '45px', 'position': 'relative', 'float': 'left',
                                   'margin-left': '5px', 'margin-top': '15px'}),
                        html.Div([html.B(html.H5('Total Count: ' + str(d[1]['count'])))],
                                 style={'width': '250px', 'position': 'relative', 'float': 'left',
                                        'margin-left': '15px', 'margin-top': '5px'}),
                        html.Div([html.B(html.H5('Max Confidence(Model 1): ' + str(d[1]['model_1_sign_confidence'])))],
                                 style={'width': '250px', 'position': 'relative', 'float': 'left',
                                        'margin-left': '15px', 'margin-top': '5px'}),
                        html.Div([html.H5('Sign Recognized(Model 2): ' + d[1]['model_2_detected_sign'])],
                                 style={'width': '250px', 'position': 'relative', 'float': 'left', 'margin-left': '15px'
                                        })]
                        , style={'position': 'relative', 'float': 'left'}
                    )
                ])
        return children


@dash_app_home.callback(
    Output('image_updated', 'src'),
    Output('sidebar_data', 'children'),
    Input('upload-data', 'contents'),
    Input('loading', "value"))
def view_updated(content, loading):
    if content.startswith('data:image'):
        # cleanup()
        content = content.replace('data:image/png;base64,', '')
        with open("temp/image.jpg", 'wb') as f:
            f.write(base64.b64decode(content))
        detect.detect_object('temp/image.jpg', 'image')
        with open("yolov5/out/out_image.jpg", "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
        children = get_summary('img')
        return 'data:image/png;base64,' + str(my_string.decode("utf-8")), children
    else:
        pass


@dash_app_home.callback(
    Output('video_org', 'src'),
    Input('upload-data', 'contents'),
    Input('loading', "value"))
def view_org_video(content, loading):
    if content.startswith('data:image'):
        pass
    else:
        # cleanup()
        content = content.replace('data:video/quicktime;base64,', '')
        with open("static/video/video.mov", 'wb') as f:
            f.write(base64.b64decode(content))
        return "static/video/video.mov"


@dash_app_home.callback(
    Output('video_updated', 'src'),
    Output('sidebar_data_vid', 'children'),
    Input('upload-data', 'contents'),
    Input('loading', "value"), )
def view_updated_video(content, loading):
    if content.startswith('data:image'):
        pass
    else:
        detect.detect_object("static/video/video.mov", 'video')
        children = get_summary('vid')
        return 'static/video/out_video.mp4', children


@app.route('/')
@app.route("/home/")
def home():
    return dash_app_home.index()


@app.route('/model')
def analytics():
    return dash_app_analytics.index()


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    app.register_error_handler(404, page_not_found)
    app.debug = True
    serve(app, port=9091)
