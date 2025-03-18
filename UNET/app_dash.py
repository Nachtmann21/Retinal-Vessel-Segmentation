import os
import cv2
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64

# Image path
path_to_img = '../Classification/RIDB_SEGM/IM000001_1_bin_seg.png'

def load_image():
    img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
    return img

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_skeletonization(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skel

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{encoded}'

app = dash.Dash(__name__)

def serve_layout():
    img = load_image()
    return html.Div([
        html.Div([
            html.Img(id='original-image', src=encode_image(img), style={'width': '45%', 'display': 'inline-block'}),
            html.Img(id='processed-image', src=encode_image(img), style={'width': '45%', 'display': 'inline-block'})
        ]),
        html.Button('Apply CLAHE', id='clahe-btn', n_clicks=0),
        html.Button('Apply Gaussian Blur', id='blur-btn', n_clicks=0),
        html.Button('Skeletonize', id='skeleton-btn', n_clicks=0)
    ])

app.layout = serve_layout

@app.callback(
    Output('processed-image', 'src'),
    [Input('clahe-btn', 'n_clicks'),
     Input('blur-btn', 'n_clicks'),
     Input('skeleton-btn', 'n_clicks')]
)
def update_image(clahe_clicks, blur_clicks, skeleton_clicks):
    img = load_image()
    ctx = dash.callback_context
    if not ctx.triggered:
        return encode_image(img)
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'clahe-btn':
        img = apply_clahe(img)
    elif button_id == 'blur-btn':
        img = apply_gaussian_blur(img)
    elif button_id == 'skeleton-btn':
        img = apply_skeletonization(img)
    return encode_image(img)

if __name__ == '__main__':
    app.run_server(debug=True)
