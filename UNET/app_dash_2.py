import os
import cv2
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import skimage.morphology as morph
import skimage.feature as feature


# Load Image Function
def load_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


# Skeletonization
def apply_skeletonization(img):
    img = (img > 127).astype(np.uint8)  # Ensure binary format
    skeleton = morph.skeletonize(img) * 255
    return skeleton.astype(np.uint8)


# Bifurcation Detection
def detect_bifurcations(img, color, point_size=3, include_skeleton=False):
    skeleton = apply_skeletonization(img)
    bifurcation_points = feature.corner_harris(skeleton, method='k', sigma=1)
    bifurcation_points = (bifurcation_points > 0.01 * bifurcation_points.max()) * 255

    # Create an empty image if skeleton is NOT included
    result = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR) if include_skeleton else np.zeros_like(img)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    y, x = np.where(bifurcation_points > 0)
    for i in range(len(x)):
        size = point_size - (i % 2)
        cv2.circle(result, (x[i], y[i]), size, color, -1)
    return result


# Encode Image for Dash
def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


# App Setup
app = dash.Dash(__name__)
app.title = "Retina Analysis App"

# Default Image Paths
BINARY_IMAGE_PATH = "../Classification/RIDB_SEGM/IM000001_1_bin_seg.png"
GRAYSCALE_IMAGE_PATH = "../Classification/RIDB_SEGM/IM000001_1_seg.png"

# Layout
app.layout = html.Div([
    html.H1("Retina Blood Vessel Analysis", style={'textAlign': 'center'}),

    html.Div([
        html.H3("Binary Image"),
        html.Img(id='original-binary', src=encode_image(load_image(BINARY_IMAGE_PATH)),
                 style={'width': '40%', 'display': 'inline-block'}),
        html.Img(id='processed-binary', src=encode_image(load_image(BINARY_IMAGE_PATH)),
                 style={'width': '40%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Grayscale Image"),
        html.Img(id='original-grayscale', src=encode_image(load_image(GRAYSCALE_IMAGE_PATH)),
                 style={'width': '40%', 'display': 'inline-block'}),
        html.Img(id='processed-grayscale', src=encode_image(load_image(GRAYSCALE_IMAGE_PATH)),
                 style={'width': '40%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Overlay Comparison"),
        html.Img(id='overlay-bifurcations', style={'width': '40%', 'display': 'inline-block'}),
        html.Img(id='overlay-skeleton-bifurcations', style={'width': '40%', 'display': 'inline-block'})
    ]),

    html.Button('Skeletonize', id='skeleton-btn', n_clicks=0),
    html.Button('Detect Bifurcations', id='bifurcation-btn', n_clicks=0),
    html.Button('Overlay Both', id='overlay-btn', n_clicks=0),
    html.Button('Download Overlay Image', id='download-btn', n_clicks=0),
    dcc.Download(id='download-image')
])


# Callback to update images
@app.callback(
    [Output('processed-binary', 'src'), Output('processed-grayscale', 'src'),
     Output('overlay-bifurcations', 'src'), Output('overlay-skeleton-bifurcations', 'src')],
    [Input('skeleton-btn', 'n_clicks'),
     Input('bifurcation-btn', 'n_clicks'),
     Input('overlay-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_images(skeleton_clicks, bifurcation_clicks, overlay_clicks):
    bin_img = load_image(BINARY_IMAGE_PATH)
    gray_img = load_image(GRAYSCALE_IMAGE_PATH)
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    overlay_bifurcations = np.zeros_like(bin_img)
    overlay_skeleton_bifurcations = np.zeros_like(bin_img)

    if button_id == 'skeleton-btn':
        bin_img = apply_skeletonization(bin_img)
        gray_img = apply_skeletonization(gray_img)
    elif button_id == 'bifurcation-btn':
        bin_img = detect_bifurcations(bin_img, (255, 0, 0))  # Red for binary
        gray_img = detect_bifurcations(gray_img, (0, 0, 255))  # Blue for grayscale
    elif button_id == 'overlay-btn':
        bin_bifurcations = detect_bifurcations(bin_img, (255, 0, 0))  # Red bifurcations
        gray_bifurcations = detect_bifurcations(gray_img, (0, 0, 255))  # Blue bifurcations
        overlay_bifurcations = cv2.addWeighted(bin_bifurcations, 0.5, gray_bifurcations, 0.5, 0)

        skeleton_three_channel = cv2.cvtColor(apply_skeletonization(bin_img), cv2.COLOR_GRAY2BGR)
        overlay_skeleton_bifurcations = cv2.addWeighted(skeleton_three_channel, 0.5, overlay_bifurcations, 0.5, 0)

    return encode_image(bin_img), encode_image(gray_img), encode_image(overlay_bifurcations), encode_image(
        overlay_skeleton_bifurcations)


# Callback for downloading the second overlayed image
@app.callback(
    Output('download-image', 'data'),
    Input('download-btn', 'n_clicks'),
    State('overlay-skeleton-bifurcations', 'src'),
    prevent_initial_call=True
)
def download_overlay_image(n_clicks, overlay_skeleton_bifurcations_src):
    overlay_skeleton_bifurcations_data = base64.b64decode(overlay_skeleton_bifurcations_src.split(',')[1])
    return dcc.send_bytes(overlay_skeleton_bifurcations_data, "overlay_skeleton_bifurcations.png")


# Run Server
if __name__ == '__main__':
    app.run_server(debug=True)
