import numpy as np
import plotly.graph_objects as go
import pickle, argparse, json


parser = argparse.ArgumentParser(description='Visualize the Handwriting Data (X, Y, Force)')
parser.add_argument('dataset', type=str, default='class_a', help='Dataset name (e.g class_a, class_b)')
parser.add_argument('index', type=int, default=0, help='Index of the sample to visualize')

args = parser.parse_args()

with open(args.dataset + ".json", 'r') as f:
    data = json.load(f)

with open(args.dataset + ".pickle", 'wb') as f:
    pickle.dump(data, f)

# Load the data from the pickle dataset
data = np.load(args.dataset + ".pickle", allow_pickle=True)

Samples = []

Writer = data['writerCode']
for i, sample in enumerate(data['samples']):
    x, y, force = [], [], []
    for stroke in data['samples'][i]['strokes']:
        for point in stroke['points']:
            # Access x, y, force, timeOffset, size, opacity, azimuth, altitude
            # Append the points to the lists
            x.append(point['x'])
            y.append(point['y'])
            force.append(point['force'])
        x.append(None)
        y.append(None)
        force.append(None)
    Samples.append([x, y, force])


visualize_num = args.index
if visualize_num >= len(Samples):
    raise ValueError(f"Index {visualize_num} is out of range for the dataset with {len(Samples)} samples.")


# Plotting the points

fig = go.Figure(data=go.Scatter3d(
    x = Samples[visualize_num][0],
    y = Samples[visualize_num][1],
    z = Samples[visualize_num][2],
    mode='lines+markers',
    marker=dict(
        size=5,
        color='rgba(0, 0, 255, 0.5)',
        line=dict(width=0.5)
    ),
    line=dict(
        width=2,
        color='rgba(0, 0, 255, 0.5)'
    )
))

fig.update_layout(
    title=f'Stroke {visualize_num} - Writer: {Writer}',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Force'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    scene_aspectmode='cube',
)

fig.show()