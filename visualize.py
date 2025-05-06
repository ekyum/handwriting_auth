import numpy as np
import plotly.graph_objects as go
import pickle

# Load the data from the pickle dataset
data = np.load('/Users/haro/works/handwriting_authenticate/data/signature_data.pickle', allow_pickle=True)

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


visualize_num = int(input(f"Input the index of the sample you want to visualize (0-{len(data['samples'])-1}): "))


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