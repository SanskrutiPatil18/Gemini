import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Title and Description based on the sources
st.title("DBSCAN Clustering Explorer")
st.write("""
This app visualizes how DBSCAN 'walks around' and finds groups based on density. 
It identifies **Core Points**, **Border Points**, and **Noise** as it expands clusters [1].
""")

# Sidebar for Parameters (Source [1])
st.sidebar.header("Parameters")
eps = st.sidebar.slider("Maximum distance (eps ε)", 0.1, 2.0, 0.5, help="Radius around a point [2]")
min_samples = st.sidebar.slider("Minimum points (min_samples)", 2, 20, 5, help="Points needed for a Core Point [1]")

# Generate synthetic data for visualization
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Run DBSCAN (External Library implementing logic from Source [3], [4])
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Identify point types (Source [1], [3])
# Core points are those found by the algorithm to meet the density criteria
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for different clusters
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for Noise Points [1]
        col = [1]

    class_member_mask = (labels == k)

    # Plot Core Points (Large dots) [3]
    xy = X[class_member_mask & core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10, label=f'Cluster {k} Core' if k != -1 else 'Noise')

    # Plot Border Points (Smaller dots) [1], [4]
    xy = X[class_member_mask & ~core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

ax.set_title(f'Estimated number of clusters: {len(unique_labels) - (1 if -1 in labels else 0)}')
st.pyplot(fig)

# Explanation of what is happening (Source [1], [2])
st.info(f"""
**How it works:**
1. The app places a circle of radius **{eps} (eps)** around each point [2].
2. If a circle contains **{min_samples} (min_samples)** or more points, it becomes a **Core Point** [1], [3].
3. The cluster expands by adding all **directly density-reachable** points (those within the circle) [4].
4. Points that are neither core nor border are marked as **Noise** (black dots) [1].
""")