# Network Anomaly Detection Dashboard

This project is a **Network Anomaly Detection Dashboard** built with [Streamlit](https://streamlit.io/), designed to visualize network traffic, generate graph embeddings, and detect anomalies using rule-based techniques. It supports datasets like UNSW-NB15, CIC-IDS2017, or synthetic network traffic data, and includes features like graph visualization, performance metrics, and attack simulation.

## Features
- **Data Upload**: Upload custom CSV datasets (e.g., UNSW-NB15, CIC-IDS2017) or use a default synthetic dataset.
- **Graph Embeddings**: Generate first-order and second-order graph embeddings based on network traffic features.
- **Visualization**: Interactive graphs (using NetworkX and Matplotlib) and anomaly frequency charts (using Plotly).
- **Rule-Based Anomaly Detection**: Detect anomalies with customizable rules for protocols like HTTP, DNS, FTP, etc.
- **Performance Metrics**: Evaluate detection performance with accuracy, precision, recall, F1-score, and confusion matrix.
- **Attack Simulation**: Simulate the attack scenarios (e.g., DoS, Port Errors, Suspicious Ports) and download generated traffic data.

## Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/veenuuu/network-anomaly-detection.git
   cd network-anomaly-detection
