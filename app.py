import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

#testing
# Set page configuration
st.set_page_config(page_title="Network Anomaly Detection ", layout="wide")

# Title and description
st.title("Network Anomaly Detection")
# st.markdown("""
# This dashboard visualizes network traffic using graph embeddings and detects anomalies using rule-based methods.
# """)

# Define risk levels and colors
RISK_LEVELS = {
    'HIGH': '#B71C1C',    # Deep Red
    'MODERATE': '#4A148C', # Deep Purple
    'LOW': '#1A237E'      # Deep Blue
}

# Map anomaly types to risk levels
ANOMALY_RISK_MAPPING = {
    'Potential DoS': 'HIGH',
    'Potential Exploit': 'HIGH', 
    'HTTP Potential DoS': 'HIGH',
    'Reconnaissance (Scanning)': 'HIGH',
    'Sequential Port Scanning': 'HIGH',
    'DNS Long Duration': 'MODERATE',
    'DNS Large Bytes': 'MODERATE',
    'HTTP Fast Large Transfer': 'MODERATE',
    'FTP Fast Large Transfer': 'MODERATE',
    'No Service Large Bytes': 'MODERATE',
    'Excessive Duration': 'LOW',
    'High Byte Count': 'LOW',
    'Unidirectional Traffic (Src to Dst)': 'LOW',
    'Unidirectional Traffic (Dst to Src)': 'LOW',
    'Suspicious Port': 'LOW',
    'DNS Wrong Port': 'LOW',
    'HTTP Wrong Port': 'LOW',
    'FTP Wrong Port': 'LOW'
}

# Helper function to determine color based on risk level
def color_risk(flags):
    if isinstance(flags, list):
        max_risk = 'LOW'
        for flag in flags:
            risk = ANOMALY_RISK_MAPPING.get(flag, 'LOW')
            if risk == 'HIGH':
                return f'background-color: {RISK_LEVELS["HIGH"]}; opacity: 0.3'
            elif risk == 'MODERATE' and max_risk != 'HIGH':
                max_risk = 'MODERATE'
        return f'background-color: {RISK_LEVELS[max_risk]}; opacity: 0.3'
    return ''

# --- Step 1: Data Loading and Preprocessing ---
st.header("Dataset Statistics")
uploaded_file = st.file_uploader("", type="csv")

@st.cache_data
def load_and_process_data(file):
    data_path = file if file else "synthetic_network_traffic.csv"
    df = pd.read_csv(data_path, low_memory=False)

    column_map = {
        'source_ip': 'src_ip',
        'destination_ip': 'dst_ip', 
        'source_bytes': 'src_bytes',
        'destination_bytes': 'dst_bytes',
        'label': 'attack_type'
    }
    df.rename(columns=column_map, inplace=True)

    columns = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'connection_state',
               'duration', 'src_bytes', 'dst_bytes', 'src_packets', 'dst_packets',
               'src_ttl', 'dst_ttl', 'service', 'attack_type']
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    numerical_columns = ['src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
                         'src_packets', 'dst_packets', 'src_ttl', 'dst_ttl']
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2025-03-27', periods=len(df), freq='S')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert attack_type to binary (0 for normal, 1 for attack)
    df['attack_type'] = df['attack_type'].apply(lambda x: 0 if str(x).lower() in ['normal', '0', 'benign'] else 1)

    return df

if uploaded_file or st.button("Use Default Dataset"):
    df = load_and_process_data(uploaded_file)

    stats_data = {
        "Metric": ["Total Samples"],
        "Value": [len(df)]
    }
    st.table(pd.DataFrame(stats_data))

    # --- Step 2: Graph Embeddings Visualization ---
    def generate_graph_embeddings(df):
        # Prepare nodes and initial embeddings
        if 'User' not in df.columns:
            vocab = df['src_ip'].unique().tolist()
            embeddings_df = df.groupby('src_ip')[['src_bytes', 'dst_bytes', 'src_packets', 'dst_packets']].mean().reset_index()
            initial_embeddings = embeddings_df[['src_bytes', 'dst_bytes', 'src_packets', 'dst_packets']].values
        else:
            vocab = df['User'].tolist()
            initial_embeddings = df[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']].values

        # Convert embeddings to PyTorch tensors
        node_features = torch.FloatTensor(initial_embeddings)

        # Build First-Order Graph with improved similarity threshold
        G1 = nx.Graph()
        for user in vocab:
            G1.add_node(user)
        similarity_matrix = cosine_similarity(initial_embeddings)
        threshold = 0.85  # Increased threshold for better accuracy
        edges = []
        for i in range(len(vocab)):
            for j in range(i+1, len(vocab)):
                if similarity_matrix[i, j] > threshold:
                    G1.add_edge(vocab[i], vocab[j])
                    edges.append([i, j])
                    edges.append([j, i])

        edge_index = torch.LongTensor(edges).t()

        # Enhanced GAT layer with more heads and features
        gat_layer = GATConv(initial_embeddings.shape[1], 8, heads=8)
        
        # Apply GAT with improved attention
        graph_embeddings = gat_layer(node_features, edge_index)
        graph_embeddings = F.elu(graph_embeddings)
        graph_embeddings = F.dropout(graph_embeddings, p=0.2, training=True)
        graph_embeddings = graph_embeddings.mean(dim=1)
        graph_embeddings = graph_embeddings.detach().numpy()

        # Build Second-Order Graph with improved connectivity
        G2 = G1.copy()
        for node in G1.nodes():
            neighbors = list(G1.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if similarity_matrix[vocab.index(neighbors[i]), vocab.index(neighbors[j])] > 0.75:
                        G2.add_edge(neighbors[i], neighbors[j])

        # Calculate Enhanced Graph Features
        degree_centrality_g1 = nx.degree_centrality(G1)
        clustering_coeff_g1 = nx.clustering(G1)
        betweenness_centrality_g1 = nx.betweenness_centrality(G1)
        degree_centrality_g2 = nx.degree_centrality(G2)
        clustering_coeff_g2 = nx.clustering(G2)
        betweenness_centrality_g2 = nx.betweenness_centrality(G2)

        results_df = pd.DataFrame({
            'Node': vocab,
            'Degree_Centrality_G1': [degree_centrality_g1[node] for node in vocab],
            'Clustering_Coeff_G1': [clustering_coeff_g1[node] for node in vocab],
            'Betweenness_Centrality_G1': [betweenness_centrality_g1[node] for node in vocab],
            'Degree_Centrality_G2': [degree_centrality_g2[node] for node in vocab],
            'Clustering_Coeff_G2': [clustering_coeff_g2[node] for node in vocab],
            'Betweenness_Centrality_G2': [betweenness_centrality_g2[node] for node in vocab]
        })

        return {
            'first_order_graph': G1,
            'second_order_graph': G2,
            'graph_embeddings': graph_embeddings,
            'results_df': results_df,
            'similarity_matrix': similarity_matrix,
            'vocab': vocab
        }

    with st.spinner("Building graphs and generating embeddings..."):
        graph_results = generate_graph_embeddings(df)

    # Visualize First Order and Second Order Graphs
    st.subheader("Graph Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("First Order Graph")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        pos1 = nx.spring_layout(graph_results['first_order_graph'])
        nx.draw(graph_results['first_order_graph'], pos1, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, ax=ax1)
        st.pyplot(fig1)
        
    with col2:
        st.write("Second Order Graph")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        pos2 = nx.spring_layout(graph_results['second_order_graph'])
        nx.draw(graph_results['second_order_graph'], pos2, with_labels=True, node_color='lightgreen',
                node_size=500, font_size=8, ax=ax2)
        st.pyplot(fig2)

    # --- Step 3: Enhanced Rule-Based Anomaly Detection ---
    def detect_anomaly(row, ip_connections):
        flags = []

        # Enhanced General Rules with more precise thresholds
        if row['duration'] > 2.0 and row['service'] not in ['ftp', 'ssh']:
            flags.append("Excessive Duration")
        if (row['src_bytes'] > 6000 or row['dst_bytes'] > 6000) and row['service'] not in ['ftp', 'http']:
            flags.append("High Byte Count")
        if row['src_packets'] > 30 and row['dst_packets'] < 2:
            flags.append("Unidirectional Traffic (Src to Dst)")
        if row['dst_packets'] > 30 and row['src_packets'] < 2:
            flags.append("Unidirectional Traffic (Dst to Src)")
        if row['dst_port'] > 1024 and row['service'] == '-' and row['src_bytes'] > 500:
            flags.append("Suspicious Port")

        # Enhanced Service-Specific Rules
        if row['service'] == 'dns':
            if row['duration'] > 0.005:
                flags.append("DNS Long Duration")
            if row['src_bytes'] > 150 or row['dst_bytes'] > 150:
                flags.append("DNS Large Bytes")
            if row['dst_port'] != 53:
                flags.append("DNS Wrong Port")

        if row['service'] == 'http':
            if row['duration'] < 0.005 and row['src_bytes'] > 600:
                flags.append("HTTP Fast Large Transfer")
            if row['dst_port'] not in [80, 443, 8080]:
                flags.append("HTTP Wrong Port")
            if row['src_packets'] > 60 and row['duration'] < 0.5:
                flags.append("HTTP Potential DoS")

        if row['service'] in ['ftp', 'ftp-data']:
            if row['duration'] < 0.05 and row['src_bytes'] > 3000:
                flags.append("FTP Fast Large Transfer")
            if row['dst_port'] not in [20, 21]:
                flags.append("FTP Wrong Port")

        if row['service'] == '-':
            if row['src_bytes'] > 600 or row['dst_bytes'] > 600:
                flags.append("No Service Large Bytes")
            if row['dst_port'] > 49152 and row['src_bytes'] > 400:
                flags.append("Potential Exploit (Ephemeral Port)")

        if row['dst_bytes'] > 30000 and row['duration'] > 0.5:
            flags.append("Potential DoS")
        if row['duration'] < 0.0005 and row['src_bytes'] > 300:
            flags.append("Potential Exploit")

        # Enhanced Reconnaissance Detection
        src_ip = row['src_ip']
        if src_ip in ip_connections:
            connections = ip_connections[src_ip]
            unique_dsts = len(set(conn['dst_ip'] for conn in connections))
            time_window = (max(pd.to_datetime(conn['timestamp']) for conn in connections) -
                          min(pd.to_datetime(conn['timestamp']) for conn in connections)).total_seconds()
            if unique_dsts > 4 and time_window < 5:
                flags.append("Reconnaissance (Scanning)")
            
            dst_ports = [conn.get('dst_port', 0) for conn in connections if 'dst_port' in conn]
            if len(dst_ports) > 2:
                port_diff = [dst_ports[i+1] - dst_ports[i] for i in range(len(dst_ports)-1)]
                if any(abs(diff) == 1 for diff in port_diff):
                    flags.append("Sequential Port Scanning")

        return flags if flags else "Normal"

    ip_connections = defaultdict(list)
    for idx, row in df.iterrows():
        ip_connections[row['src_ip']].append({
            'dst_ip': row['dst_ip'],
            'dst_port': row['dst_port'],
            'timestamp': row['timestamp']
        })

    with st.spinner("Detecting anomalies with enhanced rules..."):
        df['anomaly_flags'] = df.apply(lambda row: detect_anomaly(row, ip_connections), axis=1)
        df['predicted_anomaly'] = df['anomaly_flags'].apply(lambda x: 0 if x == "Normal" else 1)

    # Rule-Based Results
    normal_df = df[df['anomaly_flags'] == "Normal"]
    anomalous_df = df[df['anomaly_flags'] != "Normal"]
    st.subheader("Anomalies")
    
    if not anomalous_df.empty:
        display_df = anomalous_df[['src_ip', 'dst_ip', 'dst_port', 'duration', 'src_bytes',
                                 'dst_bytes', 'src_packets', 'dst_packets', 'service', 'anomaly_flags']]
        st.dataframe(display_df)
        
        # Create separate table for color-coded alerts
        st.subheader("Detected Anomalies by Risk Level")
        
        # Group anomalies by risk level
        high_risk = []
        moderate_risk = []
        low_risk = []
        
        for flags in anomalous_df['anomaly_flags']:
            if isinstance(flags, list):
                for flag in flags:
                    risk_level = ANOMALY_RISK_MAPPING.get(flag)
                    if risk_level == 'HIGH':
                        high_risk.append(flag)
                    elif risk_level == 'MODERATE':
                        moderate_risk.append(flag)
                    elif risk_level == 'LOW':
                        low_risk.append(flag)
        
        # Create DataFrame for risk levels
        risk_data = {
            'Risk Level': ['HIGH', 'MODERATE', 'LOW'],
            'Detected Anomalies': [
                ', '.join(set(high_risk)) if high_risk else 'None',
                ', '.join(set(moderate_risk)) if moderate_risk else 'None',
                ', '.join(set(low_risk)) if low_risk else 'None'
            ],
            'Count': [len(high_risk), len(moderate_risk), len(low_risk)]
        }
        
        risk_df = pd.DataFrame(risk_data)
        
        # Apply color coding
        def color_risk_level(row):
            color = RISK_LEVELS[row['Risk Level']]
            return [f'background-color: {color}; opacity: 0.3'] * len(row)
        
        styled_risk_df = risk_df.style.apply(color_risk_level, axis=1)
        st.dataframe(styled_risk_df)
        
    else:
        st.write("No rule-based anomalies detected.")

    # Rule-Based Visualization with Risk Levels
    anomaly_counts = defaultdict(int)
    for flags in anomalous_df['anomaly_flags']:
        if isinstance(flags, list):
            for flag in flags:
                anomaly_counts[flag] += 1
    
    if anomaly_counts:
        anomaly_counts_df = pd.DataFrame(list(anomaly_counts.items()), columns=['Anomaly Type', 'Count'])
        # Add risk level column
        anomaly_counts_df['Risk Level'] = anomaly_counts_df['Anomaly Type'].map(ANOMALY_RISK_MAPPING)
        # Create color map based on risk levels
        color_map = {'HIGH': '#B71C1C', 'MODERATE': '#4A148C', 'LOW': '#1A237E'}
        anomaly_counts_df['Color'] = anomaly_counts_df['Risk Level'].map(color_map)
        
        fig1 = px.bar(anomaly_counts_df, x='Count', y='Anomaly Type', orientation='h',
                      title="Frequency of Rule-Based Anomaly Types by Risk Level",
                      color='Risk Level',
                      color_discrete_map=color_map)
        st.plotly_chart(fig1, use_container_width=True)
    
    # Performance Metrics
    st.subheader("Model Performance Metrics")
    
    # Create two columns for metrics and heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate actual performance metrics
        y_true = df['attack_type'].astype(int)  # Convert to int
        y_pred = df['predicted_anomaly'].astype(int)  # Convert to int
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })
        st.table(metrics_df.set_index('Metric', drop=True))
    
    with col2:
        # Create heatmap of anomalies by source and destination IPs
        # First create pivot table and fill NaN with 0
        pivot_table = pd.crosstab(df['src_ip'], df['dst_ip'], values=df['predicted_anomaly'], aggfunc='sum').fillna(0)
        
        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="Destination IP", y="Source IP", color="Anomaly Count"),
            title="Anomaly Distribution Heatmap",
            color_continuous_scale="Viridis"
        )
        
        fig_heatmap.update_layout(
            width=400,
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Step 4: Attack Demonstration ---
    st.subheader("Demonstration: Multiple Attack Types")
    st.markdown("""
    Below are examples of different types of attacks that might appear in network traffic:
    1. HTTP DoS Attack - High volume of packets in short time
    2. DNS Tunneling - Abnormal DNS traffic patterns
    3. FTP Data Exfiltration - Large data transfers over FTP
    """)

    # Simulate different attack scenarios
    attack_demos = pd.DataFrame({
        'src_ip': ['192.168.1.100', '192.168.1.101', '192.168.1.102'],
        'dst_ip': ['10.0.0.50', '10.0.0.51', '10.0.0.52'],
        'dst_port': [80, 53, 21],
        'duration': [0.5, 0.02, 0.05],
        'src_bytes': [5000, 400, 6000],
        'dst_bytes': [200, 350, 100],
        'src_packets': [150, 10, 20],
        'dst_packets': [10, 8, 5],
        'service': ['http', 'dns', 'ftp'],
        'timestamp': [pd.Timestamp('2025-03-27 10:00:00')] * 3
    })

    attack_flags = []
    for _, row in attack_demos.iterrows():
        flags = detect_anomaly(row, {row['src_ip']: [{'dst_ip': row['dst_ip'], 'dst_port': row['dst_port'], 'timestamp': row['timestamp']}]})
        attack_flags.append(flags)
    attack_demos['anomaly_flags'] = attack_flags

    st.write("Simulated Attack Scenarios:")
    # Color code the attack demos based on risk level
    display_attack_demos = attack_demos[['src_ip', 'dst_ip', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
                                       'src_packets', 'dst_packets', 'service', 'anomaly_flags']]
    styled_attack_demos = display_attack_demos.style.apply(lambda x: [color_risk(x['anomaly_flags'])] * len(x), axis=1)
    st.dataframe(styled_attack_demos)

    # --- Step 5: Attack Simulation ---
    st.subheader("Attack Simulation")
    
    attack_type = st.selectbox(
        "Select Attack Type",
        ["DoS Attack", "Port Error", "Suspicious Port"]
    )
    
    if st.button("Simulate Attack"):
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate attack data based on selected type
        num_normal = 490
        num_anomalies = 10
        total_packets = 500
        
        normal_data = []
        attack_data = []
        
        # Generate normal packets
        for i in range(num_normal):
            normal_packet = {
                'src_ip': f'192.168.1.{np.random.randint(1,255)}',
                'dst_ip': f'10.0.0.{np.random.randint(1,255)}',
                'dst_port': 80,
                'duration': np.random.uniform(0.001, 0.01),
                'src_bytes': np.random.randint(100, 500),
                'dst_bytes': np.random.randint(50, 200),
                'src_packets': np.random.randint(2, 8),
                'dst_packets': np.random.randint(1, 5),
                'service': 'http',
                'timestamp': pd.Timestamp('2025-03-27 10:00:00') + pd.Timedelta(seconds=i/10),
                'protocol': 'TCP',
                'connection_state': 'ESTABLISHED',
                'src_ttl': 64,
                'dst_ttl': 64,
                'attack_type': 0
            }
            normal_data.append(normal_packet)
            
            progress = (i + 1) / total_packets
            progress_bar.progress(progress)
            status_text.text(f"Generating normal traffic: {i+1}/{num_normal}")
            
        # Generate attack packets based on type
        for i in range(num_anomalies):
            if attack_type == "DoS Attack":
                attack_packet = {
                    'src_ip': f'192.168.1.{np.random.randint(1,255)}',
                    'dst_ip': f'10.0.0.{np.random.randint(1,255)}',
                    'dst_port': 80,
                    'duration': 0.001,
                    'src_bytes': np.random.randint(8000, 10000),
                    'dst_bytes': np.random.randint(100, 200),
                    'src_packets': np.random.randint(150, 200),
                    'dst_packets': np.random.randint(1, 5),
                    'service': 'http',
                    'attack_type': 1
                }
            elif attack_type == "Port Error":
                attack_packet = {
                    'src_ip': f'192.168.1.{np.random.randint(1,255)}',
                    'dst_ip': f'10.0.0.{np.random.randint(1,255)}',
                    'dst_port': np.random.randint(1025, 2000),
                    'duration': np.random.uniform(0.001, 0.01),
                    'src_bytes': np.random.randint(100, 500),
                    'dst_bytes': np.random.randint(50, 200),
                    'src_packets': np.random.randint(2, 8),
                    'dst_packets': np.random.randint(1, 5),
                    'service': 'http',
                    'attack_type': 1
                }
            else:  # Suspicious Port
                attack_packet = {
                    'src_ip': f'192.168.1.{np.random.randint(1,255)}',
                    'dst_ip': f'10.0.0.{np.random.randint(1,255)}',
                    'dst_port': np.random.randint(50000, 65535),
                    'duration': np.random.uniform(0.001, 0.01),
                    'src_bytes': np.random.randint(100, 500),
                    'dst_bytes': np.random.randint(50, 200),
                    'src_packets': np.random.randint(2, 8),
                    'dst_packets': np.random.randint(1, 5),
                    'service': '-',
                    'attack_type': 1
                }
            
            attack_packet.update({
                'timestamp': pd.Timestamp('2025-03-27 10:00:00') + pd.Timedelta(seconds=(num_normal+i)/10),
                'protocol': 'TCP',
                'connection_state': 'ESTABLISHED',
                'src_ttl': 64,
                'dst_ttl': 64
            })
            
            attack_data.append(attack_packet)
            
            progress = (num_normal + i + 1) / total_packets
            progress_bar.progress(progress)
            status_text.text(f"Generating attack traffic: {i+1}/{num_anomalies}")
            
        # Combine and shuffle data
        all_data = normal_data + attack_data
        np.random.shuffle(all_data)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        filename = f"{attack_type.lower().replace(' ', '_')}_traffic.csv"
        df.to_csv(filename, index=False)
        
        # Final status
        status_text.text(f"Attack simulation completed! Data saved to {filename}")
        
        # Display sample of generated data
        st.write("Sample of Generated Traffic Data:")
        st.dataframe(df.head())
        
        # Download button
        with open(filename, 'rb') as f:
            st.download_button(
                label=f"Download {attack_type} Traffic Data",
                data=f,
                file_name=filename,
                mime='text/csv'
            )

else:
    st.info("Please upload a CSV file or click 'Use Default Dataset' to proceed.")
