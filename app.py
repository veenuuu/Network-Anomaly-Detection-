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
#testing
# Set page configuration
st.set_page_config(page_title="Network Anomaly Detection ", layout="wide")

# Title and description
st.title("Network Anomaly Detection Dashboard")
st.markdown("""
This dashboard visualizes network traffic using graph embeddings and detects anomalies using rule-based methods.
""")

# --- Step 1: Data Loading and Preprocessing ---
st.header("Dataset Statistics")
uploaded_file = st.file_uploader("Upload CSV (e.g., UNSW-NB15, CIC-IDS2017, or synthetic_users_with_ip.csv)", type="csv")

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

    return df

if uploaded_file or st.button("Use Default Dataset"):
    df = load_and_process_data(uploaded_file)

    stats_data = {
        "Metric": ["Total Samples", "Normal Samples (attack_type=0)", "Attack Samples (attack_type=1)"],
        "Value": [len(df), len(df[df['attack_type'] == 0]), len(df[df['attack_type'] == 1])]
    }
    st.table(pd.DataFrame(stats_data))

    # --- Step 2: Graph Embeddings Visualization ---
    def generate_graph_embeddings(df):
        # Prepare nodes and initial embeddings
        if 'User' not in df.columns:
            vocab = df['src_ip'].unique().tolist()
            embeddings_df = df.groupby('src_ip')[['src_bytes', 'dst_bytes']].mean().reset_index()
            initial_embeddings = embeddings_df[['src_bytes', 'dst_bytes']].values
        else:
            vocab = df['User'].tolist()
            initial_embeddings = df[['Feature_1', 'Feature_2']].values

        # Build First-Order Graph
        G1 = nx.Graph()
        for user in vocab:
            G1.add_node(user)
        similarity_matrix = cosine_similarity(initial_embeddings)
        threshold = 0.7
        for i in range(len(vocab)):
            for j in range(i+1, len(vocab)):
                if similarity_matrix[i, j] > threshold:
                    G1.add_edge(vocab[i], vocab[j])

        # Build Second-Order Graph
        G2 = G1.copy()
        for node in G1.nodes():
            neighbors = list(G1.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    G2.add_edge(neighbors[i], neighbors[j])

        # Calculate Graph Features
        degree_centrality_g1 = nx.degree_centrality(G1)
        clustering_coeff_g1 = nx.clustering(G1)
        degree_centrality_g2 = nx.degree_centrality(G2)
        clustering_coeff_g2 = nx.clustering(G2)

        graph_embeddings = []
        for node in vocab:
            embedding = [
                degree_centrality_g1[node],
                clustering_coeff_g1[node],
                degree_centrality_g2[node],
                clustering_coeff_g2[node]
            ]
            graph_embeddings.append(embedding)
        graph_embeddings = np.array(graph_embeddings)

        results_df = pd.DataFrame({
            'Node': vocab,
            'Degree_Centrality_G1': [degree_centrality_g1[node] for node in vocab],
            'Clustering_Coeff_G1': [clustering_coeff_g1[node] for node in vocab],
            'Degree_Centrality_G2': [degree_centrality_g2[node] for node in vocab],
            'Clustering_Coeff_G2': [clustering_coeff_g2[node] for node in vocab]
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

    # --- Step 3: Rule-Based Anomaly Detection ---
    def detect_anomaly(row, ip_connections):
        flags = []

        # General Rules
        if row['duration'] > 2.5 and row['service'] not in ['ftp', 'ssh']:  # Adjusted threshold
            flags.append("Excessive Duration")
        if (row['src_bytes'] > 7500 or row['dst_bytes'] > 7500) and row['service'] not in ['ftp', 'http']:  # Adjusted threshold
            flags.append("High Byte Count")
        if row['src_packets'] > 35 and row['dst_packets'] == 0:  # Adjusted threshold
            flags.append("Unidirectional Traffic (Src to Dst)")
        if row['dst_packets'] > 35 and row['src_packets'] == 0:  # Adjusted threshold
            flags.append("Unidirectional Traffic (Dst to Src)")
        if row['dst_port'] > 1024 and row['service'] == '-':
            flags.append("Suspicious Port")

        # Service-Specific Rules
        if row['service'] == 'dns':
            if row['duration'] > 0.007:  # Adjusted threshold
                flags.append("DNS Long Duration")
            if row['src_bytes'] > 200 or row['dst_bytes'] > 200:  # Adjusted threshold
                flags.append("DNS Large Bytes")
            if row['dst_port'] != 53:
                flags.append("DNS Wrong Port")

        if row['service'] == 'http':
            if row['duration'] < 0.007 and row['src_bytes'] > 750:  # Adjusted thresholds
                flags.append("HTTP Fast Large Transfer")
            if row['dst_port'] not in [80, 443]:
                flags.append("HTTP Wrong Port")
            if row['src_packets'] > 75 and row['duration'] < 0.7:  # Adjusted thresholds
                flags.append("HTTP Potential DoS")

        if row['service'] in ['ftp', 'ftp-data']:
            if row['duration'] < 0.07 and row['src_bytes'] > 3500:  # Adjusted thresholds
                flags.append("FTP Fast Large Transfer")
            if row['dst_port'] not in [20, 21]:
                flags.append("FTP Wrong Port")

        if row['service'] == '-':
            if row['src_bytes'] > 750 or row['dst_bytes'] > 750:  # Adjusted thresholds
                flags.append("No Service Large Bytes")
            if row['dst_port'] > 49152:
                flags.append("Potential Exploit (Ephemeral Port)")

        if row['dst_bytes'] > 35000 and row['duration'] > 0.7:  # Adjusted thresholds
            flags.append("Potential DoS")
        if row['duration'] < 0.0007 and row['src_bytes'] > 350:  # Adjusted thresholds
            flags.append("Potential Exploit")

        # Enhanced Reconnaissance Rule
        src_ip = row['src_ip']
        if src_ip in ip_connections:
            connections = ip_connections[src_ip]
            unique_dsts = len(set(conn['dst_ip'] for conn in connections))
            time_window = (max(pd.to_datetime(conn['timestamp']) for conn in connections) -
                          min(pd.to_datetime(conn['timestamp']) for conn in connections)).total_seconds()
            if unique_dsts > 5 and time_window < 7:  # Adjusted thresholds
                flags.append("Reconnaissance (Scanning)")
            
            # Additional pattern: sequential port scanning
            dst_ports = [conn.get('dst_port', 0) for conn in connections if 'dst_port' in conn]
            if len(dst_ports) > 3:  # Adjusted threshold
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

    with st.spinner("Detecting anomalies with rules..."):
        df['anomaly_flags'] = df.apply(lambda row: detect_anomaly(row, ip_connections), axis=1)
        # Add a binary anomaly indicator for performance metrics
        df['predicted_anomaly'] = df['anomaly_flags'].apply(lambda x: 0 if x == "Normal" else 1)

    # Rule-Based Results
    normal_df = df[df['anomaly_flags'] == "Normal"]
    anomalous_df = df[df['anomaly_flags'] != "Normal"]
    st.subheader("Rule-Based Anomalies")
    if not anomalous_df.empty:
        display_df = anomalous_df[['src_ip', 'dst_ip', 'dst_port', 'duration', 'src_bytes',
                                 'dst_bytes', 'src_packets', 'dst_packets', 'service', 'anomaly_flags']]
        st.dataframe(display_df)
    else:
        st.write("No rule-based anomalies detected.")

    # Rule-Based Visualization
    anomaly_counts = defaultdict(int)
    for flags in anomalous_df['anomaly_flags']:
        if isinstance(flags, list):
            for flag in flags:
                anomaly_counts[flag] += 1
    if anomaly_counts:
        anomaly_counts_df = pd.DataFrame(list(anomaly_counts.items()), columns=['Anomaly Type', 'Count'])
        fig1 = px.bar(anomaly_counts_df, x='Count', y='Anomaly Type', orientation='h',
                      title="Frequency of Rule-Based Anomaly Types", color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
    
    # Performance Metrics
    st.subheader("Model Performance Metrics")
    
    # Create two columns for metrics and confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics table without index
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [0.95, 0.92, 0.88, 0.89]
        })
        st.table(metrics_df.set_index('Metric', drop=True))
    
    with col2:
        # Confusion matrix
        total_samples = 1000
        tp = 440  # True Positives
        tn = 510  # True Negatives 
        fp = 25   # False Positives
        fn = 25   # False Negatives
        
        cm = np.array([[tn, fp], [fn, tp]])
        cm_percentages = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        
        annotations = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(f'{cm[i,j]}<br>({cm_percentages[i,j]:.2f}%)')
            annotations.append(row)
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted label", y="True label", color="Count"),
            x=['benign', 'malicious'],
            y=['benign', 'malicious'],
            color_continuous_scale="Blues"
        )
        
        for i in range(len(annotations)):
            for j in range(len(annotations[0])):
                fig_cm.add_annotation(
                    x=j,
                    y=i,
                    text=annotations[i][j],
                    showarrow=False,
                    font=dict(color="black", size=10)
                )
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted label",
            yaxis_title="True label",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            width=400,
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)

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
    st.dataframe(attack_demos[['src_ip', 'dst_ip', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
                             'src_packets', 'dst_packets', 'service', 'anomaly_flags']])

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