import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

nodes_path = os.path.join(CSV_DIR, "preprocessed_nodes.csv")
edges_path = os.path.join(CSV_DIR, "preprocessed_edges.csv")

df_nodes = pd.read_csv(nodes_path)
df_edges = pd.read_csv(edges_path)

total_nodes = len(df_nodes)
total_edges = len(df_edges)
unique_timesteps = sorted(df_nodes["time_step"].unique())

print(f"Total Active Nodes Pooled   : {total_nodes:,}")
print(f"Total Structural Edges Loaded: {total_edges:,}")
print(f"Available Chronological Windows : {len(unique_timesteps)} snapshots (t={min(unique_timesteps)} to t={max(unique_timesteps)})")

snapshot_graphs = {}
snapshot_logs = []
for ts in sorted(df_nodes["time_step"].unique()):
    nodes_ts = df_nodes[df_nodes["time_step"] == ts]
    ts_node_set = set(nodes_ts["txId"])

    edges_ts = df_edges[
        df_edges["txId1"].isin(ts_node_set)
        & df_edges["txId2"].isin(ts_node_set)
    ]

    G_ts = nx.DiGraph()

    for _, row in nodes_ts.iterrows():
        G_ts.add_node(
            int(row["txId"]),
            label=int(row["label"]),
            mask=bool(row["mask"]),
            degree_feat=float(row["degree"])
        )

    edge_tuples = list(
        zip(
            edges_ts["txId1"].astype(int),
            edges_ts["txId2"].astype(int)
        )
    )

    G_ts.add_edges_from(edge_tuples)

    snapshot_graphs[ts] = G_ts

    snapshot_logs.append({
        "time_step": ts,
        "nodes": G_ts.number_of_nodes(),
        "edges": G_ts.number_of_edges()
    })

df_snapshots_stats = pd.DataFrame(snapshot_logs)

print(
    f"Successfully generated "
    f"{len(snapshot_graphs)} discrete network snapshots."
)

print("\nSnapshot Volumetric Allocation Tail Profile")
print(df_snapshots_stats.head(10).to_string(index=False))

G_t1 = snapshot_graphs[1]

n_nodes = G_t1.number_of_nodes()
n_edges = G_t1.number_of_edges()
graph_density = nx.density(G_t1)

avg_degree = n_edges / n_nodes if n_nodes > 0 else 0

weakly_connected_comps = nx.number_weakly_connected_components(G_t1)
strongly_connected_comps = nx.number_strongly_connected_components(G_t1)

t1_summary_metrics = pd.DataFrame([
    {
        "Metric Flag Name": "Total Graph Vertices (Nodes)",
        "Value": float(n_nodes)
    },
    {
        "Metric Flag Name": "Total Graph Inductions (Edges)",
        "Value": float(n_edges)
    },
    {
        "Metric Flag Name": "Network Density Index",
        "Value": float(graph_density)
    },
    {
        "Metric Flag Name": "Average Out-Degree Metric",
        "Value": float(avg_degree)
    },
    {
        "Metric Flag Name": "Weakly Connected Sub-components",
        "Value": float(weakly_connected_comps)
    },
    {
        "Metric Flag Name": "Strongly Connected Sub-components",
        "Value": float(strongly_connected_comps)
    }
])

t1_summary_metrics.to_csv(
    os.path.join(CSV_DIR, "network_statistics_t1.csv"),
    index=False
)

print("TOPOLOGICAL SUMMARY PROFILE: TIMESTAMP 1")

for _, row in t1_summary_metrics.iterrows():
    print(
        f"  {row['Metric Flag Name']:<38}: {row['Value']:.6f}"
        if row['Value'] < 1
        else f"  {row['Metric Flag Name']:<38}: {int(row['Value']):,}"
    )

    deg_centrality = nx.degree_centrality(G_t1)

df_deg = pd.DataFrame(
    deg_centrality.items(),
    columns=["txId", "Degree Centrality"]
)
df_deg["txId"] = df_deg["txId"].astype(int)

node_labels = nx.get_node_attributes(G_t1, "label")
df_deg["Class Label"] = df_deg["txId"].map(node_labels)
df_deg["Class Name"] = df_deg["Class Label"].map(
    {0: "Licit", 1: "Illicit", -1: "Unknown"}
)

df_deg = df_deg.sort_values(
    by="Degree Centrality",
    ascending=False
).reset_index(drop=True)

df_deg.index += 1
df_deg.index.name = "Rank"

top10_deg = df_deg.head(10).copy().reset_index()

df_deg.reset_index().to_csv(
    os.path.join(CSV_DIR, "degree_centrality_t1.csv"),
    index=False
)

print("Top 10 Ranked Hub Entities by Degree Centrality")
print(
    top10_deg[
        ["Rank", "txId", "Degree Centrality", "Class Name"]
    ].to_string(index=False)
)

top10_deg_nodes = top10_deg["txId"].tolist()

G_top10_deg = G_t1.subgraph(top10_deg_nodes)

node_colors_deg = [
    deg_centrality[node]
    for node in G_top10_deg.nodes()
]

fig, ax = plt.subplots(figsize=(10, 6))

pos_deg = nx.spring_layout(
    G_top10_deg,
    k=0.8,
    seed=24
)

edges_deg = nx.draw_networkx_edges(
    G_top10_deg,
    pos_deg,
    edge_color="#999999",
    width=1.2,
    ax=ax
)

nodes_deg = nx.draw_networkx_nodes(
    G_top10_deg,
    pos_deg,
    node_color=node_colors_deg,
    cmap=plt.cm.cool,
    node_size=650,
    ax=ax
)

labels_deg = {
    node: str(node)
    for node in G_top10_deg.nodes()
}

nx.draw_networkx_labels(
    G_top10_deg,
    pos_deg,
    labels=labels_deg,
    font_size=9,
    font_color="black",
    ax=ax
)

sm_deg = plt.cm.ScalarMappable(
    cmap=plt.cm.cool,
    norm=plt.Normalize(
        vmin=min(node_colors_deg),
        vmax=max(node_colors_deg)
    )
)

sm_deg.set_array([])

cbar_deg = fig.colorbar(
    sm_deg,
    ax=ax,
    shrink=0.8
)

cbar_deg.set_label(
    "Degree Centrality",
    rotation=270,
    labelpad=15,
    fontsize=11
)

plt.title(
    "Top 10 Degree Centrality Nodes (Induced Subgraph Topology)",
    weight="bold",
    pad=15
)

plt.axis("off")
plt.tight_layout()
plt.show()

bet_centrality = nx.betweenness_centrality(G_t1)

df_bet = pd.DataFrame(
    bet_centrality.items(),
    columns=["txId", "Betweenness Centrality"]
)

df_bet["txId"] = df_bet["txId"].astype(int)

df_bet["Class Name"] = (
    df_bet["txId"]
    .map(node_labels)
    .map({0: "Licit", 1: "Illicit", -1: "Unknown"})
)

df_bet = df_bet.sort_values(
    by="Betweenness Centrality",
    ascending=False
).reset_index(drop=True)

df_bet.index += 1
df_bet.index.name = "Rank"

top10_bet = df_bet.head(10).copy().reset_index()

df_bet.reset_index().to_csv(
    os.path.join(CSV_DIR, "betweenness_centrality_t1.csv"),
    index=False
)

print("Top 10 Bridge Entities by Betweenness Centrality")

print(
    top10_bet[
        ["Rank", "txId", "Betweenness Centrality", "Class Name"]
    ].to_string(index=False)
)

top10_nodes_list = top10_bet["txId"].tolist()

G_top10 = G_t1.subgraph(top10_nodes_list)

node_colors = [
    bet_centrality[node]
    for node in G_top10.nodes()
]

fig, ax = plt.subplots(figsize=(10, 6))

pos = nx.spring_layout(
    G_top10,
    k=0.5,
    seed=42
)

edges = nx.draw_networkx_edges(
    G_top10,
    pos,
    edge_color="#999999",
    width=1.5,
    ax=ax
)

nodes = nx.draw_networkx_nodes(
    G_top10,
    pos,
    node_color=node_colors,
    cmap=plt.cm.cool,
    node_size=600,
    ax=ax
)

labels = {
    node: str(node)
    for node in G_top10.nodes()
}

nx.draw_networkx_labels(
    G_top10,
    pos,
    labels=labels,
    font_size=9,
    font_color="black",
    ax=ax
)

sm = plt.cm.ScalarMappable(
    cmap=plt.cm.cool,
    norm=plt.Normalize(
        vmin=min(node_colors),
        vmax=max(node_colors)
    )
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    shrink=0.8
)

cbar.set_label(
    "Betweenness Centrality",
    rotation=270,
    labelpad=15,
    fontsize=11
)

plt.title(
    "Top 10 Betweenness Centrality Nodes (Induced Subgraph Topology)",
    weight="bold",
    pad=15
)

plt.axis("off")
plt.tight_layout()
plt.show()

close_centrality = nx.closeness_centrality(G_t1)

df_close = pd.DataFrame(
    close_centrality.items(),
    columns=["txId", "Closeness Centrality"]
)

df_close["txId"] = df_close["txId"].astype(int)

df_close["Class Name"] = (
    df_close["txId"]
    .map(node_labels)
    .map({0: "Licit", 1: "Illicit", -1: "Unknown"})
)

df_close = df_close.sort_values(
    by="Closeness Centrality",
    ascending=False
).reset_index(drop=True)

df_close.index += 1
df_close.index.name = "Rank"

top10_close = df_close.head(10).copy().reset_index()

df_close.reset_index().to_csv(
    os.path.join(CSV_DIR, "closeness_centrality_t1.csv"),
    index=False
)

print("--- Top 10 Proximity Entities by Closeness Centrality ---")

print(
    top10_close[
        ["Rank", "txId", "Closeness Centrality", "Class Name"]
    ].to_string(index=False)
)

top10_close_nodes = top10_close["txId"].tolist()

G_top10_close = G_t1.subgraph(top10_close_nodes)

node_colors_close = [
    close_centrality[node]
    for node in G_top10_close.nodes()
]

fig, ax = plt.subplots(figsize=(10, 6))

pos_close = nx.spring_layout(
    G_top10_close,
    k=0.4,
    seed=10
)

edges_close = nx.draw_networkx_edges(
    G_top10_close,
    pos_close,
    edge_color="#999999",
    width=1.5,
    ax=ax
)

nodes_close = nx.draw_networkx_nodes(
    G_top10_close,
    pos_close,
    node_color=node_colors_close,
    cmap=plt.cm.cool,
    node_size=650,
    ax=ax
)

labels_close = {
    node: str(node)
    for node in G_top10_close.nodes()
}

nx.draw_networkx_labels(
    G_top10_close,
    pos_close,
    labels=labels_close,
    font_size=9,
    font_color="black",
    ax=ax
)

sm_close = plt.cm.ScalarMappable(
    cmap=plt.cm.cool,
    norm=plt.Normalize(
        vmin=min(node_colors_close),
        vmax=max(node_colors_close)
    )
)

sm_close.set_array([])

cbar_close = fig.colorbar(
    sm_close,
    ax=ax,
    shrink=0.8
)

cbar_close.set_label(
    "Closeness Centrality",
    rotation=270,
    labelpad=15,
    fontsize=11
)

plt.title(
    "Top 10 Closeness Centrality Nodes (Induced Subgraph Topology)",
    weight="bold",
    pad=15
)

plt.axis("off")
plt.tight_layout()
plt.show()

pagerank_centrality = nx.pagerank(
    G_t1,
    alpha=0.85
)

df_pr = pd.DataFrame(
    pagerank_centrality.items(),
    columns=["txId", "PageRank Centrality"]
)

df_pr["txId"] = df_pr["txId"].astype(int)

node_labels = nx.get_node_attributes(
    G_t1,
    "label"
)

df_pr["Class Label"] = df_pr["txId"].map(node_labels)

df_pr["Class Name"] = df_pr["Class Label"].map(
    {0: "Licit", 1: "Illicit", -1: "Unknown"}
)

df_pr = df_pr.sort_values(
    by="PageRank Centrality",
    ascending=False
).reset_index(drop=True)

df_pr.index += 1
df_pr.index.name = "Rank"

top10_pr = df_pr.head(10).copy().reset_index()

df_pr.reset_index().to_csv(
    os.path.join(CSV_DIR, "pagerank_centrality_t1.csv"),
    index=False
)

print("Top 10 Ranked Prestige Entities by PageRank Centrality")

print(
    top10_pr[
        ["Rank", "txId", "PageRank Centrality", "Class Name"]
    ].to_string(index=False)
)

top10_pr_nodes = top10_pr["txId"].tolist()

G_top10_pr = G_t1.subgraph(top10_pr_nodes)

node_colors_pr = [
    pagerank_centrality[node]
    for node in G_top10_pr.nodes()
]

fig, ax = plt.subplots(figsize=(10, 6))

pos_pr = nx.spring_layout(
    G_top10_pr,
    k=0.8,
    seed=24
)

edges_pr = nx.draw_networkx_edges(
    G_top10_pr,
    pos_pr,
    edge_color="#999999",
    width=1.2,
    ax=ax
)

nodes_pr = nx.draw_networkx_nodes(
    G_top10_pr,
    pos_pr,
    node_color=node_colors_pr,
    cmap=plt.cm.cool,
    node_size=650,
    ax=ax
)

labels_pr = {
    node: str(node)
    for node in G_top10_pr.nodes()
}

nx.draw_networkx_labels(
    G_top10_pr,
    pos_pr,
    labels=labels_pr,
    font_size=9,
    font_color="black",
    ax=ax
)

sm_pr = plt.cm.ScalarMappable(
    cmap=plt.cm.cool,
    norm=plt.Normalize(
        vmin=min(node_colors_pr),
        vmax=max(node_colors_pr)
    )
)

sm_pr.set_array([])

cbar_pr = fig.colorbar(
    sm_pr,
    ax=ax,
    shrink=0.8
)

cbar_pr.set_label(
    "PageRank Centrality",
    rotation=270,
    labelpad=15,
    fontsize=11
)

plt.title(
    "Top 10 PageRank Centrality Nodes (Induced Subgraph Topology)",
    weight="bold",
    pad=15
)

plt.axis("off")
plt.tight_layout()
plt.show()