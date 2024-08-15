import os

import networkx as nx
import numpy as np
import pandas as pd
from django.conf import settings
from node2vec import Node2Vec

nilai_to_bobot = {
    'A': 4.00,
    'A-': 3.75,
    'B+': 3.50,
    'B': 3.00,
    'B-': 2.75,
    'C+': 2.50,
    'C': 2.00,
    'D': 1.00,
    'E': 0.00,
    # tambahkan pemetaan lainnya sesuai kebutuhan
}


def read_data_train():
    return pd.read_excel(os.path.join(settings.BASE_DIR, 'media', '2015.xlsx'))


def pivots(df: pd.DataFrame):
    kolom_kode_matkul = [f'Kode_matkul{i}' for i in range(1, 78)]

    dfs_pivot = []
    for kolom in kolom_kode_matkul:
        df_pivot = df.pivot(index=['NIM', 'STATUS'], columns=kolom, values=f'Grade{kolom[11:]}')
        df_pivot.reset_index(inplace=True)
        dfs_pivot.append(df_pivot)

    df_pivots = dfs_pivot[0]
    for i in range(1, len(dfs_pivot)):
        df_pivots = df_pivots.merge(dfs_pivot[i], on=['NIM', 'STATUS'])

    df_pivots.fillna(0, inplace=True)
    df_status = df_pivots[['NIM', 'STATUS']]
    df_status['NIM'] = df_status['NIM'].astype(str)
    df_mhs = df_pivots.drop('STATUS', axis=1)
    return df_status, df_mhs


def table_melting(df: pd.DataFrame):
    df = pd.melt(df, id_vars=['NIM'], var_name='kode_matkul', value_name='nilai', ignore_index=True)
    df = df[df['nilai'] != 0]
    df = df.reset_index(drop=True)
    df['bobot'] = df['nilai'].map(nilai_to_bobot)
    df['NIM'] = df['NIM'].astype(str)
    return df


def graph_edit_distance(G1, G2, threshold=0.50):
    ged_value = 0
    for u, v, data in G1.edges(data=True):
        if G2.has_edge(u, v):
            weight_diff = abs(data['bobot'] - G2[u][v]['bobot'])
            ged_value += min(weight_diff, threshold)
        else:
            ged_value += data['bobot']
    for u, v, data in G2.edges(data=True):
        if not G1.has_edge(u, v):
            ged_value += data['bobot']
    return ged_value


def create_graph(df_mhs: pd.DataFrame):
    num_walks = 10
    walk_length = 10
    dimensions = 2

    graphs_nim = {}
    nimList = []
    graphs_embedding = {}
    i = 0

    for nim, group in df_mhs.groupby("NIM"):
        G = nx.Graph()

        for _, row in group.iterrows():
            kode_matkul = row["kode_matkul"]
            bobot = row["bobot"]  # Ambil bobot dari kolom bobot dalam DataFrame
            G.add_node('MHS', bipartite=0)
            G.add_node(kode_matkul, bipartite=1)
            G.add_edge('MHS', kode_matkul, bobot=bobot)  # Tambahkan atribut bobot pada tepi
        nimList.append(nim)
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        graph_embedding = model.wv[G.nodes]
        graphs_nim[nim] = G
        graphs_embedding[nim] = graph_embedding
        i = i + 1
        print("Iteration = ", i)
    print("\nTotal iteration : ", i)

    return G, graphs_nim, graphs_embedding, nimList, i


def split_data(i, df: pd.DataFrame, graphs_nim, graphs_embedding, train: float = .8, test: float = .2, ):
    trn, tst = int(i * train), int(i * test)
    train_graphs_keys = [str(df['NIM'][i]) for i in range(0, int(trn + 1))]  # 80 percent from the total input data
    train_graphs1 = {key: graphs_nim[key] for key in train_graphs_keys}
    train_graphs = {key: graphs_embedding[key] for key in train_graphs_keys}

    test_graphs_keys = [str(df['NIM'][i]) for i in range(tst, i)]  # 20 percent from the total input data
    test_graphs1 = {key: graphs_nim[key] for key in test_graphs_keys}
    test_graphs = {key: graphs_embedding[key] for key in test_graphs_keys}
    return train_graphs1, train_graphs, test_graphs1, test_graphs


def split_data_for_dev(i, df: pd.DataFrame, graphs_nim, graphs_embedding, ):
    train_graphs_keys = [str(df['NIM'][i]) for i in range(0, i)]  # 80 percent from the total input data
    train_graphs1 = {key: graphs_nim[key] for key in train_graphs_keys}
    train_graphs = {key: graphs_embedding[key] for key in train_graphs_keys}

    test_graphs_keys = [str(df['NIM'][i]) for i in range(1, 2)]  # 20 percent from the total input data
    test_graphs1 = {key: graphs_nim[key] for key in test_graphs_keys}
    test_graphs = {key: graphs_embedding[key] for key in test_graphs_keys}
    return train_graphs1, train_graphs, test_graphs1, test_graphs


def get_ged_score(train_graphs1, train_graphs, test_graphs1, test_graphs):
    test_graph_names1 = list(test_graphs1.keys())
    train_graph_names1 = list(train_graphs1.keys())

    # Buatlah list untuk menampung hasil perbandingan dan GED
    comparison_data = []

    # Lakukan perulangan untuk semua pasangan graf
    for test_graph_name1 in test_graph_names1:
        for train_graph_name1 in train_graph_names1:
            G1 = test_graphs1[test_graph_name1]
            G2 = train_graphs1[train_graph_name1]

            nodes_students1 = [node for node, data in G1.nodes(data=True) if data['bipartite'] == 0]
            nodes_courses1 = [node for node, data in G1.nodes(data=True) if data['bipartite'] == 1]

            nodes_students2 = [node for node, data in G2.nodes(data=True) if data['bipartite'] == 0]
            nodes_courses2 = [node for node, data in G2.nodes(data=True) if data['bipartite'] == 1]

            adj_matrix_G1 = nx.bipartite.biadjacency_matrix(G1, row_order=nodes_students1, column_order=nodes_courses1)
            adj_matrix_G2 = nx.bipartite.biadjacency_matrix(G2, row_order=nodes_students2, column_order=nodes_courses2)

            adj_array_G1 = np.array(adj_matrix_G1.toarray())
            adj_array_G2 = np.array(adj_matrix_G2.toarray())

            if adj_array_G1.shape[1] > adj_array_G2.shape[1]:
                padding = np.zeros((adj_array_G2.shape[0], adj_array_G1.shape[1] - adj_array_G2.shape[1]))
                adj_array_G2 = np.concatenate((adj_array_G2, padding), axis=1)

            if adj_array_G2.shape[1] > adj_array_G1.shape[1]:
                padding = np.zeros((adj_array_G1.shape[0], adj_array_G2.shape[1] - adj_array_G1.shape[1]))
                adj_array_G1 = np.concatenate((adj_array_G1, padding), axis=1)

            ged_value = graph_edit_distance(G1, G2)

            comparison_data.append({'GRAPH_TEST': test_graph_name1, 'GRAPH_TRAIN': train_graph_name1, 'GED': ged_value})

    comparison_table = pd.DataFrame(comparison_data)
    comparison_table = comparison_table.dropna()
    return comparison_table


def merging_table(comparison_table: pd.DataFrame, df_status: pd.DataFrame):
    df_status['STATUS'] = df_status['STATUS'].map({'LULUS': 1, 'DROP-OUT': 0})
    return comparison_table.merge(df_status, how='left', left_on='GRAPH_TRAIN', right_on='NIM')


def find_max_val(df_merged) -> pd.DataFrame:
    averages = []
    for value in df_merged['GRAPH_TRAIN'].unique():
        average_value = df_merged[df_merged['GRAPH_TRAIN'] == value]['GED'].max()
        averages.append({'GRAPH_TRAIN': value, 'Average_GED': average_value})

    return pd.DataFrame(averages)


def mapping(x):
    if x:
        return "LULUS"
    return "DROP-OUT"
