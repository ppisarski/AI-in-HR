import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from csv import reader

DATA_URL = 'input/chat_groups.csv'


def load_data():
    data = pd.DataFrame(columns=["First", "Second", "Count"])
    with open(DATA_URL, 'r', encoding='utf-8-sig') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row.sort()
            filtered_row = [emp for emp in row if len(emp) > 0]
            # generate employee pairs
            for i in range(0, len(filtered_row) - 1):
                for j in range(i + 1, len(filtered_row)):
                    first = filtered_row[i]
                    second = filtered_row[j]

                    curr_rec = data[(data['First'] == first) & (data['Second'] == second)]
                    if curr_rec.empty:
                        new_df = pd.DataFrame([{'First': first,
                                                'Second': second,
                                                'Count': 1}])
                        data = data.append(new_df, ignore_index=True)
                    else:
                        curr_rec.at[curr_rec.index[0], 'Count'] = curr_rec.at[curr_rec.index[0], 'Count'] + 1
                        data.update(curr_rec)

    print(data.head())
    return data


def sort_dict(dict):
    sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_dict:
        print(key, " = ", value)


def graph(df):
    graph_emps = nx.Graph()
    for i, row in df.iterrows():
        graph_emps.add_edge(row['First'],
                            row['Second'],
                            weight=row['Count'])

    print("Network Summary: \n", nx.info(graph_emps))

    elarge = [(x1, x2) for (x1, x2, data) in graph_emps.edges(data=True) if data['weight'] > 5]
    emedium = [(x1, x2) for (x1, x2, data) in graph_emps.edges(data=True) if 3 < data['weight'] <= 5]
    esmall = [(x1, x2) for (x1, x2, data) in graph_emps.edges(data=True) if data['weight'] <= 3]

    pos = nx.spring_layout(graph_emps)
    nx.draw_networkx_nodes(graph_emps, pos, node_size=700, node_color='orange')
    nx.draw_networkx_edges(graph_emps, pos, edgelist=elarge, width=6, edge_color='blue')
    nx.draw_networkx_edges(graph_emps, pos, edgelist=emedium, width=4, edge_color='green')
    nx.draw_networkx_edges(graph_emps, pos, edgelist=esmall, width=2, edge_color='gray')
    nx.draw_networkx_labels(graph_emps, pos, font_size=16, font_family='Arial')

    plt.axis('off')
    plt.show()

    print("Nodes Mason is connected with:")
    print(nx.degree(graph_emps, 'Mason'))

    print("Clustering coefficient:")
    sort_dict(nx.clustering(graph_emps, weight='weight'))

    print("Centrality:")
    sort_dict(nx.degree_centrality(graph_emps))

    print("Betweenness:")
    sort_dict(nx.betweenness_centrality(graph_emps))

def main():
    data = load_data()

    graph(data)


if __name__ == "__main__":
    main()
