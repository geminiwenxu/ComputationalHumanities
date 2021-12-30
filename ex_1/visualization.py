import igraph
from igraph import Graph, EdgeSeq
import numpy as np
import csv

if __name__ == '__main__':
    with open('dict_BLEU.csv') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
        print(mydict)
    edge_label_list = []
    v_label = ['zh']
    for key, value in mydict.items():
        print(float(value))
        value = round(float(value), 2)
        edge_label_list.append(value)
        v_label.append(key)
    print(edge_label_list)

    nr_vertices = 6
    # v_label = list(map(str, range(nr_vertices)))
    v_label = v_label
    G = Graph.Tree(nr_vertices, 5)
    lay = G.layout('rt')

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    labels = v_label
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=5),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=100,
                                         color='#6175c1',  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             text=labels,
                             hoverinfo='text',
                             opacity=0.8
                             ))


    def make_annotations(pos, text, font_size=20, font_color='rgb(250,250,250)'):
        L = len(pos)
        if len(text) != L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=labels[k],  # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=2 * M - position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations


    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=True,
                showgrid=True,
                showticklabels=False,
                )

    fig.update_layout(title='Tree with Reingold-Tilford Layout',
                      annotations=make_annotations(position, v_label),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )

    edge_label_list = edge_label_list  # average value of round-trip translation b;ue value
    layt = np.asarray(lay)
    elabel_pos = []
    for e in E:
        elabel_pos.append((layt[e[0]] + layt[e[1]]) / 2)

    elabel_pos = np.asarray(elabel_pos)
    fig.add_scatter(x=elabel_pos[:, 0], y=2 * M - elabel_pos[:, 1], mode='text', text=edge_label_list)
    fig.show()
