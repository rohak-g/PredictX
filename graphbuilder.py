from collections import defaultdict
import json
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import zipfile
import tempfile


def process_json(json_path, png_dir):
    with open(json_path) as f:
        netlist = json.load(f)

    module = list(netlist["modules"].values())[0]
    cells = module["cells"]
    ports = module["ports"]
    netnames = module["netnames"]

    G = nx.DiGraph()
    bit_to_name = {}
    for name, data in netnames.items():
        for bit in data["bits"]:
            bit_to_name[bit] = name

    net_to_source = {}
    net_to_dest = {}

    for pname, pdata in ports.items():
        direction = pdata["direction"]
        for bit in pdata["bits"]:
            G.add_node(pname, type="input" if direction == "input" else "output")
            if direction == "input":
                net_to_source[bit] = pname
            else:
                net_to_dest.setdefault(bit, []).append(pname)

    output_ports = {"ZN", "Q", "Y", "Z", "OUT", "O", "SUM", "COUT"}
    for cname, cdata in cells.items():
        gtype = cdata["type"]
        G.add_node(cname, type=gtype)
        for port, bits in cdata["connections"].items():
            for bit in bits:
                if port.upper() in output_ports:
                    net_to_source[bit] = cname
                else:
                    net_to_dest.setdefault(bit, []).append(cname)

    for bit, src in net_to_source.items():
        if bit in net_to_dest:
            for dst in net_to_dest[bit]:
                if dst != src:
                    G.add_edge(src, dst, signal=f"net {bit_to_name.get(bit, bit)}")

    # Layout and drawing
    layers = {"input": 0, "gate": 1, "output": 2}
    pos = {}
    layer_nodes = {0: [], 1: [], 2: []}
    for node, data in G.nodes(data=True):
        if data["type"] == "input":
            layer = 0
        elif data["type"] == "output":
            layer = 2
        else:
            layer = 1
        layer_nodes[layer].append(node)
    for layer, nodes in layer_nodes.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (layer, -i)
    color_map = []
    for node, data in G.nodes(data=True):
        if data["type"] == "input":
            color_map.append("gold")
        elif data["type"] == "output":
            color_map.append("orchid")
        elif data["type"].startswith("NAND"):
            color_map.append("skyblue")
        elif data["type"].startswith("XNOR"):
            color_map.append("lightgreen")
        elif data["type"].startswith("OAI21"):
            color_map.append("lightcoral")
        else:
            color_map.append("lightgrey")
    gate_counts = defaultdict(int)
    node_labels = {}
    for node, data in G.nodes(data=True):
        ntype = data["type"]
        if ntype not in ["input", "output"]:
            gate_type = ntype.split("_")[0]
            gate_counts[gate_type] += 1
            label = f"{gate_type}{gate_counts[gate_type]}"
            node_labels[node] = label
        else:
            node_labels[node] = node
    # plt.figure(figsize=(10, 7))
    # nx.draw(G, pos, labels=node_labels, node_color=color_map,
    #         node_size=1800, font_size=10, font_weight='bold',
    #         arrows=True, edgecolors="black")
    # edge_labels = {(u, v): d["signal"] for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    # plt.title(f"Hierarchical Gate-Level Graph: {os.path.basename(json_path)}", fontsize=14)
    # plt.tight_layout()
    # # Save PNG
    # png_name = os.path.splitext(os.path.basename(json_path))[0] + ".png"
    # png_path = os.path.join(png_dir, png_name)
    # plt.savefig(png_path)
    # plt.close()


def main(zip_path):
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as pngdir:
        adjdir = os.path.join(tmpdir, "adj_matrices")
        adj_csv_dir = os.path.join(tmpdir, "adj_csv")
        jsondir = os.path.join(tmpdir, "node_features_json")
        csvdir = os.path.join(tmpdir, "node_features_csv")

        # Ensure they exist
        os.makedirs(adjdir, exist_ok=True)
        os.makedirs(adj_csv_dir, exist_ok=True)
        os.makedirs(jsondir, exist_ok=True)
        os.makedirs(csvdir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(tmpdir)
            # Find all JSON files in extracted directory (recursively)
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.json'):
                        json_path = os.path.join(root, file)
                        process_json(json_path, pngdir)
        # # Zip all PNGs
        # png_zip_path = os.path.join(os.getcwd(), 'graph_pngs.zip')
        # with zipfile.ZipFile(png_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        #     for file in os.listdir(pngdir):
        #         if file.endswith('.png'):
        #             zipf.write(os.path.join(pngdir, file), arcname=file)
        # print(f"All PNGs zipped to {png_zip_path}")

        # Zip all adjacency matrices (.npy)
        adj_zip_path = os.path.join(os.getcwd(), 'adj_matrices.zip')
        with zipfile.ZipFile(adj_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(adjdir):
                if file.endswith('.npy'):
                    zipf.write(os.path.join(adjdir, file), arcname=file)
        print(f"All adjacency matrices zipped to {adj_zip_path}")

        # Zip all adjacency matrices (.csv)
        adj_csv_zip_path = os.path.join(os.getcwd(), 'adj_matrices_csv.zip')
        with zipfile.ZipFile(adj_csv_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(adj_csv_dir):
                if file.endswith('.csv'):
                    zipf.write(os.path.join(adj_csv_dir, file), arcname=file)
        print(f"All adjacency matrix CSVs zipped to {adj_csv_zip_path}")

        # Zip all node features JSON
        json_zip_path = os.path.join(os.getcwd(), 'node_features_json.zip')
        with zipfile.ZipFile(json_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(jsondir):
                if file.endswith('.json'):
                    zipf.write(os.path.join(jsondir, file), arcname=file)
        print(f"All node features JSON zipped to {json_zip_path}")

        # Zip all node features CSV
        csv_zip_path = os.path.join(os.getcwd(), 'node_features_csv.zip')
        with zipfile.ZipFile(csv_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(csvdir):
                if file.endswith('.csv'):
                    zipf.write(os.path.join(csvdir, file), arcname=file)
        print(f"All node features CSV zipped to {csv_zip_path}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        print("Usage: python graphbuilder.py json_only.zip")
        sys.exit(1)
    main(zip_path)