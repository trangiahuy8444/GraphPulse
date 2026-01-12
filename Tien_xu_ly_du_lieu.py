from analyzer.network_parser import NetworkParser

parser = NetworkParser()

# Cấu hình đường dẫn (nếu cần)
parser.file_path = "./data/all_network/"
parser.timeseries_file_path = "./data/all_network/TimeSeries/"
# Xử lý dataset dgd (ví dụ)
network_name = "networkdgd.txt"

# Bước 1: Tạo graph features và statistics
parser.create_graph_features(network_name)

# Bước 2: Tạo temporal graph snapshots với TDA features
parser.create_time_series_graphs(network_name)

# Bước 3: Tạo RNN sequences (cho RNN models)
# parser.create_time_series_rnn_sequence(network_name)