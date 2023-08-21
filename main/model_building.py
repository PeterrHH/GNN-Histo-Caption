from ml.models import CellGraphModel, TissueGraphModel
'''
Create our own ml package
'''
def build_model(args):
    if 'bracs_cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            lstm_params = None,
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)
    elif 'bracs_tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            lstm_params = None,
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)
    elif 'bracs_hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            lstm_params = None,
            cg_node_dim=NODE_DIM,
            tg_node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')