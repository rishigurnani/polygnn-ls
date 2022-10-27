from polygnn import featurize as feat
from polygnn import contrast as cst
import polygnn_trainer as pt
import torch.nn.functional as F
import pandas as pd

# Make pyg.Data objects from sample_data
bond_config = feat.BondConfig(True, False, True)
atom_config = feat.AtomConfig(
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
)
data = pd.read_csv("sample_data/sample.csv")["smiles_string"].unique().tolist()
print(f"There are {len(data)} unique SMILES strings.")
train_pts = [
    feat.get_minimum_graph_tensor(x, bond_config, atom_config, "monocycle")
    for x in data
]
# Initialize the hyperparameters.
hps = pt.hyperparameters.HpConfig()
embedding_dim = pt.hyperparameters.ModelParameter(int)
setattr(hps, "embedding_dim", embedding_dim)
hps.set_values(
    {
        "capacity": 2,
        "batch_size": 10,
        "r_learn": 0.003,
        "dropout_pct": 0.0,
        "activation": F.leaky_relu,
        "embedding_dim": 256,
    }
)

# Initialize the model.
model = cst.models.preTrainContrastivePolyGNN(
    node_size=atom_config.n_features,
    edge_size=bond_config.n_features,
    selector_dim=0,
    hps=hps,
    normalize_embedding=True,
)
# Initialize a trainConfig object.
cfg = pt.train.trainConfig(
    cst.loss.contrast_loss(),
    amp=False,
    hps=hps,
    model_save_path=None,  # change if you want the model to save.
)
cfg.epochs = 100
# Train.
add_noise = lambda x: cst.noise.add_noise(atom_config, x)
cst.train.train(model, train_pts, cfg, [add_noise])
