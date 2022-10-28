from polygnn import featurize as feat
from polygnn import contrast as cst
import polygnn_trainer as pt
import torch.nn.functional as F
import pandas as pd
import random, torch
import numpy as np

# fix random seeds
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

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
# Load raw data
data = pd.read_csv("sample_data/sample.csv")["smiles_string"].unique().tolist()
print(f"There are {len(data)} unique SMILES strings.")
# Turn raw data into graphs
all_data = [
    feat.get_minimum_graph_tensor(x, bond_config, atom_config, "monocycle")
    for x in data
]

# Split graph data into training and validation sets.
train_pts = []  # training data points
val_pts = []  # validation data points
for point in all_data:
    if random.uniform(0, 1) <= 0.8:
        train_pts.append(point)
    else:
        val_pts.append(point)
del all_data  # save space

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
cfg.epochs = 10
cfg.break_on_bad_grads = False
# Train.
add_noise = lambda x: cst.noise.add_noise(atom_config, x)
cst.train.train(model, train_pts, val_pts, cfg, [add_noise])
