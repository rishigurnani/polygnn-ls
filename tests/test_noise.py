import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import random
import polygnn_trainer as pt
import polygnn
from polygnn.contrast import noise
from sklearn.model_selection import train_test_split
import os
import shutil
import time
import argparse

if os.path.exists("example_models"):
   shutil.rmtree("example_models")

parser = argparse.ArgumentParser()
parser.add_argument("--polygnn", default=False, action="store_true")
parser.add_argument("--polygnn2", default=False, action="store_true")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()
if (not args.polygnn) and (not args.polygnn2):
    raise ValueError("Neither the polygnn nor the polygnn2 flags are set. Choose one.")
elif args.polygnn and args.polygnn2:
    raise ValueError("Both the polygnn and the polygnn2 flags are set. Choose one.")


# #########
# constants
# #########
# For improved speed, some settings below differ from those used in the
# companion paper. In such cases, the values used in the paper are provided
# as a comment.
RANDOM_SEED = 100
HP_EPOCHS = 20  # companion paper used 200
SUBMODEL_EPOCHS = 100  # companion paper used 1000
N_FOLDS = 3  # companion paper used 5
HP_NCALLS = 10  # companion paper used 25
MAX_BATCH_SIZE = 50  # companion paper used 450
capacity_ls = list(range(2, 6))  # companion paper used list(range(2, 14))
weight_decay = 0
N_PASSES = 2  # companion paper used 10

start = time.time()
# The companion paper trains multi-task (MT) models for six groups. In this
# example file, we will only train an MT model for properties in the "electronic"
# group.
PROPERTY_GROUPS = {
    "electronic": [
        "Egc",
        "Egb",
        "Ea",
        "Ei",
    ],
}

bond_config = polygnn.featurize.BondConfig(True, True, True)
atom_config = polygnn.featurize.AtomConfig(
    True,
    True,
    True,
    True,
    True,
    True,
    combo_hybrid=False,  # if True, SP2/SP3 are combined into one feature
    aromatic=True,
)
#####################

# fix random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data. This data set is a subset of the data used to train the
# electronic-properties MT models shown in the companion paper. The full
# data set can be found at khazana.gatech.edu.
master_data = pd.read_csv("./sample_data/sample.csv")

# Split the data.
train_data, test_data = train_test_split(
    master_data,
    test_size=0.2,
    stratify=master_data.prop,
    random_state=RANDOM_SEED,
)

if args.polygnn:
    featurization_scheme = "monocycle"
elif args.polygnn2:
    featurization_scheme = "trimer"

smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
    x,
    bond_config,
    atom_config,
    featurization_scheme,
)

# Make a directory to save our models in.
os.mkdir("example_models/")

# Train one model per group. We only have one group, "electronic", in this
# example file.
for group in PROPERTY_GROUPS:
    prop_cols = sorted(PROPERTY_GROUPS[group])
    print(
        f"Working on group {group}. The following properties will be modeled: {prop_cols}",
        flush=True,
    )

    selector_dim = len(prop_cols)
    # Define a directory to save the models for this group of properties.
    root_dir = "example_models/" + group

    group_train_data = train_data.loc[train_data.prop.isin(prop_cols), :]
    group_test_data = test_data.loc[test_data.prop.isin(prop_cols), :]
    
    ######################
    # prepare data
    ######################
    group_data = pd.concat([group_train_data, group_test_data], ignore_index=False)
    group_data, scaler_dict = pt.prepare.prepare_train(
        group_data, smiles_featurizer=smiles_featurizer, root_dir=root_dir
    )
    
    ######################
    # Test Adding Noise
    ######################
    result = noise.add_noise(atom_config, group_data['data'].iloc[0]['x'])
    epsilon = 1e-6
    for row in result:
        assert abs(row[0:44].sum() - 1) < epsilon
        assert abs(row[44:55].sum() - 1) < epsilon
        assert abs(row[55:62].sum() - 1) < epsilon
        assert abs(row[64:69].sum() - 1) < epsilon
        assert row[69] <= 1
        assert row[69] >= 0
    print("IT WORKS")
