import torch
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.PandasTools import LoadSDF
from sklearn.model_selection import train_test_split




def _get_edge_features(mol):
    all_edge_feats = []
    for bond in mol.GetBonds():
        edge_feats = []
        # Feature 1: Bond type (as double)
        # 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE, 3 for ???
        edge_feats.append(bond.GetBondTypeAsDouble())
        # Feature 2: Rings
        # 0 for not in ring, 1 for in ring
        edge_feats.append(bond.IsInRing())
        # Append node features to matrix (twice, per direction)
        all_edge_feats += [edge_feats, edge_feats]

    all_edge_feats = np.asarray(all_edge_feats)
    return torch.tensor(all_edge_feats, dtype=torch.float)

def _get_adjacency_info(mol):
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    return edge_indices

def _get_node_features_from_sdf(sdf):
    sdf_conformero = sdf.GetConformer()
    all_node_feats = []
    for atom in sdf.GetAtoms():
        node_feats = []
        #/ Feature 1: Atomic number    
        node_feats.append(atom.GetAtomicNum())
        #/ Feature 2: Atom degree
        node_feats.append(atom.GetDegree())
        #/ Feature 3: Formal charge
        node_feats.append(atom.GetFormalCharge())
        #/ Feature 4: Hybridization
        node_feats.append(atom.GetHybridization())
        #/ Feature 5: Aromaticity
        node_feats.append(atom.GetIsAromatic())
        #/ Feature 6: Total Num Hs
        node_feats.append(atom.GetTotalNumHs())
        #/ Feature 7: Radical Electrons
        node_feats.append(atom.GetNumRadicalElectrons())
        #/ Feature 8: In Ring
        node_feats.append(atom.IsInRing())
        #/ Feature 9: Chirality
        node_feats.append(atom.GetChiralTag())
        #/ coords 
        p1 = sdf_conformero.GetAtomPosition(atom.GetIdx())
        node_feats.append(p1.x) #p1.x, p1.y, p1.z
        node_feats.append(p1.y)
        node_feats.append(p1.z)
        #/ idx
        node_feats.append(atom.GetIdx())
        all_node_feats.append(node_feats)

    return torch.tensor(all_node_feats)

def savePickle(path_to_save, data_to_save, file_name=None):
    """
    Method that save pickle in a chosen directory.
    
    # Arguments: \n
        * path_to_save {str} -- Path where to save the pickle file in format "dir/dir/dir/" \n
        * file_name {str} -- Chosen name for the pickle file in format "name.pickle" \n
        * data_to_save {data_variable} -- Data to save in pickle \n
    
    # Returns:
        Save file in a chosen directory
    """
    if file_name is None:
        path = path_to_save
        with open(path, 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return f"Salvataggio Completato"
    else:
        path = path_to_save+file_name
        with open(path, 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return f"Salvataggio Completato"

def create_dataset_with_sdf(file_path, protein_name=None, sdf_path="df_Smiles_SDF_preparati.sdf"):
    '''
        Creazione del dataset che include le informazioni dei file sdf
        Ritorna una lista con 3 elementi:
        - La lista di oggetti Data (grafi molecolari)
        - La lista delle relative smiles
        - La lista delle relative labels
    '''
    data = pd.read_csv(file_path)
    protein_list = ["ACK", "ALK", "CDK1", "CDK2", "CDK6", "INSR","ITK","JKA2","JNK3","MELK",
                "CHK1", "CK2A1", "CLK2", "DYRK1A", "EGFR","ERK2","GSK3B","IRAK4","MAP2k1","PDK1"]
    if protein_name is not None:
        protein_index = protein_list.index(protein_name)
        db = [[x[2:-2] for x in data["Smiles"].values], data[f"label_{protein_index}"].values]
    else:
        protein_index = [f"label_{x}" for x in range(20)]
        db = [[x[2:-2] for x in data["Smiles"].values], data[protein_index].values]

    sdf_path = sdf_path
    sdf_file = Chem.SDMolSupplier(sdf_path)
    
    df = LoadSDF(sdf_path, smilesName='SMILES')
    sdf_smiles_list = df["Smiles"].values

    data_obj_list = []
    smiles_list = []
    labels = []

    zero_z_counter = 0
    zero_z_idx_list = []
    skipped_for_error = 0
    skipped_for_not_match = 0

    for idx, sdf in enumerate(tqdm(sdf_file)):
        #! controllo che la molecola possa aprirsi e la skippo se non funziona
        try: 
            node_feats = _get_node_features_from_sdf(sdf)
        except:
            skipped_for_error += 1
            continue
        
        #! controllo se la coordinata Z è tutta 0, in tal caso scarto la molecola e vado avanti 
        if node_feats[:, -1:].sum() == 0:
            zero_z_counter+=1
            zero_z_idx_list.append(idx)
            continue
        
        #! controllo l'incrocio con l'altro database 
        try:
            smiles = sdf_smiles_list[idx]
            s_idx = db[0].index(smiles)
        except:
            # print("smiles not found!")
            skipped_for_not_match += 1
            continue

        label = db[1][s_idx]
        labels.append(label)

        #? se è tutto apposto la molecola va bene
        edge_weights = _get_edge_features(sdf)
        edge_index = _get_adjacency_info(sdf)
        d = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_weights)
        data_obj_list.append(d)
        smiles_list.append(smiles)

        #! appendo la label
        # empty = 0
        # not_empty = 0

        # new_s = f"['{smiles}']"
        # label = data.loc[data['Smiles'] == new_s, f'label_{protein_index}'].values
        # if len(label) == 0:
        #     empty += 1
        # else:
        #     not_empty += 1

        # print("empty elements: ", empty)
        # print("not empty elements: ", not_empty)
    print("dataset: ", len(data_obj_list), len(smiles_list), len(labels))
    print("skipped for errors: ", skipped_for_error)
    print("skipped for missing coordinate: ", zero_z_counter)
    print("skipped for not match: ", skipped_for_not_match)
    return [data_obj_list, smiles_list, labels]

if __name__ == "__main__":
    file_path = "SMiles&LabelFull.csv"
    dataset = create_dataset_with_sdf(file_path=file_path)
    
    graphs, smiles, labels = dataset
    for i in tqdm(range(len(graphs))):
        graphs[i].y = torch.tensor(labels[i], dtype=torch.float32)
    
    train_set, val_set = train_test_split(graphs, test_size=0.1, random_state=17, shuffle=True)
    savePickle(path_to_save="train_set.pkl", data_to_save=train_set)
    savePickle(path_to_save="val_set.pkl", data_to_save=val_set)
