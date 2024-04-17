from tkinter.ttk import LabelFrame
from typing import Type
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
import pickle
import ast
from rdkit.Chem.PandasTools import LoadSDF


# versione che usa il dataframe
def create_vector_label(data):
    protein_list = data["Protein"].values
    p_list = list(dict.fromkeys(protein_list))
    p_list.sort()
    print(p_list)
    # vector_label_list = []
    for _, item in data.iterrows():
        if item["Class"] == 0:
            item["Class"] = np.zeros(20)
        else:
            temp = np.zeros(20)
            temp[p_list.index(item["Protein"])] = item["Class"]
            item["Class"] = temp
            # vector_label_list.append(list(temp))
    return data

# carica il dataset e restituisce una lista contenente la lista di grafi e la lista di labels
def classification_process(file_path, protein_family="all", crop_index=-1, salvo=False, ember=False, return_smiles=False):
    
    data = pd.read_csv(file_path)

    #! creazione vector label
    if protein_family == "all":
        smiles_list = data["Smiles"].values

        if salvo:
            with open('/home/psortino/classificatore/Train_13Ottobre22/pickle/vectorlabel/VectorlabelGraphVAE.pickle', 'rb') as f:
                labels = pickle.load(f)
        if ember:
            labels = np.empty((len(smiles_list), 20))
            for i in range(len(smiles_list)):
                labels[i] = np.array([data[f"label_{j}"][i] for j in range(20)])
            new_smiles_list = []
            for element in smiles_list:
                t = ast.literal_eval(element)
                t1 = t[0]
                new_smiles_list.append(t1)
            smiles_list = new_smiles_list
        else:    
            labels = data["Class"].values
    else:
        new_data = data[data["Protein"] == protein_family]
        smiles_list = new_data["Smiles"].values
        labels = list(new_data["Class"].values)

    data_obj_list = []

    for index, (s, l) in tqdm(enumerate(zip(smiles_list, labels[:])), total=len(smiles_list)):
        if index == crop_index:
            return [data_obj_list, labels]
        try:
            mol_obj = Chem.MolFromSmiles(s)
        except TypeError:
            print("Eccezione catturata e molecola saltata")
            labels.remove(l)
            continue
        node_feats = _get_node_features(mol_obj)
        edge_index = _get_adjacency_info(mol_obj)
        edge_feats = _get_edge_features(mol_obj)

        d = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
        data_obj_list.append(d)

    return [data_obj_list, labels, smiles_list]

def process(file_path):
    data = pd.read_csv(file_path)
    smiles_list = data["Smiles"].values

    data_obj_list = []

    for s in tqdm(smiles_list):
        try:
            mol_obj = Chem.MolFromSmiles(s)
        except:
            continue
        node_feats = _get_node_features(mol_obj)
        edge_index = _get_adjacency_info(mol_obj)
        edge_feats = _get_edge_features(mol_obj)

        d = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
        data_obj_list.append(d)

    return data_obj_list

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

def create_dataset_with_sdf(file_path, protein_name, sdf_path="/home/scontino/python/graph_vae/Dataset_sdf_coord/df_Smiles_SDF_preparati.sdf"):
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
    protein_index = protein_list.index(protein_name)

    db = [[x[2:-2] for x in data["Smiles"].values], data[f"label_{protein_index}"].values]

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

    print("dataset: ", len(data_obj_list), len(smiles_list), len(labels))
    print("skipped for errors: ", skipped_for_error)
    print("skipped for missing coordinate: ", zero_z_counter)
    print("skipped for not match: ", skipped_for_not_match)
    return [data_obj_list, smiles_list, labels]


def _get_node_features(mol):
    all_node_feats = []
    for atom in mol.GetAtoms():
        node_feats = []
        # Feature 1: Atomic number        
        # used values: [1, 3, 5, 6, 7, 8, 9, 11, 
        # 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 
        # 34, 35, 38, 47, 52, 53, 55, 56, 83]
        node_feats.append(atom.GetAtomicNum())

        # Feature 2: Atom degree
        # [0, 1, 2, 3, 4, 6]
        node_feats.append(atom.GetDegree())

        # Feature 3: Formal charge
        # [-1, 0, 1, 2, 3, 4]
        node_feats.append(atom.GetFormalCharge())

        # Feature 4: Hybridization
        # [1, 2, 3, 4, 5, 6]
        node_feats.append(atom.GetHybridization())

        # Feature 5: Aromaticity
        # [0, 1]
        node_feats.append(atom.GetIsAromatic())

        # Feature 6: Total Num Hs
        # [0, 1, 2, 3, 4]
        node_feats.append(atom.GetTotalNumHs())

        # Feature 7: Radical Electrons
        # [0, 1, 2, 7]
        node_feats.append(atom.GetNumRadicalElectrons())

        # Feature 8: In Ring
        # [0, 1]
        node_feats.append(atom.IsInRing())

        # Feature 9: Chirality
        # [0, 1, 2]
        node_feats.append(atom.GetChiralTag())

        # Append node features to matrix
        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)    

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


#/ definisco il grafo di networkX 
class MolToNetworkX():
    def __init__(self, smile):

        if isinstance(smile, str):
            self.smile = smile
            self.mol = Chem.MolFromSmiles(self.smile)
            self.graph = nx.Graph()
            Chem.AllChem.Compute2DCoords(self.mol)
            self.conformer = self.mol.GetConformer()
        elif isinstance(smile, list):
            self.smile_list = self.smile
            print("Stai creando un batch di grafi")
        else:
            raise ValueError(f"Inserire una Canonical Smiles come str invece di una {type(smile)}")
        
        
    def add_node(self):
        for id_atom, atom in enumerate(self.mol.GetAtoms()):
            pos = self.conformer.GetAtomPosition(id_atom)
            self.graph.add_node(atom_idx=atom.GetIdx(),
                        atomic_num=atom.GetAtomicNum(),
                        is_aromatic=atom.GetIsAromatic(),
                        atom_symbol=atom.GetSymbol(),
                        formal_charge =atom.GetFormalCharge(),
                        atom_map = atom.GetAtomMapNum(),
                        hybridation=atom.GetHybridization(),
                        proprierties=atom.GetPropNames(),
                        xcor=pos.x,
                        ycor=pos.y
                        )
    
    def add_edge(self):
        for bond in self.mol.GetBonds():
            self.graph.add_edge(u_of_edge = bond.GetBeginAtomIdx(),
                    v_of_edge = bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType(),
                    begin_atom = bond.GetBeginAtom().GetIdx(),
                    end_atom = bond.GetEndAtom().GetIdx()
                    )
    
    def __returnGraph__(self):
        return self.graph
    
    def __returnBatchOfGraph__(self):
        batch_of_graph = []
        for molecule in tqdm(self.smile_list, total=len(self.smile_list)):
            self.graph = nx.Graph()
            self.mol = Chem.MolFromSmiles(molecule)
            #/ aggiungo i nodi per ogni molecola 
            self.add_node()
            self.add_edge()
            graph = self.__returnGraph__()
            batch_of_graph.append(graph)
        return batch_of_graph

#/ definisco il grafo di networkX 
def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(node_for_adding=atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol(),
                   formal_charge =atom.GetFormalCharge(),
                   atom_map = atom.GetAtomMapNum(),
                   hybridation=atom.GetHybridization(),
                   proprierties=atom.GetPropNames()
                   )
        
    for bond in mol.GetBonds():
        G.add_edge(u_of_edge = bond.GetBeginAtomIdx(),
                   v_of_edge = bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   begin_atom = bond.GetBeginAtom().GetIdx(),
                   end_atom = bond.GetEndAtom().GetIdx()
                   )
        
    return G

