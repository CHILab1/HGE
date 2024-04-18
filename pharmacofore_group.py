import collections
from collections import Counter
from pathlib import Path
import operator
import time
import math
import torch
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, cluster
from tqdm import tqdm
import torch.nn.functional as F

from rdkit import RDConfig, Chem, Geometry, DistanceGeometry
from rdkit.Chem import (
    ChemicalFeatures,
    rdDistGeom,
    Draw,
    rdMolTransforms,
    AllChem,
)
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Numerics import rdAlignment
import nglview as nv

import os
import glob
import torch
import numpy as np
from cairosvg import svg2png
from skimage.io import imread
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from rdkit.Chem import rdDepictor
import matplotlib.patches as mpatches
from torch_geometric.data import Data
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import MinMaxScaler
from graphArchitecture import SingleClassGCNModel_shapLike, SingleClassGCNModel_shapLike_2


class molGraphGenerator():
    def __init__(self, path_sdf:str=None, smiles:str=None):
        from torch_geometric.data import Data
        if path_sdf is not None:
            self.sdf_path = path_sdf
        if smiles is not None:
            self.smiles = smiles

    def graph_from_sdf(self):
        supply = Chem.SDMolSupplier(self.sdf_path) # name the path of the SDF file 
        for mol in supply:
            node_feats = self._get_node_features_from_sdf(mol)
            edge_weights = self._get_edge_features(mol)
            edge_index = self._get_adjacency_info(mol)
            d = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_weights)
        return d, mol

    def _get_node_features_from_sdf(self, sdf):
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

    def _get_edge_features(self, mol):
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

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices


class MolecularGraphExplainableGradient():
    def __init__(self, molecule, model, n_atoms, anti_vanishing:int=1000):
        self.molecule = molecule
        self.model = model
        self.n_atoms = n_atoms
        self.anti_vanishing = anti_vanishing
        self.feature_colors = {
            "aromatic": (0.6, 1, 0.2),  # Green
            "Acceptors or Donors": (1, 1, 0.4),  # 
            "hydrophobics": (0, 0.5019, 1),  # azzurro
        }
        # donors_patch = mpatches.Patch(color=self.feature_colors["donors"], label='Donors')
        acceptors_or_donors_patch = mpatches.Patch(color=self.feature_colors["Acceptors or Donors"], label='Acceptors or Donors')
        hydrophobic_patch = mpatches.Patch(color=self.feature_colors["hydrophobics"], label='Hydrophobics')
        aromatic_patch = mpatches.Patch(color=self.feature_colors["aromatic"], label='Aromatic')
        other_patch = mpatches.Patch(color=(0.902, 0.345, 0.345) , label='Relevant')
        self.global_patch = [acceptors_or_donors_patch, hydrophobic_patch, aromatic_patch, other_patch]
    
    #! aggiunta 24/10/2023
    def mirror_molecule(self, mol, x_reflect:bool=True, y_reflect:bool=True):
        # Rifletti le coordinate x
        conf = mol.GetConformer()
        for i in range(conf.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            if x_reflect:
                pos.x = -pos.x
            if y_reflect:
                pos.y = -pos.y
            conf.SetAtomPosition(i, pos)
        return mol

        #! fine aggiunta 
    
    #! fine aggiunte 24/10/2023
    def img_for_mol(self, atom_weights:list=[], bond_highlights:list=[], h_custom:dict={}, x_reflect:bool=True, y_reflect:bool=True, rotation:int=60):
        highlight_kwargs = {}
        threshold = 0.70
        rdDepictor.Compute2DCoords(self.molecule)
        #! aggiunta 24/10/2023
        self.molecule = self.mirror_molecule(self.molecule, x_reflect=x_reflect, y_reflect=y_reflect)
        #! fine 24/10/2023
        
        if len(atom_weights) > 0:
            norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
            cmap = cm.get_cmap('bwr')
            plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
            if len(h_custom.keys()) == 0:
                atom_colors = {
                    i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
                }
            grad_atoms = [idx for idx, i in enumerate(atom_weights) if i > 0]
            
            highlight_kwargs = {
                'highlightAtoms': grad_atoms,
                # 'highlightAtoms': h_custom,
                'highlightAtomColors': h_custom,
                'highlightBonds': bond_highlights,
                "highlightBondColors": bond_highlights
            }

        for i, atom in enumerate(self.molecule.GetAtoms()):
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))

        
        drawer = rdMolDraw2D.MolDraw2DSVG(880, 880)

        mol = rdMolDraw2D.PrepareMolForDrawing(self.molecule)
        dopts = drawer.drawOptions()
        #? drawer parameter 
        dopts.annotationFontScale = 0
        dopts.additionalLabelPadding = 0.7
        dopts.scaleHighlightBondWidth = True
        dopts.rotate = rotation
        drawer.DrawMolecule(mol, **highlight_kwargs)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg.replace('svg:', '')
        svg2png(bytestring=svg, write_to='tmp.png', dpi=1000)
        img = imread('tmp.png')
        os.remove('tmp.png')
        return img
    
    def relevantAtoms(self, grad_w):
        hit_ats = []
        for idx, g in enumerate(grad_w): #! sistemare qui con il grad_w che c'interessa 
            if g > 0:
                hit_ats.append(idx)
        return hit_ats 
    
    def edgeIdentification(self, grad_w):
        hit_ats = self.relevantAtoms(grad_w=grad_w)
        hit_bonds = {}
        for item in hit_ats:
            for atom in self.molecule.GetAtoms():
                if atom.GetIdx() == item:
                    start_atom = atom.GetIdx()
                    for neigh in atom.GetNeighbors():
                        end_atom = neigh.GetIdx()
                        hit_bonds[self.molecule.GetBondBetweenAtoms(start_atom, end_atom).GetIdx()] = (0.902, 0.345, 0.345)            
        return hit_bonds
    
    def grad_cam(self, final_conv_acts, final_conv_grads):
        # print('grad_cam')
        node_heat_map = []
        alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
        for n in range(final_conv_acts.shape[0]): # nth node
            # node_heat = F.relu(alphas @ final_conv_acts[n]).item()
            node_heat = (alphas @ final_conv_acts[n]).item()
            node_heat_map.append(node_heat)
        return node_heat_map
    
    def grad_calculation(self):
        self.grad_w_list = []
        #/grad calculation
        conv_gradients = self.model.get_activations_gradient()
        for idx, grad_val in enumerate(conv_gradients[::-1]):
            final_conv_acts = grad_val.view(self.n_atoms, -1)
            # final_conv_grads = self.model.final_conv_grads[0].view(self.n_atoms, -1)
            feature_map = [self.model.conv_grad_1, self.model.conv_grad_2, self.model.conv_grad_3]
            final_conv_grads = feature_map[idx].view(self.n_atoms, -1) * self.anti_vanishing
            print("final_conv_acts: ", final_conv_acts.shape)
            print("final_conv_grads: ", final_conv_grads.shape)

            grad_cam_weights = self.grad_cam(final_conv_acts, final_conv_grads)[:self.molecule.GetNumAtoms()]
            scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )

            # prova paolo 04/09 per combattere il vanishing del gradiente
            self.grad_w = np.where(scaled_grad_cam_weights > 0.7, scaled_grad_cam_weights, 0)
            self.grad_w_list.append(self.grad_w)
            
        return self.grad_w_list

    def getPharmacophoreFeature(self):
        feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
        acceptors, donors, hydrophobics, aromatic = [], [], [], []

        acceptors.append(feature_factory.GetFeaturesForMol(self.molecule, includeOnly="Acceptor"))
        donors.append(feature_factory.GetFeaturesForMol(self.molecule, includeOnly="Donor"))
        hydrophobics.append(feature_factory.GetFeaturesForMol(self.molecule, includeOnly="Hydrophobe"))
        aromatic.append(feature_factory.GetFeaturesForMol(self.molecule, includeOnly="Aromatic"))
        features = {
            "donors": donors,
            "acceptors": acceptors,
            "hydrophobics": hydrophobics,
            "aromatic": aromatic
        }
        return features        
                            
    def assignFeatureType(self, idx_grad=None, global_grad=None):
        grad_w_list = self.grad_calculation()
        if idx_grad is not None:
            grad_w_list = grad_w_list[idx_grad]
        else:
            grad_w_list = global_grad
        features_type = ["donors", "acceptors", "hydrophobics", "aromatic"]
        features = self.getPharmacophoreFeature()
        self.atom_features_idx = {"Acceptors or Donors": [], "hydrophobics":[], "aromatic":[]}
        for ft_type in features_type:
            for feature in features[ft_type]:
                for ft in feature:
                    for idx, atom in enumerate(self.molecule.GetAtoms()):
                        if atom.GetIdx() == ft.GetAtomIds()[0] and grad_w_list[idx] > 0:
                            if ft_type == "donors" or ft_type == "acceptors":
                                ft_type = "Acceptors or Donors"
                            atom.SetProp("atomNote", ft_type)
                            self.atom_features_idx[ft_type].append(atom.GetIdx())

    def highlightsFeature(self, hit_bonds):
        highlights_custom = {}
        rings_idx = mol.GetRingInfo().AtomRings()
        for ft_type in self.atom_features_idx.keys():
            if ft_type == "aromatic":    
                for atom_idx in self.atom_features_idx[ft_type]:
                    print(atom_idx)
                    for rings in rings_idx:
                        if atom_idx in rings:
                            print(rings)
                            for atom_ring_idx in rings:
                                highlights_custom[atom_ring_idx] = self.feature_colors[ft_type]
                                for neigh in self.molecule.GetAtomWithIdx(atom_ring_idx).GetNeighbors():
                                    end_atom = neigh.GetIdx()
                                    hit_bonds[self.molecule.GetBondBetweenAtoms(atom_ring_idx, end_atom).GetIdx()] = self.feature_colors[ft_type]
            else:
                for atom_idx in self.atom_features_idx[ft_type]:
                    highlights_custom[atom_idx] = self.feature_colors[ft_type]
                    for neigh in self.molecule.GetAtomWithIdx(atom_idx).GetNeighbors():
                        end_atom = neigh.GetIdx()
                        hit_bonds[self.molecule.GetBondBetweenAtoms(atom_idx, end_atom).GetIdx()] = self.feature_colors[ft_type]
                    
        return highlights_custom, hit_bonds

    def plotMolecule(self, figsize:tuple=(30, 15), saveFig:bool=False, file_name:str="plot.png", hook_used:int=1, x_reflect:bool=True, y_reflect:bool=True, rotation:int=60, fontsize:str='x-large', legend_pos:str='best'):        
        grad_w_list = self.grad_calculation()
        if hook_used != 1:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            for idx, grad_w in enumerate(grad_w_list):
                self.assignFeatureType(idx_grad=idx)
                hit_bonds = self.edgeIdentification(grad_w=grad_w)
                h_custom, hit_bonds = self.highlightsFeature(hit_bonds=hit_bonds)
                print(hit_bonds)
                axes[idx].imshow(self.img_for_mol(atom_weights=grad_w, bond_highlights=hit_bonds, h_custom=h_custom, x_reflect=x_reflect, y_reflect=y_reflect, rotation=rotation))
                axes[idx].legend(handles=self.global_patch, loc=legend_pos, facecolor="white", labelcolor='black', fontsize=fontsize)
            if saveFig:
                fig.savefig(file_name, bbox_inches="tight")
                print("Immagine salvata con successo.")
        else:
            self.assignFeatureType(idx_grad=None)
            hit_bonds = self.edgeIdentification()
            h_custom, hit_bonds = self.highlightsFeature(hit_bonds=hit_bonds)
            fig, axes = plt.subplots(figsize=figsize)
            axes.imshow(self.img_for_mol(atom_weights=self.grad_w, bond_highlights=hit_bonds, h_custom=h_custom, x_reflect=x_reflect, y_reflect=y_reflect, rotation=rotation))
            axes.legend(handles=self.global_patch, loc=legend_pos, facecolor="white", labelcolor='black', fontsize=fontsize)
            if saveFig:
                fig.savefig(file_name, bbox_inches="tight")
                print("Immagine salvata con successo. ")
                
    def plotCombinedMolecule(self, figsize:tuple=(30, 15), saveFig:bool=False, file_name:str="plot.png", hook_used:int=1, x_reflect:bool=True, y_reflect:bool=True, rotation:int=60, fontsize:str='x-large', legend_pos:str='best'):
        grad_w_list = self.grad_calculation()
        if hook_used != 1:
            fig, axes = plt.subplots(figsize=figsize)
            grad_w_sum = torch.zeros(size=grad_w_list[0].shape)
            for idx, grad_w in enumerate(grad_w_list):
                self.assignFeatureType(idx_grad=idx)
                grad_w_sum = grad_w_sum + grad_w
            hit_bonds = self.edgeIdentification(grad_w=grad_w_sum)
            self.assignFeatureType(idx_grad=None, global_grad=grad_w_sum)
            h_custom, hit_bonds = self.highlightsFeature(hit_bonds=hit_bonds)
            axes.imshow(self.img_for_mol(atom_weights=grad_w_sum, bond_highlights=hit_bonds, h_custom=h_custom, x_reflect=x_reflect, y_reflect=y_reflect, rotation=rotation))
            axes.legend(handles=self.global_patch, loc=legend_pos, facecolor="white", labelcolor='black', fontsize=fontsize)
            if saveFig:
                fig.savefig(file_name, bbox_inches="tight")
                print("Immagine salvata con successo.")
        else:
            self.assignFeatureType(idx_grad=None)
            hit_bonds = self.edgeIdentification()
            h_custom, hit_bonds = self.highlightsFeature(hit_bonds=hit_bonds)
            fig, axes = plt.subplots(figsize=figsize)
            axes.imshow(self.img_for_mol(atom_weights=self.grad_w, bond_highlights=hit_bonds, h_custom=h_custom, x_reflect=x_reflect, y_reflect=y_reflect, rotation=rotation))
            axes.legend(handles=self.global_patch, loc=legend_pos, facecolor="white", labelcolor='black', fontsize=fontsize)
            if saveFig:
                fig.savefig(file_name, bbox_inches="tight")
                print("Immagine salvata con successo. ")

if __name__ == "__main__":
    path_sdf = "sdf_files/"
    sdflist = glob.glob(f'{path_sdf}/*.sdf')
    for sdf_file in tqdm(sdflist):
        gen_graph = molGraphGenerator(path_sdf=sdf_file)
        d, mol = gen_graph.graph_from_sdf()
            
        device = torch.device("cuda:0")
        net2 = SingleClassGCNModel_shapLike_2(feature_node_dim=12, num_classes=1)
        net2 = net2.to(device)

        dictio_torch = torch.load("/models/LossModel_17.tar")

        net2.load_state_dict(dictio_torch["model_state_dict"])
        d = d.to(device)
        net2.eval()
        pred, logits = net2(d.x[:, :12], d.edge_index, None, hook_start=True, hook_mid=True, hook_end=True)
        print(pred)

        pred.backward()
        grad = net2.get_activations_gradient()

        d.x = d.x[:, :12]
        n_atoms = d.x.shape[0]
        print(n_atoms)

        name = sdf_file.split("/")[-1][:-4]
        path_folder = f"explainer_3_hook_2811_v1/{name}"
        print(path_folder, name)
        os.makedirs(path_folder, exist_ok=True)
        #/ plot dell'immagine grad 
        
        grad_img = MolecularGraphExplainableGradient(molecule=mol, model=net2, n_atoms=n_atoms)
        grad_img.plotMolecule(figsize=(30, 20), fontsize="large", saveFig=True, hook_used=3, file_name=f"{path_folder}/{name}.png", rotation=0, x_reflect=False, y_reflect=True)
        grad_img.plotCombinedMolecule(figsize=(30, 20), fontsize="xx-large", saveFig=True, hook_used=3, file_name=f"{path_folder}/{name}_merged.png", rotation=0, x_reflect=False, y_reflect=True)
