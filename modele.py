lazy from math import sqrt
import torch
from torch import nn


def attention(Q, K, V, mask):
    scores = (Q @ K.transpose(-2, -1)) / sqrt(Q.shape[-1])     # (lot, têtes, longueur, longueur)
    scores = scores.masked_fill(mask == 0, float("-inf"))      # cases futures interdites
    poids = torch.softmax(scores, dim=-1)                      # chaque ligne somme à 1
    sortie = poids @ V                                         # (lot, têtes, longueur, colonnes par tête)
    return sortie, poids

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__() #héritage
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        self.number_heads = heads

    def decouper_tetes(self, t):
        
        lot, longueur, dim = t.shape
        colonnes_par_tete = dim // self.number_heads
        t = t.view(lot, longueur, self.number_heads, colonnes_par_tete)   # (lot, longueur, têtes, colonnes)
        
        return t.transpose(1, 2)  

    def recoller_tetes(self, t):
        lot, tetes, longueur, colonnes_par_tete = t.shape
        t = t.transpose(1, 2).contiguous()                                # (lot, longueur, têtes, colonnes)
        return t.view(lot, longueur, tetes * colonnes_par_tete)           # (lot, longueur, dim)

    def forward(self, x, mask):                       # x : (lot, longueur, dim)
        # 1. projections
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. découpage en têtes
        Q = self.decouper_tetes(Q)
        K = self.decouper_tetes(K)
        V = self.decouper_tetes(V)

        # 3. attention, chaque tête séparément
        sortie, self.poids = attention(Q, K, V, mask)

        # 4. recollage des têtes
        sortie = self.recoller_tetes(sortie)

        # 5. projection de sortie
        return self.w_o(sortie)
    
class Bloc(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norme_attention = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, heads)
        self.norme_dense = nn.LayerNorm(dim)
        self.dense = nn.Sequential(
            nn.Linear(dim, 4 * dim),     # élargit chaque vecteur : 128 -> 512
            nn.GELU(),                   # non-linéarité
            nn.Linear(4 * dim, dim),     # revient à la dimension : 512 -> 128
        )

    def forward(self, x, mask):                                    # x : (lot, longueur, dim)
        x = x + self.attention(self.norme_attention(x), mask)      # les caractères échangent de l'information
        x = x + self.dense(self.norme_dense(x))                    # chaque caractère traite la sienne, seul
        return x                                                   # (lot, longueur, dim)
    
class GPT(nn.Module):
    def __init__(self, symboles, contexte, dim=256, heads=4, blocs=4):
        super().__init__()
        self.plongement_caracteres = nn.Embedding(symboles, dim)    # un vecteur par caractère
        self.plongement_positions = nn.Embedding(contexte, dim)     # un vecteur par position 0..contexte-1
        self.blocs = nn.ModuleList([Bloc(dim, heads) for _ in range(blocs)])
        self.norme_finale = nn.LayerNorm(dim)
        self.tete = nn.Linear(dim, symboles)                        # un score par caractère possible

    def forward(self, indices):                                     # indices : (lot, longueur), des entiers
        lot, longueur = indices.shape
        positions = torch.arange(longueur, device=indices.device)  # 0, 1, ..., longueur-1
        x = self.plongement_caracteres(indices) + self.plongement_positions(positions)   # (lot, longueur, dim)

        mask = torch.tril(torch.ones(longueur, longueur, device=indices.device))       # triangle inférieur
        for bloc in self.blocs:
            x = bloc(x, mask)                                       # (lot, longueur, dim)

        return self.tete(self.norme_finale(x))                     # (lot, longueur, symboles)