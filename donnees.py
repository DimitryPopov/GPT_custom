from pathlib import Path

import torch

CONTEXTE = 256

texte = Path("input.txt").read_text()
CARACTERES = sorted(set(texte))

j=0
INDICES={}
for i in CARACTERES :
    INDICES[i]=j
    j+=1

def tokenizer(chaine):
    return torch.tensor([INDICES[z] for z in chaine])

test = "First Citizen: Before we proceed any further, hear me speak. All: Speak, speak. First Citizen: You are all resolved rather to die than to famish?"
#print(tokenizer(test))

def detokenizer(indices):
    return "".join([CARACTERES[h] for h in indices])
#print(detokenizer(tokenizer(test)))

donnees=tokenizer(texte)
coupure = int(0.9*len(donnees))
apprentissage = donnees[:coupure]
validation = donnees[coupure:]

def tirer_lot(source, taille_lot):
    debuts = torch.randint(len(source) - CONTEXTE - 1, (taille_lot,)) 
    fenetres = []
    for debut in debuts:
        fenetre = source[debut:debut + CONTEXTE]   # tranche de 256 entiers
        fenetres.append(fenetre)
    entrees = torch.stack(fenetres)
    cibles = torch.stack([source[debut + 1:debut + CONTEXTE + 1] for debut in debuts])
    return entrees, cibles
#print(tirer_lot(apprentissage, 3)) #3 batch de 256 carac dans tout apprantissage

