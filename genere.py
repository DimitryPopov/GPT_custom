import torch

from donnees import CARACTERES, CONTEXTE, detokenizer, tokenizer
from modele import GPT

TEMPERATURE = 0.5
LONGUEUR = 1000
DEBUT = "ANNA:\nHello, my name is Anna"

modele = GPT(len(CARACTERES), CONTEXTE)
modele.load_state_dict(torch.load("modele.pt", map_location="cpu"))
modele.eval()

indices = tokenizer(DEBUT)[None]                         # (1, longueur) : un lot d'une seule fenêtre

with torch.no_grad():
    for _ in range(LONGUEUR):
        fenetre = indices[:, -CONTEXTE:]               # le modèle ne voit que les 256 derniers caractères
        logits = modele(fenetre)[:, -1]                # scores du caractère suivant, après la dernière position
        probabilites = torch.softmax(logits / TEMPERATURE, dim=-1)
        suivant = torch.multinomial(probabilites, 1)   # tirage au sort selon ces probabilités
        indices = torch.cat([indices, suivant], dim=1)

print(detokenizer(indices[0].tolist()))