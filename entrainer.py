import torch
import torch.nn.functional as F

from donnees import CARACTERES, CONTEXTE, apprentissage, tirer_lot, validation
from modele import GPT

PAS = 5000
TAILLE_LOT = 32
TAUX = 3e-4
EVALUER_TOUS_LES = 500
LOTS_EVALUATION = 20

appareil = "mps" if torch.backends.mps.is_available() else "cpu"


def calculer_perte(modele, source):
    entrees, cibles = tirer_lot(source, TAILLE_LOT)
    entrees, cibles = entrees.to(appareil), cibles.to(appareil)
    logits = modele(entrees)                                    # (lot, longueur, 65)
    return F.cross_entropy(logits.transpose(1, 2), cibles)      # cross_entropy veut les classes en 2e position


@torch.no_grad()
def evaluer(modele, source):
    modele.eval()
    perte = sum(calculer_perte(modele, source).item() for _ in range(LOTS_EVALUATION)) / LOTS_EVALUATION
    modele.train()
    return perte


torch.manual_seed(0)
modele = GPT(len(CARACTERES), CONTEXTE).to(appareil)
optimiseur = torch.optim.AdamW(modele.parameters(), lr=TAUX)
meilleure_perte = float("inf")

for pas in range(1, PAS + 1):
    optimiseur.zero_grad()
    calculer_perte(modele, apprentissage).backward()
    optimiseur.step()

    if pas % EVALUER_TOUS_LES == 0:
        perte_apprentissage = evaluer(modele, apprentissage)
        perte_validation = evaluer(modele, validation)
        print(f"pas {pas}/{PAS} : apprentissage {perte_apprentissage:.3f}, validation {perte_validation:.3f}", flush=True)
        if perte_validation < meilleure_perte:
            meilleure_perte = perte_validation
            torch.save(modele.state_dict(), "modele.pt")