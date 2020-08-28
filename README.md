# Mètode d'incrustació holomòrfica

Aquest Treball Fi de Grau (TFG) tracta sobre el mètode d'incrustació holomòrfica (comercialment conegut com a HELM). 

### Eines que s'inclouen:

* **Aproximants de Padé:** utilitzats per a accelerar la convergència de les sèries.
* **Aproximants de Thévenin:** permeten traçar les corbes PV i PQ, tant amb les solucions de la branca estable com de la inestable.
* **Aproximants Sigma:** recurs de diagnòstic per a validar la solució. Generen el gràfic Sigma.
* **Padé-Weierstrass:** a partir dels resultats inicials de la formulació inicial genera una solució amb menys error.
* **Mètodes recurrents:** Delta d'Aitken, transformacions de Shanks, Rho de Wynn, Èpsilon de Wynn, Theta de Brezinski i Eta de Bauer. Per a computar la solució final de forma recurrent.

### Arxius que incorpora:

```
MIH_original.py: formulació original. Altrament anomenat canonical embedding. Porta a dins el P-W.
MIH_propi.py: formulació pròpia. Presenta variacions a la incrustació.
Funcions.py: Padé, Thévenin, Sigma i mètodes recurrents.
```
