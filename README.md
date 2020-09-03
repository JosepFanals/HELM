# Holomorphic Embedding Load-Flow Method

The Holomorphic Embedding Load-Flow Method (HELM) is a novel technique for solving the power flow problem of an electric power system. This has been developed as part of the final year project for the electrical engineering degree. 

### Tools:

* **Padé approximants:** analytical resource employed to obtain the final value of the power series.
* **Recursive methods:** same usage as Padé approximants, only that they compute the solution in a recursive fashion. Specifically, the methods are: Aitken's delta-squared, Shanks transformation, Wynn's rho, Wynn's epsilon, Brezinski' theta and Bauer's eta.
* **Thévenin approximants:** they are able to construct the PV and PQ curves, i.e. both the stable and unstable solutions.
* **Sigma approximants:** diagnostic tool that validates the solution of the system as a whole.
* **Padé-Weierstrass:** subprogram that improves the results, useful when the mismatches become unacceptable. 

### Relevant files:

* ***main.pdf***: the thesis as such, with details about the method, development of equations and interesting results
* ***Codi/MIH_original.py***: main program for the original formulation of the HELM, also called canonical embedding.
* ***Codi/MIH_propi.py***: main program for the alternative approach to the embedding of equations. The thesis details its advantages and disadvantages in comparison to the canonical embedding.
* ***Codi/Funcions.py***: contains the tools explained above.

All this was possible thanks to the help I received from [Santiago Peñate Vera](https://github.com/SanPen). You can find a more polished code in [GridCal](https://github.com/SanPen/GridCal).
