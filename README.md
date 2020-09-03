# Holomorphic Embedding Load-Flow Method

The Holomorphic Embedding Load-Flow Method (HELM) is a novel technique for solving the power flow problem of an electric power system. This has been developed as part of the final year project for the electrical engineering degree and it has been partially implemented in [GridCal](https://github.com/SanPen/GridCal).

-----------------
### Installation

There are a couple of options to install the software:
1. (Recommended) Install GridCal by following the instructions detailed [here](https://gridcal.readthedocs.io/en/latest/getting_started/install.html).
2. Use the code provided in this repository. There are two ways to do that:

   2.1. Clone the [HELM repository from GitHub][1]:
   
   *Use this option if you are familiar with Git*
   
    - From the command line:
        - `git clone https://github.com/JosepFanals/HELM`
    - Or from the [HELM GitHub repository page][1]:
        - Click the green **Clone or download** button, then **Open in Desktop**.

   2.2. Download the repository as a .zip file from the GitHub page.
    - Go to the [HELM GitHub repository page][1].
    - Click the green **Clone or download** button, then **Download ZIP**.
    
----------------
### Running HELM

If you have installed GridCal you may want to take a look at the [Running GridCal document](https://gridcal.readthedocs.io/en/latest/getting_started/install.html).

Otherwise, you need to run one of the two files:

   ```Codi/MIH_original.py```: program for the original formulation of the HELM, also called canonical embedding.

   ```Codi/MIH_propi.py```: program for the alternative approach to the embedding of equations. The [thesis](https://github.com/JosepFanals/HELM/blob/master/main.pdf) details its advantages and disadvantages in comparison to the canonical embedding.
   
   
-----------------------
### Learning about HELM

There are multiple appropriate sources to learn how HELM works and what makes it special:
* **[GridCal's documentation](https://gridcal.readthedocs.io/en/latest/theory/power_flow/holomorphic_embedding.html)**: a clear and quick explanation about how HELM works.
* **[The Holomorphic Embedding Load-Flow Method: Foundations and Implementations](https://www.nowpublishers.com/article/Details/EES-015)**, by Antonio Trias: the best and only book about HELM. 
* **[My undergraduate thesis](https://github.com/JosepFanals/HELM/blob/master/main.pdf)**: only convenient if you have a good command of Catalan.


------------------
### Tools included

* **Padé approximants:** analytical resource employed to obtain the final value of the power series.
* **Recursive methods:** same usage as Padé approximants, only that they compute the solution in a recursive fashion. Specifically, the methods are: Aitken's delta-squared, Shanks transformation, Wynn's rho, Wynn's epsilon, Brezinski's theta and Bauer's eta.
* **Thévenin approximants:** they are able to construct the PV and PQ curves, i.e. both the stable and unstable solutions.
* **Sigma approximants:** diagnostic tool that validates the solution of the system as a whole.
* **Padé-Weierstrass:** subprogram that improves the results, useful when the mismatches become unacceptable. 

------------
### License

This works is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---------------------
### Acknowledgements

All this was possible thanks to the help I received from [Santiago Peñate Vera](https://github.com/SanPen).

[1]: https://github.com/JosepFanals/HELM
