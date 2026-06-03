"""
VECTORES DE UMBRALES DE ENERGIA  (CONSTANTES del programa)
===========================================================
Generado automaticamente por entrenar.py el 2026-06-03.

Estos vectores estan escritos de forma EXPLICITA y NUMERICA. El
reconocimiento NO procesa el banco de grabaciones: solo compara el vector
de energia de la senal contra estas constantes con la diferencia total
absoluta  E_XC = suma_i |E_Xi - E_Ci|.

Cada vector E_C tiene 20 componentes (una por subbanda),
normalizado (suma = 1) para ser independiente del volumen.
"""

import numpy as np

LIMITES_SUBBANDAS_HZ = [300, 500, 700, 900, 1100, 1300, 1500, 1800, 2100, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
FS    = 22050
N_FFT = 1024
SALTO = 256

# E_C -> VECTOR DE UMBRALES DE ENERGIA de cada tipo de canto (CONSTANTE)
#        E_C[especie] = [E_C1, E_C2, ..., E_CS]
E_C = {
    # Canario coronado  (60 grabaciones)
    "Sicalis_flaveola": np.array([0.13353498, 0.06732664, 0.04069629, 0.02815888, 0.01996975, 0.01449803, 0.01077410, 0.01061852, 0.01163545, 0.01191423, 0.02407608, 0.04525705, 0.06830925, 0.09491956, 0.09827407, 0.11718979, 0.10289833, 0.06240317, 0.02719823, 0.01034760]),
    # Sinsonte tropical  (60 grabaciones)
    "Mimus_gilvus": np.array([0.07649073, 0.05177174, 0.02436347, 0.01914816, 0.01768724, 0.02154464, 0.09703910, 0.21030341, 0.23761062, 0.12098494, 0.03534422, 0.01834529, 0.01185822, 0.00819829, 0.00968826, 0.00871603, 0.00828603, 0.01036308, 0.00829174, 0.00396480]),
    # Siriri comun  (60 grabaciones)
    "Tyrannus_melancholicus": np.array([0.14834286, 0.07699107, 0.04669857, 0.03813661, 0.03736236, 0.02738662, 0.01810975, 0.01167328, 0.01269476, 0.01449910, 0.01118702, 0.01243768, 0.02115436, 0.06411640, 0.17477538, 0.20146709, 0.06573332, 0.01109841, 0.00431160, 0.00182377]),
    # Bichofue  (60 grabaciones)
    "Pitangus_sulphuratus": np.array([0.08326456, 0.05092157, 0.02838904, 0.02756769, 0.02333663, 0.02616532, 0.03755739, 0.02862308, 0.02546398, 0.06979648, 0.17613157, 0.15956572, 0.10445415, 0.06431218, 0.02631713, 0.01562907, 0.01339486, 0.01210603, 0.01484540, 0.01215815]),
    # Currucutu  (60 grabaciones)
    "Megascops_choliba": np.array([0.05397433, 0.22867284, 0.36577406, 0.04123985, 0.02368042, 0.02903300, 0.03722149, 0.02413042, 0.02915374, 0.03057810, 0.01879292, 0.03189817, 0.01908348, 0.01109814, 0.01491794, 0.01073886, 0.01196410, 0.00902648, 0.00740542, 0.00161624]),
}

# SIGMA_C -> desviacion estandar por subbanda (variabilidad de la energia).
SIGMA_C = {
    "Sicalis_flaveola": np.array([0.16091438, 0.05609868, 0.03699603, 0.04133945, 0.02241353, 0.01659057, 0.01298614, 0.02355267, 0.01490973, 0.01513237, 0.03788773, 0.05739167, 0.06441452, 0.08519572, 0.08014010, 0.09273428, 0.09140284, 0.07784552, 0.04428503, 0.13278375]),
    "Mimus_gilvus": np.array([0.19023578, 0.10268397, 0.02361563, 0.01649365, 0.01488198, 0.01707769, 0.11441217, 0.16190168, 0.18543513, 0.17048451, 0.11484813, 0.05641341, 0.04097260, 0.02178732, 0.03359730, 0.03253854, 0.02131360, 0.09783665, 0.05147467, 0.09685474]),
    "Tyrannus_melancholicus": np.array([0.22057861, 0.06569518, 0.06138472, 0.04692683, 0.03750200, 0.02909322, 0.01396769, 0.00851548, 0.02260237, 0.04679189, 0.02850583, 0.04037787, 0.06365133, 0.13616474, 0.16492519, 0.18995963, 0.12248124, 0.02669128, 0.01528281, 0.00990559]),
    "Pitangus_sulphuratus": np.array([0.18229291, 0.07468847, 0.04582135, 0.05237451, 0.02606214, 0.02284290, 0.03426224, 0.02045458, 0.02940463, 0.06451330, 0.13770157, 0.13266329, 0.09434944, 0.08945461, 0.03769808, 0.03608507, 0.03578423, 0.03014751, 0.02899286, 0.05099542]),
    "Megascops_choliba": np.array([0.14092840, 0.17311990, 0.25971513, 0.06817190, 0.04992810, 0.05354441, 0.11010910, 0.06021562, 0.07025926, 0.08856349, 0.08085549, 0.10114996, 0.05601498, 0.09113810, 0.07470958, 0.05762243, 0.05567522, 0.05343610, 0.06203842, 0.02552593]),
}

NOMBRES_ESPECIES = {
    "Sicalis_flaveola": ("Canario coronado", "Sicalis flaveola"),
    "Mimus_gilvus": ("Sinsonte tropical", "Mimus gilvus"),
    "Tyrannus_melancholicus": ("Siriri comun", "Tyrannus melancholicus"),
    "Pitangus_sulphuratus": ("Bichofue", "Pitangus sulphuratus"),
    "Megascops_choliba": ("Currucutu", "Megascops choliba"),
}

