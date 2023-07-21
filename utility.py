import numpy as np
import pandas as pd

def stiffnesstensor2modulus(stiffness_matrix):
    assert stiffness_matrix.shape == (6, 6)
    
    # Get compliance tensor (reciprocal of stiffness tensor) 
    compliance_matrix = np.linalg.inv(stiffness_matrix.astype(np.float64))

    # bulk modulus, K_V (Voigt average)
    K_V = np.sum(stiffness_matrix[:3,:3])/9
    # bulk modulus, K_R (Reuss average)
    K_R = 1/np.sum(compliance_matrix[:3,:3])
    # bulk modulus, K_H (Hill average)
    K_H = (K_R + K_V)/2

    # shear modulu, G_V (Voigt average)
    G_V = (np.sum(np.diag(stiffness_matrix[:3,:3])) 
            - (stiffness_matrix[0,1] + stiffness_matrix[0,2] + stiffness_matrix[1,2])
            + 3*np.sum(np.diag(stiffness_matrix[3:,3:])))/15
    # shear modulus, G_R (Reuss average)
    G_R = (15/(4*np.sum(np.diag(compliance_matrix[:3,:3]))
              - 4*np.sum(compliance_matrix[0,1] + compliance_matrix[0,2] + compliance_matrix[1,2]) 
               + 3*np.sum(np.diag(compliance_matrix[3:,3:]))))
    # shear modulus, G_H (Hill average)
    G_H = (G_R + G_V)/2

    # isotropic Poisson ratio, n_V
    n_V = (3*K_V - 2*G_V)/(6*K_V + 2*G_V)
    # isotropic Poisson ratio, n_R
    n_R = (3*K_R - 2*G_R)/(6*K_R + 2*G_R)        
    # isotropic Poisson ratio, n_H
    # n_H = (3*K_H - 2*G_H)/(6*K_H + 2*G_H)
    n_H = (n_V + n_R)/2

    # Young's modulus, E_R
    E_R = 2*(1+n_R)*G_R
    # Young's modulus, E_V
    E_V = 2*(1+n_V)*G_V
    # Young's modulus, E_H
    # E_H = 2*(1+n_H)*G_H
    E_H = (E_V + E_R)/2
    # Young's modulus, E_RH
    E_RH = (E_H + E_R)/2

    # anisotropy, A_L
    A_L = np.sqrt(np.log(K_V/K_R)**2 + 5*np.log(G_V/G_R)**2)
    
    return E_V, E_R, E_H, E_RH, K_V, K_R, K_H, G_V, G_R, G_H, n_V, n_R, n_H, A_L


def get_tersor(data, sheets, method):
    stiffness_matrices = []
    columns = []
    columns.append('ID')
    columns.append('Method')
    for i in range(6):
        for j in range(6):
            columns.append(f'C{i+1}{j+1}')

    # Get stiffness matrix
    for name in sheets:
        sheet = data[name]
        stiffness_matrix = []
        stiffness_matrix.append(name)
        stiffness_matrix.append(method)
        for i in range(6):
            for j in range(6):
                C = sheet.cell(row=5+i, column=12+j).value
                stiffness_matrix.append(C)
        stiffness_matrices.append(stiffness_matrix)
    df = pd.DataFrame(stiffness_matrices, columns=columns)
        
    return df