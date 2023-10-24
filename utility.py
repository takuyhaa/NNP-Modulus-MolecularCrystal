import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 13


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
    # bulm modulus, K_RH
    K_RH = (K_R + K_H)/2

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
    # shear modulus, G_RH
    G_RH = (G_R + G_H)/2

    # isotropic Poisson ratio, n_V
    n_V = (3*K_V - 2*G_V)/(6*K_V + 2*G_V)
    # isotropic Poisson ratio, n_R
    n_R = (3*K_R - 2*G_R)/(6*K_R + 2*G_R)        
    # isotropic Poisson ratio, n_H
    n_H = (3*K_H - 2*G_H)/(6*K_H + 2*G_H)
    # n_H = (n_V + n_R)/2
    # isotropic Poisson ratio, n_RH
    n_RH = (n_R + n_H)/2

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
    
    return E_V, E_R, E_H, E_RH, K_V, K_R, K_H, K_RH, G_V, G_R, G_H, G_RH, n_H, A_L


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


# Preparation for scatter visualization
def vis_opt_lattice(df, color, save, save_name):
    column_exp = [column for column in list(df.columns) if 'exp' in column]
    column_opt = [column for column in list(df.columns) if 'opt' in column]
    df_exp = df[column_exp]
    df_opt = df[column_opt]

    order = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    labels = [r'$a$ ($\mathrm{\mathring{A}}$)',
              r'$b$ ($\mathrm{\mathring{A}}$)',
              r'$c$ ($\mathrm{\mathring{A}}$)',
              r'$\alpha$ ($^{\circ}$)',
              r'$\beta$ ($^{\circ}$)',
              r'$\gamma$ ($^{\circ}$)',
              r'$V$ ($\mathrm{\mathring{A}}^3$)',
              r'Density (g/cm$^3$)']

    fig = plt.figure(figsize=(12, 13))
    ec = 'black'
    width = 0.5
    axisfont = 15

    for i in range(len(labels)):
        ax = fig.add_subplot(3, 3, i+1)
        ax.scatter(df_exp.iloc[:,i], df_opt.iloc[:,i], c=color, ec=ec, linewidth=width)
        max_value = max(df_exp.iloc[:,i].max(), df_opt.iloc[:,i].max())*1.05
        min_value = max(df_exp.iloc[:,i].min(), df_opt.iloc[:,i].min())*0.95
        ax.plot([min_value, max_value], [min_value, max_value], c='k', linestyle='dashed')
        ax.set_xlabel(f'exp. {labels[i]}', fontsize=axisfont)
        ax.set_ylabel(f'opt. {labels[i]}', fontsize=axisfont)
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        ax.text(-0.2, 1.1, f'({order[i]})', 
                transform=ax.transAxes, 
                horizontalalignment='left', 
                verticalalignment='top', 
                fontsize=18)
    if save is True:
        fig.tight_layout()
        fig.savefig(save_name, dpi=300)
        
        
def prep_df(filepath):
    df = pd.read_csv(filepath)
    df['deltaV'] = (df['opt_V']-df['exp_V'])/df['exp_V']*100
    df['deltaV_HF'] = pd.Series()
    return df


def arrange_hf(df_nnp, df_hf):
    for ref in df_hf['refcode'].tolist():
        for i in range(df_nnp.shape[0]):
            if ref == df_nnp['refcode'][i]:
                df_nnp['deltaV_HF'][i] = df_hf[df_hf['refcode']==ref]['deltaV']
    opt_V_hf = df_nnp['deltaV_HF']*df_nnp['exp_V']/100+df_nnp['exp_V']
    df_nnp['opt_V_HF'] = opt_V_hf
    return df_nnp
