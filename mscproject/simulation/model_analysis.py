import pandas as pd
from matplotlib import pyplot as plt


# def plot4():
#     glc_uptakes = list(pd.read_csv('data/glc_uptakes.csv')['value_1'])
#     df = pd.read_csv('output/glc_pfba_fluxes.csv', index_col=0)
#     pfba = list(df.loc['net_flux', df.columns != 'control'])
#     df = pd.read_csv('output/glc_etfl_fluxes.csv', index_col=0)
#     etfl = list(df.loc['net_flux', df.columns != 'control'])
#
#     f, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
#     ax.set_xlabel("Glucose uptake (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     ax.set_ylabel("Net flux (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     plt.xlim([0, 16])
#     ax.plot(glc_uptakes, pfba, label='iML1515 pFBA')
#     ax.plot(glc_uptakes, etfl, label='iML1515 ETFL')
#     ax.legend()
#     plt.show()

# def plot3():
#     glc_uptakes = list(pd.read_csv('data/glc_uptakes.csv')['value_1'])
#     df = pd.read_csv('output/glc_pfba_iJO1366_fluxes.csv', index_col=0)
#     pfba_1366 = list(df.loc['EX_co2_e', df.columns != 'control'])
#     df = pd.read_csv('output/glc_pfba_fluxes.csv', index_col=0)
#     pfba = list(df.loc['EX_co2_e', df.columns != 'control'])
#     df = pd.read_csv('output/glc_etfl_fluxes.csv', index_col=0)
#     etfl = list(df.loc['EX_co2_e', df.columns != 'control'])
#
#     f, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=100)
#     ax.set_xlabel("Glucose uptake rate (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     ax.set_ylabel("CO$_2$ production rate (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     plt.xlim([0, 15.9])
#     plt.ylim([0, 30])
#     ax.plot(glc_uptakes, pfba_1366, label='iJO1366 + pFBA')
#     ax.plot(glc_uptakes, pfba, label='iML1515 + pFBA')
#     ax.plot(glc_uptakes, etfl, label='iML1515 + ETFL', color='red')
#     ax.legend()
#     plt.show()

def plot2():
    glc_uptakes = list(pd.read_csv('data/glc_uptakes.csv')['value_1'])
    df = pd.read_csv('output/glc_pfba_iJO1366_fluxes.csv', index_col=0)
    pfba_1366 = list(df.loc['EX_o2_e_b', df.columns != 'control'])
    df = pd.read_csv('output/glc_pfba_fluxes.csv', index_col=0)
    pfba = list(df.loc['EX_o2_e_b', df.columns != 'control'])
    df = pd.read_csv('output/glc_etfl_fluxes.csv', index_col=0)
    etfl = list(df.loc['EX_o2_e_b', df.columns != 'control'])

    f, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
    ax.set_xlabel("Glucose uptake rate (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
    ax.set_ylabel("O$_2$ consumption rate (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
    plt.xlim([0, 15.9])
    plt.ylim([0, 20])
    ax.plot(glc_uptakes, pfba_1366, label='iJO1366 + pFBA')
    ax.plot(glc_uptakes, pfba, label='iML1515 + pFBA')
    ax.plot(glc_uptakes, etfl, label='iML1515 + ETFL', color='red')

    ax.plot([0, glc_uptakes[9]], [pfba_1366[9], pfba_1366[9]], '--', lw=1, color='gray')
    ax.annotate(f'({glc_uptakes[9]}, {pfba_1366[9]})', (glc_uptakes[9], pfba_1366[9]))
    ax.plot([0, glc_uptakes[9]], [pfba[9], pfba[9]], '--', lw=1, color='gray')
    ax.annotate(f'({glc_uptakes[9]}, {pfba[9]})', (glc_uptakes[9], pfba[9]))
    ax.plot([0, glc_uptakes[9]], [etfl[9], etfl[9]], '--', lw=1, color='gray')
    ax.annotate(f'({glc_uptakes[9]}, {etfl[9]})', (glc_uptakes[9], etfl[9]))
    ax.plot([glc_uptakes[9], glc_uptakes[9]], [0, pfba[9]], '--', lw=1, color='gray')

    ax.legend()
    plt.show()


def plot1():
    glc_uptakes = list(pd.read_csv('data/glc_uptakes.csv')['value_1'])
    df = pd.read_csv('output/glc_pfba_iJO1366_fluxes.csv', index_col=0)
    pfba_1366 = list(df.loc['BIOMASS_Ec_iJO1366_core_53p95M', df.columns != 'control'])
    df = pd.read_csv('output/glc_pfba_fluxes.csv', index_col=0)
    pfba = list(df.loc['BIOMASS_Ec_iML1515_core_75p37M', df.columns != 'control'])
    df = pd.read_csv('output/glc_etfl_fluxes.csv', index_col=0)
    etfl = list(df.loc['BIOMASS_Ec_iML1515_core_75p37M', df.columns != 'control'])

    f, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
    ax.set_xlabel("Glucose uptake rate (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
    ax.set_ylabel("Growth rate (h$^{-1}$)")
    plt.xlim([0, 15.9])
    plt.ylim([0, 1.3])
    ax.plot(glc_uptakes, pfba_1366, label='iJO1366 + pFBA')
    ax.plot(glc_uptakes, pfba, label='iML1515 + pFBA')
    ax.plot(glc_uptakes, etfl, label='iML1515 + ETFL', color='red')

    ax.plot([0, glc_uptakes[9]], [pfba_1366[9], pfba_1366[9]], '--', lw=1, color='gray')
    ax.annotate(f'({glc_uptakes[9]}, {pfba_1366[9]})', (glc_uptakes[9], pfba_1366[9]))
    ax.plot([0, glc_uptakes[9]], [pfba[9], pfba[9]], '--', lw=1, color='gray')
    ax.annotate(f'({glc_uptakes[9]}, {pfba[9]})', (glc_uptakes[9], pfba[9]))
    ax.plot([0, glc_uptakes[9]], [etfl[9], etfl[9]], '--', lw=1, color='gray')
    ax.annotate(f'({glc_uptakes[9]}, {etfl[9]})', (glc_uptakes[9], etfl[9]))
    ax.plot([glc_uptakes[9], glc_uptakes[9]], [0, pfba_1366[9]], '--', lw=1, color='gray')

    ax.legend()
    plt.show()


# def plot2():
#     glc_uptakes = list(pd.read_csv('data/glc_uptakes.csv')['value_1'])
#     df = pd.read_csv('output/glc_pfba_iJO1366_fluxes.csv', index_col=0)
#     bm_1366 = list(df.loc['net_flux', df.columns != 'control'])
#     df = pd.read_csv('output/glc_pfba_fluxes.csv', index_col=0)
#     bm_1515 = list(df.loc['net_flux', df.columns != 'control'])
#
#     f, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
#     # ax.set_title(f"iJO1366 vs. iML1515")
#     ax.set_xlabel("Glucose uptake (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     ax.set_ylabel("Net flux (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     plt.xlim([0, 16])
#     # plt.ylim([0, 1.6])
#     ax.plot(glc_uptakes, bm_1366, label='iJO1366 pFBA')
#     ax.plot(glc_uptakes, bm_1515, label='iML1515 pFBA')
#     ax.legend()
#     plt.show()
#
#
# def plot1():
#     glc_uptakes = list(pd.read_csv('data/glc_uptakes.csv')['value_1'])
#     df = pd.read_csv('output/glc_pfba_iJO1366_fluxes.csv', index_col=0)
#     bm_1366 = list(df.loc['BIOMASS_Ec_iJO1366_core_53p95M', df.columns != 'control'])
#     df = pd.read_csv('output/glc_pfba_fluxes.csv', index_col=0)
#     bm_1515 = list(df.loc['BIOMASS_Ec_iML1515_core_75p37M', df.columns != 'control'])
#
#     f, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
#     # ax.set_title(f"iJO1366 vs. iML1515")
#     ax.set_xlabel("Glucose uptake (mmol$\cdot$gDW$^{-1}\cdot$h$^{-1}$)")
#     ax.set_ylabel("Growth rate (h$^{-1}$)")
#     plt.xlim([0, 16])
#     # plt.ylim([0, 1.6])
#     ax.plot(glc_uptakes, bm_1366, label='iJO1366 pFBA')
#     ax.plot(glc_uptakes, bm_1515, label='iML1515 pFBA')
#     ax.legend()
#     plt.show()


if __name__ == '__main__':
    plot1()
    plot2()
