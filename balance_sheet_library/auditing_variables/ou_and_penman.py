import pandas as pd
import numpy as np
def current_ratio(df_global):
    numerator = (df_global['   B.III. CREDITI FIN. A BREVE'].to_numpy() + df_global[' C. ATTIVO CIRCOLANTE'].to_numpy() +
                 df_global['   C.II. Crediti a oltre'].to_numpy() + df_global[' D. RATEI E RISCONTI'].to_numpy())
    denominator = df_global['  D. DEBITI A BREVE'].to_numpy() + df_global[' E. RATEI E RISCONTI'].to_numpy()
    return (numerator / denominator)

def change_in_current_ratio(df_global_year, df_global_prec_year):
    c_r_year = current_ratio(df_global_year)
    c_r_prec_year = current_ratio(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def quick_ratio(df_global):
    numerator = (df_global['   B.III. CREDITI FIN. A BREVE'].to_numpy() + df_global[' C. ATTIVO CIRCOLANTE'].to_numpy() +
                 df_global['   C.II. Crediti a oltre'].to_numpy() + df_global[' D. RATEI E RISCONTI'].to_numpy() +
                 df_global['  C.I. TOTALE RIMANENZE'].to_numpy())
    denominator = df_global['  D. DEBITI A BREVE'].to_numpy() + df_global[' E. RATEI E RISCONTI'].to_numpy()
    return (numerator / denominator)

def change_in_quick_ratio(df_global_year, df_global_prec_year):
    c_r_year = quick_ratio(df_global_year)
    c_r_prec_year = quick_ratio(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def days_sales_in_accounts_receivable(df_global):
    numerator = df_global['  C.II. TOTALE CREDITI'].to_numpy()
    denominator = df_global['  A.1. Ricavi vendite e prestazioni'].to_numpy() * 365
    return (numerator / denominator)

def change_in_days_sales_in_accounts_receivable(df_global_year, df_global_prec_year):
    c_r_year = days_sales_in_accounts_receivable(df_global_year)
    c_r_prec_year = days_sales_in_accounts_receivable(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def inventory_turnover(df_global):
    numerator = df_global['  A.1. Ricavi vendite e prestazioni'].to_numpy() * 365
    denominator = df_global['  C.I. TOTALE RIMANENZE'].to_numpy()
    return (numerator / denominator)
def change_in_inventory_turnover(df_global_year, df_global_prec_year):
    c_r_year = inventory_turnover(df_global_year)
    c_r_prex_year = inventory_turnover(df_global_prec_year)
    return (c_r_year - c_r_prex_year) / c_r_year

def inventory_total_assets(df_global):
    numerator = df_global['  C.I. TOTALE RIMANENZE'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)
def change_in_inventory_total_assets(df_global_year, df_global_prec_year):
    c_r_year = inventory_total_assets(df_global_year)
    c_r_prec_year = inventory_total_assets(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year
def change_in_inventory(df_global_year, df_global_prec_year):
    t_r_year = df_global_year['  C.I. TOTALE RIMANENZE'].to_numpy()
    t_r_prec_year = df_global_prec_year['  C.I. TOTALE RIMANENZE'].to_numpy()
    return (t_r_year - t_r_prec_year) / t_r_year

def change_in_sales(df_global_year, df_global_prec_year):
    t_r_year = df_global_year['  Ricavi delle vendite'].to_numpy()
    t_r_prec_year = df_global_prec_year['  Ricavi delle vendite'].to_numpy()
    return (t_r_year - t_r_prec_year) / t_r_year

def change_in_depreciation(df_global_year, df_global_prec_year):
    t_r_year = df_global_year['  B.10. TOT Ammortamenti e svalut.'].to_numpy()
    t_r_prec_year = df_global_prec_year['  B.10. TOT Ammortamenti e svalut.'].to_numpy()
    return (t_r_year - t_r_prec_year) / t_r_year

def depreciation_plant_assets(df_global):
    numerator = df_global['  B.10. TOT Ammortamenti e svalut.'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)

def change_in_depreciation_plant_assets(df_global_year, df_global_prec_year):
    c_r_year = depreciation_plant_assets(df_global_year)
    c_r_prec_year = depreciation_plant_assets(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def return_on_opening_equity(df_global):
    numerator = df_global['  21. UTILE/PERDITA DI ESERCIZIO'].to_numpy()
    denominator = df_global[' A. TOTALE PATRIMONIO NETTO'].to_numpy()
    return (numerator / denominator)

def debt_equity_ratio(df_global):
    numerator = df_global[' D. TOTALE DEBITI'].to_numpy()
    denominator = df_global[' A. TOTALE PATRIMONIO NETTO'].to_numpy()
    return (numerator / denominator)
def change_in_debt_equity_ratio(df_global_year, df_global_prec_year):
    c_r_year = debt_equity_ratio(df_global_year)
    c_r_prec_year = debt_equity_ratio(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def lt_debt_to_equity(df_global):
    numerator = df_global['  D. DEBITI A OLTRE'].to_numpy()
    denominator = df_global[' A. TOTALE PATRIMONIO NETTO'].to_numpy()
    return (numerator / denominator)

def change_in_lt_debt_to_equity(df_global_year, df_global_prec_year):
    c_r_year = lt_debt_to_equity(df_global_year)
    c_r_prec_year = lt_debt_to_equity(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def equity_to_fixed_assets(df_global):
    numerator = df_global[' A. TOTALE PATRIMONIO NETTO'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)

def change_in_equity_to_fixed_assets(df_global_year, df_global_prec_year):
    c_r_year = equity_to_fixed_assets(df_global_year)
    c_r_prec_year = equity_to_fixed_assets(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def times_interest_earned(df_global):
    numerator = df_global[' RISULTATO OPERATIVO'].to_numpy()
    denominator = df_global['  C.17. Totale Oneri finanziari'].to_numpy()
    return (numerator / denominator)

def change_in_times_interest_earned(df_global_year, df_global_prec_year):
    c_r_year = times_interest_earned(df_global_year)
    c_r_prec_year = times_interest_earned(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def sales_total_assets(df_global):
    numerator = df_global['  Ricavi delle vendite'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)

def change_in_sales_total_assets(df_global_year, df_global_prec_year):
    c_r_year = sales_total_assets(df_global_year)
    c_r_prec_year = sales_total_assets(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def return_on_total_assets(df_global):
    numerator = df_global['  21. UTILE/PERDITA DI ESERCIZIO'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)

def return_on_closing_equity(df_global):
    numerator = df_global['  21. UTILE/PERDITA DI ESERCIZIO'].to_numpy()
    denominator = df_global[' A. TOTALE PATRIMONIO NETTO'].to_numpy()
    return (numerator / denominator)

def gross_margin_ratio(df_global):
    numerator = df_global['  Ricavi delle vendite'].to_numpy() - df_global[' B. COSTI DELLA PRODUZIONE'].to_numpy()
    denominator = df_global['  Ricavi delle vendite'].to_numpy()
    return (numerator / denominator)
def change_in_gross_margin_ratio(df_global_year, df_global_prec_year):
    c_r_year = gross_margin_ratio(df_global_year)
    c_r_prec_year = gross_margin_ratio(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def operating_profit_before_depreciation_to_sales(df_global):
    numerator = df_global[' RISULTATO OPERATIVO'].to_numpy()
    denominator = df_global['  Ricavi delle vendite'].to_numpy()
    return (numerator / denominator)

def change_in_operating_profit_before_depreciation_to_sales(df_global_year, df_global_prec_year):
    c_r_year = operating_profit_before_depreciation_to_sales(df_global_year)
    c_r_prec_year = operating_profit_before_depreciation_to_sales(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def pretax_income_to_sales(df_global):
    numerator = df_global[' RISULTATO PRIMA DELLE IMPOSTE'].to_numpy()
    denominator = df_global['  Ricavi delle vendite'].to_numpy()
    return (numerator / denominator)

def change_in_pretax_income_to_sales(df_global_year, df_global_prec_year):
    c_r_year = pretax_income_to_sales(df_global_year)
    c_r_prec_year = pretax_income_to_sales(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def net_profit_margin(df_global):
    numerator = df_global['  21. UTILE/PERDITA DI ESERCIZIO'].to_numpy()
    denominator = df_global['  Ricavi delle vendite'].to_numpy()
    return (numerator / denominator)

def change_in_net_profit_margin(df_global_year, df_global_prec_year):
    c_r_year = net_profit_margin(df_global_year)
    c_r_prec_year = net_profit_margin(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def sales_to_total_cash(df_global):
    numerator = df_global['  C.IV. TOT. DISPON. LIQUIDE'].to_numpy()
    denominator = df_global['  Ricavi delle vendite'].to_numpy()
    return (numerator / denominator)
def sales_to_accounts_receivable(df_global):
    numerator = df_global['  C.II. TOTALE CREDITI'].to_numpy()
    denominator = df_global['  Ricavi delle vendite'].to_numpy()
    return (numerator / denominator)

def sales_to_working_capital(df_global):
    numerator = df_global['  Ricavi delle vendite'].to_numpy()
    denominator = df_global[' C. ATTIVO CIRCOLANTE'].to_numpy()
    return (numerator / denominator)

def change_in_sales_to_working_capital(df_global_year, df_global_prec_year):
    c_r_year = sales_to_working_capital(df_global_year)
    c_r_prec_year = sales_to_working_capital(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year

def sales_to_fixed_assets(df_global):
    numerator = df_global['  Ricavi delle vendite'].to_numpy()
    denominator = df_global[' B. TOTALE IMMOBILIZZAZIONI sep.ind. Di quelle conc. In loc. Finanz.'].to_numpy()
    return (numerator / denominator)

def change_in_total_assets(df_global_year, df_global_prec_year):
    c_r_year = df_global_year[' TOTALE ATTIVO'].to_numpy()
    c_r_prec_year = df_global_prec_year[' TOTALE ATTIVO'].to_numpy()
    return (c_r_year - c_r_prec_year) / c_r_year

def cash_flow_to_total_debt(df_global):
    numerator = df_global['   - Flusso di cassa di gestione'].to_numpy()
    denominator = df_global[' D. TOTALE DEBITI'].to_numpy()
    return (numerator / denominator)

def working_capital_total_assets(df_global):
    numerator = df_global[' C. ATTIVO CIRCOLANTE'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)
def change_in_working_capital_total_assets(df_global_year, df_global_prec_year):
    c_r_year = working_capital_total_assets(df_global_year)
    c_r_prec_year = working_capital_total_assets(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year
def operating_income_total_assets(df_global):
    numerator = df_global[' RISULTATO OPERATIVO'].to_numpy()
    denominator = df_global[' TOTALE ATTIVO'].to_numpy()
    return (numerator / denominator)
def change_in_operating_income_total_assets(df_global_year, df_global_prec_year):
    c_r_year = operating_income_total_assets(df_global_year)
    c_r_prec_year = operating_income_total_assets(df_global_prec_year)
    return (c_r_year - c_r_prec_year) / c_r_year
def repayment_of_lt_debt_as_percentage_of_total_lt_debt(df_global_year, df_global_prec_year):
    numerator = df_global_year['  D. DEBITI A OLTRE'].to_numpy() - df_global_prec_year['  D. DEBITI A OLTRE'].to_numpy()
    denominator = df_global_prec_year['  D. DEBITI A OLTRE'].to_numpy()
    return (numerator / denominator)
def issuance_of_lt_debt_as_percentage_of_total_lt_debt(df_global_year, df_global_prec_year):
    numerator = df_global_year['  D. DEBITI A OLTRE'].to_numpy() - df_global_prec_year['  D. DEBITI A OLTRE'].to_numpy()
    denominator = df_global_year['  D. DEBITI A OLTRE'].to_numpy()
    return (numerator / denominator)
def change_in_lt_debt(df_global_year, df_global_prec_year):
    numerator = df_global_year['  D. DEBITI A OLTRE'].to_numpy()
    denominator = df_global_prec_year['  D. DEBITI A OLTRE'].to_numpy()
    return (numerator / denominator)
def change_in_working_capital(df_global_year, df_global_prec_year):
    numerator = df_global_year[' C. ATTIVO CIRCOLANTE'].to_numpy()
    denominator = df_global_prec_year[' C. ATTIVO CIRCOLANTE'].to_numpy()
    return (numerator / denominator)

def net_income_over_cash_flow(df_global):
    numerator = df_global['  21. UTILE/PERDITA DI ESERCIZIO'].to_numpy()
    denominator = df_global['   - Flusso di cassa di gestione'].to_numpy()
    return (numerator / denominator)

def generate_ds(ds, ds_base, start_year=2013, end_year=2021):
    cr = current_ratio(ds_base)
    ccr = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        ccr.extend(change_in_current_ratio(n_y_ds, y_ds))
    ccr = np.array(ccr)

    qr = quick_ratio(ds_base)
    cqr = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cqr.extend(change_in_quick_ratio(n_y_ds, y_ds))
    cqr = np.array(cqr)

    dsar = days_sales_in_accounts_receivable(ds_base)
    cdsar = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cdsar.extend(change_in_days_sales_in_accounts_receivable(n_y_ds, y_ds))
    cdsar = np.array(cdsar)

    it = inventory_turnover(ds_base)
    cit = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cit.extend(change_in_inventory_turnover(n_y_ds, y_ds))
    cit = np.array(cit)

    ita = inventory_total_assets(ds_base)
    cita = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cita.extend(change_in_inventory_total_assets(n_y_ds, y_ds))
    cita = np.array(cita)

    ci = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        ci.extend(change_in_inventory(n_y_ds, y_ds))
    ci = np.array(ci)

    cs = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cs.extend(change_in_sales(n_y_ds, y_ds))
    cs = np.array(cs)

    cd = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cd.extend(change_in_depreciation(n_y_ds, y_ds))
    cd = np.array(cd)

    dpa = depreciation_plant_assets(ds_base)
    cdpa = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cdpa.extend(change_in_depreciation_plant_assets(n_y_ds, y_ds))
    cdpa = np.array(cdpa)

    roe = return_on_opening_equity(ds_base)

    der = return_on_opening_equity(ds_base)
    cder = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cder.extend(change_in_debt_equity_ratio(n_y_ds, y_ds))
    cder = np.array(cder)

    ltde = lt_debt_to_equity(ds_base)
    cltde = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cltde.extend(change_in_lt_debt_to_equity(n_y_ds, y_ds))
    cltde = np.array(cltde)

    efa = equity_to_fixed_assets(ds_base)
    cefa = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cefa.extend(change_in_equity_to_fixed_assets(n_y_ds, y_ds))
    cefa = np.array(cefa)

    tie = times_interest_earned(ds_base)
    ctie = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        ctie.extend(change_in_times_interest_earned(n_y_ds, y_ds))
    ctie = np.array(ctie)

    sta = sales_total_assets(ds_base)
    csta = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        csta.extend(change_in_sales_total_assets(n_y_ds, y_ds))
    csta = np.array(csta)

    roa = return_on_total_assets(ds_base)

    rce = return_on_closing_equity(ds_base)

    gmr = gross_margin_ratio(ds_base)
    cgmr = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cgmr.extend(change_in_gross_margin_ratio(n_y_ds, y_ds))
    cgmr = np.array(cgmr)

    opbds = operating_profit_before_depreciation_to_sales(ds_base)
    copbds = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        copbds.extend(change_in_operating_profit_before_depreciation_to_sales(n_y_ds, y_ds))
    copbds = np.array(copbds)

    pis = pretax_income_to_sales(ds_base)
    cpis = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cpis.extend(change_in_pretax_income_to_sales(n_y_ds, y_ds))
    cpis = np.array(cpis)

    npm = net_profit_margin(ds_base)
    cnpm = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cnpm.extend(change_in_net_profit_margin(n_y_ds, y_ds))
    cnpm = np.array(cnpm)

    stc = sales_to_total_cash(ds_base)

    sar = sales_to_accounts_receivable(ds_base)

    swc = sales_to_working_capital(ds_base)
    cswc = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cswc.extend(change_in_sales_to_working_capital(n_y_ds, y_ds))
    cswc = np.array(cswc)

    sfa = sales_to_fixed_assets(ds_base)

    cta = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cta.extend(change_in_total_assets(n_y_ds, y_ds))
    cta = np.array(cta)

    cftd = cash_flow_to_total_debt(ds_base)

    wcta = working_capital_total_assets(ds_base)
    cwcta = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cwcta.extend(change_in_working_capital_total_assets(n_y_ds, y_ds))
    cwcta = np.array(cwcta)

    oita = operating_income_total_assets(ds_base)
    coita = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        coita.extend(change_in_operating_income_total_assets(n_y_ds, y_ds))
    coita = np.array(coita)

    lt_perc = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        lt_perc.extend(repayment_of_lt_debt_as_percentage_of_total_lt_debt(n_y_ds, y_ds))
    lt_perc = np.array(lt_perc)

    iss = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        iss.extend(issuance_of_lt_debt_as_percentage_of_total_lt_debt(n_y_ds, y_ds))
    iss = np.array(iss)

    cltd = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cltd.extend(change_in_lt_debt(n_y_ds, y_ds))
    cltd = np.array(cltd)

    cwc = []
    for y in range(start_year, end_year):
        next_year = y + 1
        y_ds = ds[ds['years'] == y]
        n_y_ds = ds[ds['years'] == next_year]
        cwc.extend(change_in_working_capital(n_y_ds, y_ds))
    cwc = np.array(cwc)

    niocf = net_income_over_cash_flow(ds_base)

    new_ds = np.array(
       [cr, ccr, qr, cqr, dsar, cdsar, it, cit, ita, cita, ci, cs, cd, dpa, cdpa, roe, der, cder, ltde, cltde, efa,
        cefa, tie, ctie, sta, csta, roa, rce, gmr, cgmr, opbds, copbds, pis, cpis, npm, cnpm, stc, sar, swc,
        cswc, sfa, cta, cftd, wcta, cwcta, oita, coita, lt_perc, iss, cltd, cwc, niocf])
    new_ds = new_ds.T

    columns_name = ['cr', 'ccr', 'qr', 'cqr', 'dsar', 'cdsar', 'it', 'cit', 'ita', 'cita', 'ci', 'cs', 'cd', 'dpa', 'cdpa',
                    'roe', 'der', 'cder', 'ltde', 'cltde', 'efa', 'cefa', 'tie', 'ctie', 'sta', 'csta', 'roa', 'rce', 'gmr',
                    'cgmr', 'opbds', 'copbds', 'pis', 'cpis', 'npm', 'cnpm', 'stc', 'sar', 'swc', 'cswc', 'sfa', 'cta', 'cftd',
                    'wcta', 'cwcta', 'oita', 'coita', 'lt_perc', 'iss', 'cltd', 'cwc', 'niocf']

    new_ds = pd.DataFrame(new_ds, columns=columns_name)
    new_ds = new_ds.reset_index()

    return new_ds