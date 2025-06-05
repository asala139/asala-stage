
def debt_equity(df_global):
    numerator = df_global[' D. TOTALE DEBITI']
    denominator = df_global[' A. TOTALE PATRIMONIO NETTO']
    return numerator / denominator

def nfe(df_global):
    return df_global[' C. TOTALE PROVENTI E ONERI FINANZIARI']

def fo(df_global):
    return(df_global['  D.1. Obblig.ni entro'] + df_global['  D.2. Obblig.ni convert. entro'] + df_global['  D.3. Soci per Finanziamenti entro'] +
            df_global["  D.4. Banche entro l'esercizio"] + df_global['  D.5. Altri finanziatori entro'] + df_global["  D.8. Titoli di credito entro"] +
             df_global["  D.9. Imprese Controllate entro"] + df_global["  D.10. Imprese Collegate entro"] + df_global["  D.11. Controllanti entro"]+
             df_global["  A.I. Capitale sociale"])

def fa(df_global):
    return (df_global["  B.III. TOTALE IMMOB. FINANZIARIE"].to_numpy() + df_global["  C.III. TOTALE ATTIVITA' FINANZIARIE"].to_numpy())

def nfo(df_global):
    return fo(df_global) - fa(df_global)

def ol(df_global):
    return (df_global[" C. TRATTAMENTO DI FINE RAPPORTO"].to_numpy() + df_global["  D.6. Acconti entro"].to_numpy() + df_global["  D.7. Fornitori entro"].to_numpy()+
            df_global["  D.12. Debiti Tributari entro"].to_numpy() + df_global["  D.13. Istituti previdenza entro"].to_numpy())

def ta(df_global):
    return df_global[" TOTALE ATTIVO"].to_numpy()

def oa(df_global):
    return ta(df_global) - fa(df_global)

def tl_ps(df_global):
    return df_global[" TOTALE PASSIVO"]

def oi(df_global):
    return df_global[" RISULTATO OPERATIVO"]
def cni(df_global):
    return oi(df_global) - nfe(df_global)

def nbc(df_global_year, df_global_prec_year):
    return nfe(df_global_year) / nfo(df_global_prec_year)

def noa(df_global):
    return oa(df_global) - ol(df_global)
def ato(df_global):
    numerator = df_global["  A.1. Ricavi vendite e prestazioni"]
    denominator = noa(df_global)
    return numerator / denominator

def pm(df_global):
    numerator = ol(df_global)
    denominator = df_global["  A.1. Ricavi vendite e prestazioni"]
    return numerator / denominator

def ollev(df_global):
    return ol(df_global) / noa(df_global)

def rnoa(df_global_year, df_global_prec_year):
    numerator = df_global_year[" RISULTATO OPERATIVO"].to_numpy()
    denominator = noa(df_global_prec_year)
    return numerator / denominator

def spread(df_global_year, df_global_prec_year):
    return rnoa(df_global_year, df_global_prec_year) - nbc(df_global_year, df_global_prec_year)