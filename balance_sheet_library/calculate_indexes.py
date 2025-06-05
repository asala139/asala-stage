import balance_sheet_library.preprocessing as preprocessing

def debt_ratio(ds):
    first_number_names = ["  D.1. Obblig.ni entro",
                          "  D.1. Obblig.ni oltre",
                          "  D.2. Obblig.ni convert. entro",
                          "  D.2. Obblig.ni convert. oltre.",
                          "  D.3. Soci per Finanziamenti entro",
                          "  D.3. Soci per Finanziamenti oltre",
                          "  D.4. Banche entro l'esercizio",
                          "  D.4. Banche oltre l'esercizio",
                          "  D.5. Altri finanziatori entro",
                          "  D.5. Altri finanziatori oltre"]
    first_number = [0] * len(ds['name'])
    for name in first_number_names:
        ds[name] = preprocessing.substitution_with_zero(ds[name])
        ds[name] = ds[name].astype(float)
        first_number = first_number + ds[name].to_numpy()
    second_number_names = "  C.IV. TOT. DISPON. LIQUIDE"
    ds[second_number_names] = preprocessing.substitution_with_zero(ds[second_number_names])
    ds[second_number_names] = ds[second_number_names].astype(float)
    second_number = ds[second_number_names].to_numpy()
    value = first_number - second_number
    ds['debt_ratio'] = value
    return ds

def credit_access(ds):
    first_number_names = ["  D.1. Obblig.ni entro",
                          "  D.1. Obblig.ni oltre",
                          "  D.2. Obblig.ni convert. entro",
                          "  D.2. Obblig.ni convert. oltre.",
                          "  D.3. Soci per Finanziamenti entro",
                          "  D.3. Soci per Finanziamenti oltre",
                          "  D.4. Banche entro l'esercizio",
                          "  D.4. Banche oltre l'esercizio",
                          "  D.5. Altri finanziatori entro",
                          "  D.5. Altri finanziatori oltre"]
    first_number = [0] * len(ds['name'])
    for name in first_number_names:
        ds[name] = preprocessing.substitution_with_zero(ds[name])
        ds[name] = ds[name].astype(float)
        first_number = first_number + ds[name].to_numpy()
    second_number_names = "  C.17. Totale Oneri finanziari"
    ds[second_number_names] = preprocessing.substitution_with_zero(ds[second_number_names])
    ds[second_number_names] = ds[second_number_names].astype(float)
    second_number = ds[second_number_names].to_numpy()
    value = first_number / second_number
    ds['credit_access'] = value
    return ds