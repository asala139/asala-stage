import datetime
import os
import pandas as pd
import pathlib
def single_file_multiple_pages(filename, tag_file="aida_tags.csv", save_to_csv=False, file_out="out.csv"):
    file = pd.ExcelFile(filename)
    societies = []
    for sheet_name in file.sheet_names:
        df = pd.read_excel(file, sheet_name)
        balance = parse_aida_balance(df, tag_file)
        societies.append(balance)
    df = pd.DataFrame(societies)
    if save_to_csv:
        df.to_csv(file_out)
    return df


def single_file_folder(folder, tag_file="aida_tags.csv", save_to_csv=False, file_out="out.csv"):
    societies = []
    for file in os.listdir(folder):
        df = pd.read_excel(folder+file)
        balance = parse_aida_balance(df, tag_file)
        societies.append(balance)
    df = pd.DataFrame(societies)
    if save_to_csv:
        df.to_csv(file_out)
    return df
def parse_aida_balance(df, tag_file="aida_tags.csv", last_year=2022):
    rows = []
    for i in range(len(df)):
        cleaned_row = [x for x in df.iloc[i].to_list() if str(x) != 'nan']
        rows.append(cleaned_row)

    city = str(df.iloc[1][0])
    code = str(rows[1][2])
    name = str(df.columns[1])
    CCIAA_code = str(rows[2][1])

    columns = pd.read_csv(tag_file)
    columns = columns['Tag_name'].tolist()

    dict_res = {'name': name, 'fiscal_code': code, 'city': city, 'CCIAA_code': CCIAA_code}

    flag_finanziario = False
    for row in rows:
        if 'Profilo finanziario e dipendenti' in row:
            flag_finanziario = True
        number_found = 0
        for column in columns:
            if column in row:
                dato_anno = []
                if flag_finanziario:
                    for i in range(dict_res['Numero di anni disponibili']):
                        index = row.index(column)
                        dato_anno.append(row[index + i + 1])
                    dict_res[column] = dato_anno
                else:
                    number_found += 1
                    index = row.index(column)
                    value = row[index + 1]
                    if type(row[index + 1]) == datetime.datetime:
                        value = row[index + 1].strftime('%Y/%m/%d')
                    dict_res[column] = value
    anni = []
    for i in range(dict_res['Numero di anni disponibili']):
        anni.append(last_year - i)
    dict_res['years'] = anni
    return dict_res

def read_entire_dataset(base_folder, out_file, save_pickle=False, save_csv=False):
    df_global = pd.DataFrame([])
    for file in os.listdir(base_folder):
        if pathlib.Path(file).suffix == '.xls':
            df = single_file_multiple_pages(filename=base_folder + file)
            if df_global.empty:
                df_global = df
            else:
                df_global = pd.concat([df_global, df])
        else:
            df = single_file_folder(base_folder + file + "/")
            if df_global.empty:
                df_global = df
            else:
                df_global = pd.concat([df_global, df])

    if save_csv:
        df_global.to_csv(out_file+".csv", index=False)
    if save_pickle:
        df_global.to_pickle(out_file+".pkl")
    return df_global
