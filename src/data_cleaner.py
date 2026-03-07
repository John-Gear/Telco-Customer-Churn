from src.db import read_sql
import pandas as pd

BASE_QERY = "SELECT * FROM TelcoCustomerChurn"

def load_data():
    return read_sql(BASE_QERY)

def main_cleaner(df):

    # удаляем повторяющиеся строки
    cols_without_id = df.columns.drop('customerID')
    df = df.drop_duplicates(subset=cols_without_id, keep='first')

    # меняем все Nan в "TotalCharges" на "0" (используем формат str. чтобы не возникла потом путаница)
    mask = df['tenure'] == 0
    df.loc[mask, 'TotalCharges'] = '0'

    # Переводим все значения колонки 'TotalCharges' из str в float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    return df