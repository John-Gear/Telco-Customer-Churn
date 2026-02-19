from src.db import read_sql
import pandas as pd

BASE_QERY = "SELECT * FROM Telco-Customer-Churn"

def load_data():
    return read_sql(BASE_QERY)

df = load_data()

def main_preprocessor(df):

    # удаляем повторяющиеся строки
    cols_without_id = df.columns.drop('Идентификатор')
    df = df.drop_duplicates(subset=cols_without_id, keep='first')

    # меняем все Nan в "Совокупные платежи за всё время" на "0" (используем формат str. чтобы не возникла потом путаница)
    mask = df['Стаж клиента'] == 0
    df.loc[mask, 'Совокупные платежи за всё время'] = '0'

    # Переводим все значения колонки 'Совокупные платежи за всё время' из str в float
    df['Совокупные платежи за всё время'] = pd.to_numeric(df['Совокупные платежи за всё время'], errors='coerce')

    return df