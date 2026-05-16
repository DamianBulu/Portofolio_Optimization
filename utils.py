# Funcțiile comune: descărcare date, Train/Test split,calcul Rata Sharpe, matrice de covarianță.


import yfinance as yf
import pandas as pd
import numpy as np
import os

TICKERS=[
'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', # Tech
    'LLY', 'JNJ', 'PFE', 'UNH', 'MRK',  # Sănătate
    'JPM', 'V', 'PYPL', 'GS',   # Finanțe
    'KO', 'WMT', 'PG', 'PEP',   # Consum
    'BA', 'INTC', 'DIS', 'XOM'  # Diverse/Industrie/Energie
]

START_DATE='2022-01-01'
SPLIT_DATE='2025-01-01' # 3 ani Train(2022-2024)
END_DATE='2026-01-01'    # 1 an  Test(2025)
RISK_FREE_RATE=0.02 #constanta din formula /Sharpe Ration


# Definim căile către fișierele noastre locale
DATA_DIR='data'
TRAIN_FILE=f'{DATA_DIR}/train_returns.csv'
TEST_FILE=f'{DATA_DIR}/test_returns.csv'
MU_FILE=f'{DATA_DIR}/mu_train.csv'
SIGMA_FILE=f'{DATA_DIR}/sigma_train.csv'



def get_processed_data():
    """
        Încarcă datele salvate local (dacă există) sau le descarcă de pe Yahoo Finance,
        le procesează și le salvează în folderul 'data' pentru utilizări ulterioare.
    """

    # Creăm folderul 'data' dacă nu există
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)




    # Verificăm dacă fișierele necesare au fost deja salvate
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE) and os.path.exists(MU_FILE) and os.path.exists(SIGMA_FILE):
        print("Se încarcă datele procesate din folderul local 'data/'...")

        test_returns=pd.read_csv(TEST_FILE,index_col='Date',parse_dates=True)

        # Citim mu și sigma și le transformăm înapoi în vectori/matrici numpy (.values)
        mu_train=pd.read_csv(MU_FILE,index_col=0).squeeze("columns").values
        sigma_train=pd.read_csv(SIGMA_FILE,index_col=0).values

        return mu_train, sigma_train, test_returns

        # Dacă fișierele nu există, executăm descărcarea



    print("Datele locale nu au fost găsite. Se descarcă de pe Yahoo Finance...")

    data=yf.download(TICKERS,start=START_DATE,end=END_DATE)['Close']

    # Calculăm randamentele procentuale zilnice
    returns=data.pct_change().dropna()

    # Impartire datele: Train vs Test
    train_returns=returns.loc[:SPLIT_DATE]
    test_returns=returns.loc[SPLIT_DATE:]


    #Procesare pt algoritmi (In-Sample/Train)      Se inmulteste cu 252 pt ca se considera 252 de zile de tranzactionare pe an
    mu_train_series=train_returns.mean()*252
    sigma_train_df=train_returns.cov()*252

    # SALVAREA DATELOR ÎN FOLDERUL 'data/'
    print("Se salvează datele în folderul 'data/'...")
    train_returns.to_csv(TRAIN_FILE)
    test_returns.to_csv(TEST_FILE)
    mu_train_series.to_csv(MU_FILE,header=['Mu'])
    sigma_train_df.to_csv(SIGMA_FILE)



    print(f"Date procesate și salvate! Antrenare: {len(train_returns)} zile, Testare: {len(test_returns)} zile.")
    return mu_train_series.values, sigma_train_df.values, test_returns

def calculate_sharpe_ratio(weights,mu,sigma,risk_free_rate=RISK_FREE_RATE):
    """
        Funcția Obiectiv (Fitness) pe care TOȚI cei 3 algoritmi trebuie să o maximizeze.
    """
    weights=np.array(weights)

    portfolio_return=np.sum(mu*weights)
    portfolio_variance=np.dot(weights.T,np.dot(sigma,weights))
    portfolio_risk=np.sqrt(portfolio_variance)

    if portfolio_risk==0:
        return 0

    sharpe_ratio=(portfolio_return-risk_free_rate)/portfolio_risk
    return sharpe_ratio

def validate_and_normalize_weights(weights):
    """
        Asigură respectarea constrângerilor: sum(ponderi) == 1 și ponderi >= 0.
    """
    weights=np.maximum(0,weights)
    total_weight=np.sum(weights)
    if total_weight>0:
        return weights/total_weight
    else:
        return np.ones(len(weights))/len(weights)

#Rulare script
if __name__=='__main__':
    print("Testare modul utils.py...")
    mu,sigma,test_data=get_processed_data()

    print("\n--- Rezultate Test ---")
    print(f"Dimensiune mu (așteptat 25): {len(mu)}")
    print(f"Dimensiune sigma (așteptat 25x25): {sigma.shape}")