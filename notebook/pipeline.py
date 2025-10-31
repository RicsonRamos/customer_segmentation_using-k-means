import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR # Importa o SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------
# FUNÇÕES AUXILIARES E DE CARREGAMENTO
# -----------------------------------------------------------------------

def load_dataset(path):
    """
    Carrega um dataset CSV e exibe informações básicas.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado no caminho: {path}")
        return None
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        return None

    display(df.head(3))
    print("=== SHAPE ===")
    print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}\n")

    print("=== INFO ===")
    df.info()
    print("\n")

    print("=== DESCRIBE ===")
    print(df.describe())

    print("\n=== VALORES NULOS ===")
    print(df.isnull().sum())

    print("\n=== TIPOS DE DADOS ===")
    print(df.dtypes.value_counts())

    print("\n=== Valores ===")
    print(f"Total de valores únicos no DataFrame: {df.nunique().sum()}")
    return df


def preprocess(df: pd.DataFrame, max_unique_for_dummies: int = 10) -> pd.DataFrame:
    """
    Aplica pré-processamento, incluindo remoção de colunas de alta cardinalidade,
    codificação categórica e extração de features de data/hora.
    """
    if df is None:
        print("DataFrame de entrada é None. Retornando.")
        return None

    df = df.copy()

    # --- 1. Remoção de colunas categóricas de alta cardinalidade ---
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cols_to_drop = [col for col in cat_cols if df[col].nunique() > max_unique_for_dummies]
    if cols_to_drop:
        print(f"Colunas removidas (alta cardinalidade): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # --- 2. Codificação automática de colunas categóricas ---
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        # Se for para One-Hot Encoding ou se tiver sido removida (mas não está aqui)
        if n_unique <= max_unique_for_dummies:
            # One-hot encoding
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
            print(f"{col}: codificado com One-Hot Encoding ({n_unique} categorias)")
        else:
            # Label encoding (para as colunas que não foram dropadas, mas que têm mais de 10)
            le = LabelEncoder()
            # Garante que NaNs sejam tratados como uma categoria ('nan') antes de codificar
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"{col}: codificado com LabelEncoder ({n_unique} categorias)")


    # --- 3. Tratamento de datas e horas ---
    data_processada = False

    # Processa 'Date'
    if 'Date' in df.columns:
        col_date = 'Date'
        try:
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
            df[col_date + '_year'] = df[col_date].dt.year
            df[col_date + '_month'] = df[col_date].dt.month
            df[col_date + '_day'] = df[col_date].dt.day
            df[col_date + '_weekday'] = df[col_date].dt.weekday # 0=Segunda, 6=Domingo
            df = df.drop(columns=[col_date])
            print(f"'{col_date}' convertida e features de data extraídas.")
            data_processada = True
        except Exception as e:
            print(f"Erro ao processar a coluna '{col_date}': {e}")

    # Processa 'Time'
    if 'Time' in df.columns:
        col_time = 'Time'
        try:
            df[col_time + '_datetime'] = pd.to_datetime(df[col_time], errors='coerce')
            df[col_time + '_hour'] = df[col_time + '_datetime'].dt.hour
            df[col_time + '_minute'] = df[col_time + '_datetime'].dt.minute
            df[col_time + '_second'] = df[col_time + '_datetime'].dt.second
            df = df.drop(columns=[col_time, col_time + '_datetime'], errors='ignore')
            print(f"'{col_time}' convertida e features de tempo extraídas.")
            data_processada = True
        except Exception as e:
            print(f"Erro ao processar a coluna '{col_time}': {e}")

    if not data_processada:
        print("Nenhuma coluna 'Date' ou 'Time' encontrada ou processada.")

    # --- 4. Tratamento de valores faltantes (NaN) ---
    # Preenche NaNs com a média nas colunas numéricas restantes
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            print(f"NaNs em '{col}' preenchidos com a média.")

    # Colunas categóricas restantes (se houver, após One-Hot, não deve haver)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('missing')
            print(f"NaNs em '{col}' preenchidos com 'missing'.")

    print("\nPré-processamento completo! (Colunas não numéricas e NaNs foram tratados)")
    return df


def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove colunas numéricas que estão altamente correlacionadas.
    Esta função foi mantida única e consolidada.
    """
    if df is None:
        print("DataFrame de entrada é None. Retornando.")
        return None

    # Seleciona apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) < 2:
        print("Menos de duas colunas numéricas para calcular correlação. Retornando.")
        return df

    # Calcula matriz de correlação (apenas valores absolutos)
    corr_matrix = df[numeric_cols].corr().abs()

    # Cria uma máscara para a matriz superior triangular (excluindo a diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Colunas para remover
    cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

    if cols_to_drop:
        cols_to_drop = list(set(cols_to_drop))
        print(f"Colunas removidas por alta correlação (>{threshold}): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')
    else:
        print("Nenhuma coluna com correlação alta encontrada.")

    return df

# -----------------------------------------------------------------------
# FUNÇÕES DE MODELAGEM (Regressão e Segmentação)
# -----------------------------------------------------------------------

def compare_regression_models(df, target, corr_threshold=0.7):
    print(f"--- Iniciando Comparação de Modelos de Regressão para '{target}' ---")

    R2_MINIMO = 0.70  # Valor de corte mínimo aceitável

    X = df.drop(columns=[target])
    X_limpo = remove_highly_correlated(X, threshold=corr_threshold)
    y = df[target]

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                      max_depth=3, random_state=42, loss='squared_error')
    }

    best_r2_cv = -np.inf
    best_model_name = None
    best_model = None
    results = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_limpo, y, cv=5, scoring='r2', n_jobs=-1)
            mean_r2 = np.mean(scores)
            results[name] = {'R2_CV_Medio': mean_r2, 'R2_CV_Std': np.std(scores)}

            print(f"{name:<20}: R² médio = {mean_r2:.4f} (+/- {np.std(scores):.4f})")

            if mean_r2 >= R2_MINIMO and mean_r2 > best_r2_cv:
                best_r2_cv = mean_r2
                best_model_name = name
                best_model = model
                print(f" -> Modelo selecionado! R² CV ≥ {R2_MINIMO}")
            else:
                print(f" -> Modelo ignorado (R² abaixo do mínimo).")

        except Exception as e:
            print(f"Erro em {name}: {e}")
            results[name] = {'R2_CV_Medio': np.nan, 'R2_CV_Std': np.nan}

    if best_model is None:
        print("Nenhum modelo atingiu o R² mínimo.")
        return None, None, None, None, results

    X_train, X_test, y_train, y_test = train_test_split(X_limpo, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_test = r2_score(y_test, y_pred)

    print(f"\nMelhor modelo: {best_model_name}")
    print(f"R² Teste: {r2_test:.4f}, RMSE: {rmse:.4f}")

    return best_model, best_model_name, r2_test, mse, results


def segment_and_plot(df, max_clusters=10, random_state=42):
    """
    Realiza a segmentação K-Means em colunas numéricas, escolhe o K ótimo
    usando Silhouette Score, e plota os resultados usando PCA.
    """
    print("--- Iniciando Segmentação K-Means ---")
    df_copy = df.copy()

    # 1. Seleciona apenas colunas numéricas (incluindo as dummies e features de data)
    num_cols = df_copy.select_dtypes(include=np.number).columns
    # Remove colunas que podem ser alvo ou IDs (opcional, mas recomendado)
    cols_to_exclude = ['Cluster', 'PC1', 'PC2']

    # Filtra colunas para o clustering
    X = df_copy[[col for col in num_cols if col not in cols_to_exclude]].copy()

    if X_scaled.shape[0] < 2:
        print("Poucas amostras para clustering.")
        return df_copy, 0, None

    # 2. Normaliza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Escolhe k ótimo usando Silhouette Score
    best_score = -1
    best_k = 2
    best_labels = None

    for k in range(2, max_clusters + 1):
        # Garante que k não seja maior que o número de amostras
        if k > X_scaled.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = km.fit_predict(X_scaled)
        # Requer pelo menos 2 clusters
        if k >= 2:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    # Se não houver dados para clustering (ex: apenas 1 linha), retorna
    if best_labels is None:
        print("Segmentação não foi possível (k-means requer pelo menos 2 amostras ou falha na iteração).")
        return df_copy, 0, None


    # 4. Adiciona cluster e treina o modelo final
    df_copy['Cluster'] = best_labels
    print(f"Número ótimo de clusters: {best_k} (Silhouette Score = {best_score:.4f})")

    final_model = KMeans(n_clusters=best_k, random_state=random_state, n_init='auto')
    final_model.fit(X_scaled)

    # 5. PCA para visualizar em 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_copy['PC1'] = components[:,0]
    df_copy['PC2'] = components[:,1]

    # 6. Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_copy, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=80)
    plt.title('Segmentação de Clientes (K-Means + PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.show()

    # 7. Estatísticas por cluster
    print("\nEstatísticas médias por cluster:")
    analysis_cols = [col for col in X.columns] # Usa as colunas do X original
    display(df_copy.groupby('Cluster')[analysis_cols].mean())

    print("-----------------------------------------------------")
    return df_copy, best_k, final_model


# -----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -----------------------------------------------------------------------

from datetime import datetime

def process(df: pd.DataFrame, target_column: str = 'Sales'):
    """
    Função principal que coordena o Pré-processamento, a Regressão (comparação)
    e a Segmentação (K-Means).
    """
    if df is None:
        print("Processo abortado: DataFrame de entrada é None.")
        return None

    print("=====================================================")
    print("============= INICIANDO PROCESSO GERAL ==============")
    print("=====================================================")

    # 1. PRÉ-PROCESSAMENTO
    df_processed = preprocess(df.copy())

    if df_processed is None:
        print("Processo abortado após o pré-processamento.")
        return None

    # 2. REGRESSÃO (Comparação de Modelos)
    if target_column in df_processed.columns:
        best_model, model_name, r2_test, mse_test, history = compare_regression_models(
            df_processed,
            target=target_column
        )
        print(f"Modelo de Regressão final pronto. {model_name} com R² no teste: {r2_test:.4f}")
    else:
        print(f"ERRO: Coluna alvo '{target_column}' não encontrada após o pré-processamento. Regressão ignorada.")
        best_model, model_name, r2_test, mse_test, history = None, None, None, None, None

    # 3. SEGMENTAÇÃO
    df_segmented, optimal_k, kmeans_model = segment_and_plot(df_processed)
    print(f"Análise de Segmentação concluída. K ótimo: {optimal_k}")

    print("\n=====================================================")
    print("================== PROCESSO CONCLUÍDO ===================")
    print("=====================================================")

    # 4. GRAVA RELATÓRIO EM ARQUIVO
    with open(f"reports_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt", "w", encoding="utf-8") as f:
        f.write(
            f"""
Relatório de Desempenho - Supermarket Sales

Modelo escolhido: {model_name}
R² no teste: {r2_test}
MSE: {mse_test}

Número ótimo de clusters: {optimal_k}
"""
        )

    return {
        'df_results': df_segmented,
        'best_regression_model': best_model,
        'best_model_name': model_name,
        'regression_metrics': {
            'R2_Teste': r2_test,
            'MSE_Teste': mse_test,
            'Histórico_CV': history
        },
        'segmentation_model': kmeans_model,
        'optimal_k': optimal_k,
    }

# Uso (requer um DataFrame )
if __name__ == '__main__':

    # Loading dataset
    df = load_dataset('/content/drive/MyDrive/01 - Organização Pessoal/Estudos /Estágio /SuperMarket Analysis.csv')

    # Executando o processo
    results = process(df, target_column='Sales')
