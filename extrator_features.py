"""
extrator_features.py

Script 1: Extração de Features (MFCC)

Este script lê os arquivos de áudio das pastas especificadas,
extrai o vetor médio de MFCCs para cada arquivo e salva
os resultados em um arquivo CSV.

Mapeamento de classes:
- 'sabia': 1
- 'bemtevi': 2
- 'desconhecido': 0
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd

# ======================================
# CONFIGURAÇÕES
# ======================================
PASTA_BASE = r"C:\Users\ninho\Desktop\passaros"
# Mapeamento de pastas para labels numéricos
CATEGORIAS_MAP = {
    "sabia": 1,
    "bemtevi": 2,
    "desconhecido": 0  # Adicione uma pasta "desconhecido" se tiver
}
SR = 22050
N_MFCC = 13
ARQUIVO_SAIDA = r"C:\Users\ninho\Desktop\codigosom\features_passaros.csv"

# ======================================
# FUNÇÕES DE PROCESSAMENTO DE ÁUDIO
# ======================================
def carregar_audio(caminho, sr=SR):
    """Carrega e normaliza o áudio."""
    try:
        y, orig_sr = sf.read(caminho)
    except Exception as e:
        print(f"Erro ao ler {caminho}: {e}")
        return None

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if orig_sr != sr:
        y = librosa.resample(y.astype(float), orig_sr=orig_sr, target_sr=sr)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y


def extrair_mfcc(caminho, sr=SR, n_mfcc=N_MFCC):
    """Extrai os MFCCs e retorna o vetor médio."""
    y = carregar_audio(caminho, sr)
    if y is None:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

# ======================================
# CRIAÇÃO DO DATASET (CSV)
# ======================================
lista_de_features = []

print("Iniciando extração de features...")

for categoria, label in CATEGORIAS_MAP.items():
    pasta = os.path.join(PASTA_BASE, categoria)
    if not os.path.exists(pasta):
        print(f"[AVISO] Pasta não encontrada, pulando: {pasta}")
        continue

    print(f"Processando categoria: {categoria} (Label: {label})")
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith((".wav", ".mp3", ".ogg")):
            caminho = os.path.join(pasta, arquivo)
            vetor_mfcc = extrair_mfcc(caminho)
            
            if vetor_mfcc is not None:
                linha = {f"mfcc_{i}": vetor_mfcc[i] for i in range(N_MFCC)}
                linha["label"] = label
                linha["arquivo_origem"] = arquivo
                lista_de_features.append(linha)

# ======================================
# SALVAR CSV
# ======================================
df_features = pd.DataFrame(lista_de_features)

# Garante que a pasta de destino existe
os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)

df_features.to_csv(ARQUIVO_SAIDA, index=False)

print(f"\nExtração concluída!")
print(f"Total de {len(df_features)} arquivos processados.")
print(f"Features salvas em: {ARQUIVO_SAIDA}")
print("\nDistribuição das classes no CSV:")
print(df_features['label'].value_counts())
