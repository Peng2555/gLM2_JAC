import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def create_figure_matplotlib(contact_df: pd.DataFrame, tokens: List[str], output_filename: str) -> None:
    print(f"使用 Matplotlib 创建图表并保存到 {output_filename}...")
    
    seqlen = len(tokens)
    contact_df['i'] = pd.to_numeric(contact_df['i'])
    contact_df['j'] = pd.to_numeric(contact_df['j'])
    contact_matrix = contact_df.pivot(index='j', columns='i', values='value').values
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(contact_matrix, cmap="Blues", vmin=0, vmax=1)
    
    step = 10
    ticks = np.arange(0, seqlen, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.tick_params(axis='x', labelrotation=60, labelsize=3)
    ax.tick_params(axis='y', labelsize=3)
    ax.set_title("COEVOLUTION (Contact Map)")
    
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("图表保存成功。")

def create_subset_heatmap(
    contact_df: pd.DataFrame,
    output_filename: str,
    row_range: Tuple[int, int] = (1, 576),
    col_range: Tuple[int, int] = (578, 1234),
) -> None:
    df = contact_df.copy()
    df["i"] = pd.to_numeric(df["i"], errors="coerce")
    df["j"] = pd.to_numeric(df["j"], errors="coerce")
    df = df.dropna(subset=["i", "j"])

    row_min, row_max = row_range
    col_min, col_max = col_range

    if row_min > row_max or col_min > col_max:
        raise ValueError("row_range 或 col_range 设置不合法。")

    mask = (df["j"].between(row_min, row_max) & df["i"].between(col_min, col_max))
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        raise ValueError("过滤后的 DataFrame 为空。")

    row_index = np.arange(row_min, row_max + 1)
    col_index = np.arange(col_min, col_max + 1)

    contact_matrix = (
        df_filtered.pivot(index="j", columns="i", values="value")
        .reindex(index=row_index, columns=col_index, fill_value=0.0)
        .values
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(contact_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_title(f"COEVOLUTION (Contact Map)\nRows {row_min}-{row_max} vs Cols {col_min}-{col_max}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"子矩阵热力图保存成功：{output_filename}")