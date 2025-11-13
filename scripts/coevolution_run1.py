import os
import argparse  
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Tuple

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

# --- 模型和 Tokenizer 设置 ---
MODEL_NAME = "./models/gLM2_650M"  # 模型路径
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Token ID 定义 ---
NUC_TOKENS = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
AA_TOKENS = tuple(range(4,24)) # 20 amino acids
ALL_TOKENS = NUC_TOKENS + AA_TOKENS # 合并，方便后续使用
NUM_TOKENS = len(ALL_TOKENS)


# -------------------------
# 核心计算函数
# -------------------------

def jac_to_contact(jac, symm=True, center=True, diag="remove", apc=True):
    X = jac.copy()
    Lx,Ax,Ly,Ay = X.shape

    if center:
        for i in range(4):
            if X.shape[i] > 1:
                X -= X.mean(i,keepdims=True)

    contacts = np.sqrt(np.square(X).sum((1,3)))

    if symm and (Ax != 20 or Ay != 20):
        contacts = (contacts + contacts.T)/2
    if diag == "remove":
        np.fill_diagonal(contacts,0)
    if diag == "normalize":
        contacts_diag = np.diag(contacts)
        contacts = contacts / np.sqrt(contacts_diag[:,None] * contacts_diag[None,:])
    if apc:
        ap = contacts.sum(0,keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
        contacts = contacts - ap
    if diag == "remove":
        np.fill_diagonal(contacts,0)
    return contacts

def contact_to_dataframe(con):
    sequence_length = con.shape[0]
    idx = [str(i) for i in np.arange(1, sequence_length + 1)]
    df = pd.DataFrame(con, index=idx, columns=idx)
    df = df.stack().reset_index()
    df.columns = ['i', 'j', 'value']
    return df

def get_categorical_jacobian(sequence: str, model, tokenizer, fast: bool = False, use_apc: bool = False):
    input_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.int)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    seqlen = input_ids.shape[0]

    is_nuc_pos = torch.isin(input_ids, torch.tensor(NUC_TOKENS)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
    is_nuc_token = torch.isin(torch.tensor(ALL_TOKENS), torch.tensor(NUC_TOKENS)).view(1, -1, 1, 1).repeat(1, 1, 1, NUM_TOKENS)
    is_aa_pos = torch.isin(input_ids, torch.tensor(AA_TOKENS)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
    is_aa_token = torch.isin(torch.tensor(ALL_TOKENS), torch.tensor(AA_TOKENS)).view(1, -1, 1, 1).repeat(1, 1, 1, NUM_TOKENS)

    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        f = lambda x: model(x)[0][..., ALL_TOKENS].cpu().float()
        x = torch.clone(input_ids).to(DEVICE)
        ln = x.shape[1]
        fx = f(x)[0]
        print(fx.shape)
        if fast:
            fx_h = torch.zeros((ln, 1, ln, NUM_TOKENS), dtype=torch.float32)
        else:
            fx_h = torch.zeros((ln, NUM_TOKENS, ln, NUM_TOKENS), dtype=torch.float32)
            x = torch.tile(x, [NUM_TOKENS, 1])
        
        with tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
            for n in range(ln):
                x_h = torch.clone(x)
                if fast:
                    x_h[:, n] = MASK_TOKEN_ID
                else:
                    x_h[:, n] = torch.tensor(ALL_TOKENS)
                fx_h[n] = f(x_h)
                pbar.update(1)
        
        jac = fx_h - fx
        valid_nuc = is_nuc_pos & is_nuc_token
        valid_aa = is_aa_pos & is_aa_token
        jac = torch.where(valid_nuc | valid_aa, jac, 0.0)
        contact = jac_to_contact(jac.numpy(), apc = False)  # apc：调整背景噪声
    return jac, contact, tokens

# -------------------------
#  绘图函数 (使用 Matplotlib)
# -------------------------
def create_figure_matplotlib(contact_df: pd.DataFrame, tokens: List[str], output_filename: str):
    """
    使用 Matplotlib 创建并保存接触图。
    """
    print(f"使用 Matplotlib 创建图表并保存到 {output_filename}...")
    
    seqlen = len(tokens)
    
    # 将长格式的 DataFrame 转回绘图所需的 2D 矩阵
    # 注意：列和索引需要是数字才能正确排序
    contact_df['i'] = pd.to_numeric(contact_df['i'])
    contact_df['j'] = pd.to_numeric(contact_df['j'])
    contact_matrix = contact_df.pivot(index='j', columns='i', values='value').values
    
    # 开始绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用 imshow 绘制热力图
    # im = ax.imshow(contact_matrix, cmap="Blues", vmin=contact_matrix.min(), vmax=np.percentile(contact_matrix, 99))       # 根据矩阵中的值来量化bar
    im = ax.imshow(contact_matrix, cmap="Blues", vmin=0, vmax=1)        # 直接指定bar
    
    # 1. 定义刻度间隔 (例如，每 10 个单位一个刻度)
    step = 10
    
    # 2. 生成刻度的位置
    ticks = np.arange(0, seqlen, step)
    
    # 3. 设置 X 和 Y 轴的刻度
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # 防止标签重叠，旋转X轴标签并设置字体大小
    ax.tick_params(axis='x', labelrotation=60, labelsize=3)
    ax.tick_params(axis='y', labelsize=3)
    # 设置图表样式
    ax.set_title("COEVOLUTION (Contact Map)")
    # ax.set_xticks([]) # 隐藏 X 轴刻度
    # ax.set_yticks([]) # 隐藏 Y 轴刻度
    
    # 添加颜色条
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 保存图表到文件
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close(fig) # 关闭图形，释放内存
    print("图表保存成功。")

def create_subset_heatmap(
    contact_df: pd.DataFrame,
    output_filename: str,
    row_range: Tuple[int, int] = (1, 576),      # 定义子图范围
    col_range: Tuple[int, int] = (578, 1234),       # 定义子图范围
) -> None:
    """
    基于 contact_df 绘制指定子矩阵范围的热力图。

    参数
    ----
    contact_df: pd.DataFrame
        由 contact_to_dataframe 返回的长格式 DataFrame，包含列 ['i', 'j', 'value']。
    output_filename: str
        输出图像文件路径。
    row_range: Tuple[int, int]
        目标矩阵的行区间（包含端点），对应 contact_df 中的 'j'。
    col_range: Tuple[int, int]
        目标矩阵的列区间（包含端点），对应 contact_df 中的 'i'。
    """
    df = contact_df.copy()
    df["i"] = pd.to_numeric(df["i"], errors="coerce")
    df["j"] = pd.to_numeric(df["j"], errors="coerce")
    df = df.dropna(subset=["i", "j"])

    row_min, row_max = row_range
    col_min, col_max = col_range

    if row_min > row_max or col_min > col_max:
        raise ValueError("row_range 或 col_range 设置不合法（起始值不能大于结束值）。")

    mask = (
        df["j"].between(row_min, row_max)
        & df["i"].between(col_min, col_max)
    )
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        raise ValueError(
            "过滤后的 DataFrame 为空，请检查 row_range/col_range 是否正确覆盖原始数据。"
        )

    row_index = np.arange(row_min, row_max + 1)
    col_index = np.arange(col_min, col_max + 1)

    contact_matrix = (
        df_filtered.pivot(index="j", columns="i", values="value")
        .reindex(index=row_index, columns=col_index, fill_value=0.0)
        .values
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        contact_matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_title(
        f"COEVOLUTION (Contact Map)\nRows {row_min}-{row_max} vs Cols {col_min}-{col_max}"
    )
    # ax.set_xlabel("Position (Cols)")
    # ax.set_ylabel("Position (Rows)")
    ax.set_xticks([]) # 隐藏 X 轴刻度
    ax.set_yticks([]) # 隐藏 Y 轴刻度
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"子矩阵热力图保存成功：{output_filename}")

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
MASK_TOKEN_ID = tokenizer.mask_token_id

La_TranC = "MSVSFSLNAKKIRLENYAMKMRLYPSPTQAEQMDKMFLALRLAYNMTFHEVFQQNPAVCGDPDEDGNVWPSYKKMANKTWRKALIDQNPAIAEAPAAAITTNNGLFLSNGQKAWKTGMHNLPANKADRKDFRFYSLSKPRRSFAVQIPPDCIIPSDTNQKVARIKLPKIDGAIKARGFNRKIWFGPDGKHTYEEALAAHELSNNLTVRVSKDTCGDYFICITFSQGKVKGDKPTWEFYQEVRVSPIPEPIGLDVGIKDIAILNTGTKYENKQFKRDRAATLKKMSRQLSRRWGPANSAFRDYNKNIRAENRALEKAQQDPGSSGVGPEAPVLKSVAQPSRRYLTIQKNRAKLERKIARRRDTYYHQVTAEVAGKSSLLAVETLRVKNMLQNHRLAFALSDAAMSDFISKLKYKARRIQVPLVAIGTFQPSSQTCSVCGSINPAVKNLSIRVWTCPNCGTRHNRDINAAKNILAIAQNMLEKKVPFADEALPDEKPPAAPVKKAARKPRDAVFPDHPDLVIRFSKELTQLNDPRYVIVNKATNQIVDNAQGAGYRSAAKAKNCYKAKLAWSSKTNK"

La_array = "GCAGATATCAAATCTCAAAGTGGTGGGAGGTCTGTCCCCACCATGGGGTGCGAACCTTGTGTGCTCATCATTGCCGTGAGCGTTCGCACGTCCAAACGACCATATCATCGTTGCCCTGCGACCATTCAGCGGCAATCAAGACGCAGGCATGATATGTAACCATGCATTTCTGTGAACTGCTTCCAAAACGCACTGCTTCATTATAATCCTCCCTGTGCAGTTCTGCCGAAGCACTTCACAGAAGTGCATGGAGCAGAAGCTCCTATCTTAGGCGCGCTTAATGCGCTTGAGCTGAAGCACTTCACAGAAGTGCATGGAGCAGAAGAGCAATTCGACGGTTTCCCCCTGGATTTTTGGGGGGAAGCACTTCACAGAAGTGCATGGAGCAGAAGATCGCCACGTACCAGCGGGGGGAGATCAATAGTTGAAGCACTTCACAGAAGTGCATGGAGCAGAAGGTAGATGAAGATGCTCCCCGGCATGGTTCTGTCCGAAGCACTTCACAGAAGTGCATGGAGCAGAAGTGCATTTTCCGCCGGATAGATTCCGGGGTGACTGTAGAAGCACTTCACAGAAGTGCATGGAGCAGAAGGGTCTCAAGATGGACTCCCCCTCCGCTGCTGAGGAAGCACTTCACAGAAGTGCATGGAGCAGAAG"

La_array = La_array.lower()
sequence = f"<+>{La_TranC}<+>{La_array}"
J, contact, tokens = get_categorical_jacobian(sequence, fast=True, model=model, tokenizer=tokenizer)
df = contact_to_dataframe(contact)
create_figure_matplotlib(df, tokens, "./TranC_contact_map_fasttest.png")
create_subset_heatmap(df, "./TranC_contact_map_fast_subset2.png", (1, 576), (578, 1234))