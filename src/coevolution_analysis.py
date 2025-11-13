import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- Token ID 定义 ---
NUC_TOKENS = tuple(range(29, 33)) # 4 nucleotides a,t,c,g
AA_TOKENS = tuple(range(4,24)) # 20 amino acids
ALL_TOKENS = NUC_TOKENS + AA_TOKENS # 合并，方便后续使用
NUM_TOKENS = len(ALL_TOKENS)

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

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

def get_categorical_jacobian(sequence: str, model, tokenizer, device, fast: bool = False, use_apc: bool = False):
    MASK_TOKEN_ID = tokenizer.mask_token_id
    input_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.int)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    seqlen = input_ids.shape[0]

    # ... (函数其余部分完全不变) ...
    is_nuc_pos = torch.isin(input_ids, torch.tensor(NUC_TOKENS)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
    is_nuc_token = torch.isin(torch.tensor(ALL_TOKENS), torch.tensor(NUC_TOKENS)).view(1, -1, 1, 1).repeat(1, 1, 1, NUM_TOKENS)
    is_aa_pos = torch.isin(input_ids, torch.tensor(AA_TOKENS)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
    is_aa_token = torch.isin(torch.tensor(ALL_TOKENS), torch.tensor(AA_TOKENS)).view(1, -1, 1, 1).repeat(1, 1, 1, NUM_TOKENS)

    input_ids = input_ids.unsqueeze(0).to(device) # 使用传入的device

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        f = lambda x: model(x)[0][..., ALL_TOKENS].cpu().float()
        x = torch.clone(input_ids).to(device)
        ln = x.shape[1]
        fx = f(x)[0]
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
        
        # 规则：寻找 AA -> NUC 和 NUC -> AA 的交互

        # 规则1: 定义“氨基酸影响核酸”的交互
        # 源头是氨基酸位置 & 目标是核苷酸位置
        valid_aa_to_nuc_pos = is_aa_pos & is_nuc_pos.transpose(0, 2)
        # 源头是氨基酸令牌 & 目标是核苷酸令牌
        valid_aa_to_nuc_token = is_aa_token & is_nuc_token.transpose(1, 3)
        valid_aa_to_nuc = valid_aa_to_nuc_pos & valid_aa_to_nuc_token

        # 规则2: 定义“核酸影响氨基酸”的交互
        # 源头是核苷酸位置 & 目标是氨基酸位置
        valid_nuc_to_aa_pos = is_nuc_pos & is_aa_pos.transpose(0, 2)
        # 源头是核苷酸令牌 & 目标是氨基酸令牌
        valid_nuc_to_aa_token = is_nuc_token & is_aa_token.transpose(1, 3)
        valid_nuc_to_aa = valid_nuc_to_aa_pos & valid_nuc_to_aa_token

        # 最终的跨模态规则：满足规则3或规则4即可
        valid_cross_modal_mask = valid_aa_to_nuc | valid_nuc_to_aa
        
        # 应用新的掩码，只保留跨模态信号
        jac = torch.where(valid_cross_modal_mask, jac, 0.0)
        contact = jac_to_contact(jac.numpy(), apc=True)
    return jac, contact, tokens

