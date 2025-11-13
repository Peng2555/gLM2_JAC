import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

# ==============================================================================
#  主执行脚本
# ==============================================================================

if __name__ == "__main__":
    # --- 1. 加载模型和分词器 ---
    print("Loading gLM2 model and tokenizer...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "./models/gLM2_650M"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"Model loaded to {DEVICE}.")
    print("-" * 30)

    # --- 2. 准备一个简短的混合模态输入序列 ---
    # 这是一个虚构的序列，包含蛋白质部分 ("MKT") 和 核酸部分 ("ATCG")
    sequence = "<+>MKT<->ATCG"
    print(f"Input Sequence: {sequence}\n")

    # --- 3. 编码序列并送入模型 ---
    # `return_tensors="pt"` 会直接返回 PyTorch 张量
    inputs = tokenizer(sequence, return_tensors="pt").to(DEVICE)
    
    # 关闭梯度计算，以节省内存并加速
    with torch.no_grad():
        # `output_hidden_states=True` 是获取所有层嵌入的关键
        outputs = model(**inputs, output_hidden_states=True)

    print("-" * 30)
    
    # ==============================================================================
    #  核心输出 1: 嵌入 (Embeddings)
    # ==============================================================================
    print("Inspecting Embeddings (Hidden States)...\n")

    # `outputs.hidden_states` 是一个元组，包含了所有层的隐藏状态
    # 第 0 个是初始的词嵌入，最后一个 (索引-1) 是我们通常最关心的最后一层输出
    last_hidden_states = outputs.hidden_states[-1]
    
    # 移除批次维度 (因为我们只输入了一个序列)
    embeddings = last_hidden_states.squeeze(0) # squeeze(0) 移除第0个维度
    
    # 将结果从 GPU 移到 CPU 以便后续操作
    embeddings = embeddings.cpu().numpy()
    
    # 解码输入，看看每个令牌是什么
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    
    print(f"Shape of embeddings matrix: {embeddings.shape}")
    print(f" (Sequence Length, Embedding Dimension)")
    print(f"Sequence Length is {len(tokens)} tokens: {tokens}\n")
    
    # 让我们看看第一个令牌 (<cls>) 和 'M' 的嵌入向量
    print(f"Embedding vector for the first token ('{tokens[0]}'):")
    print(f"  {embeddings[0][:10]} ... (showing first 10 dimensions of 1280)")
    
    print(f"Embedding vector for the third token ('{tokens[2]}'):")
    print(f"  {embeddings[2][:10]} ... (showing first 10 dimensions of 1280)")
    
    # 计算整个序列的表示 (常用于分类任务)
    sequence_embedding = np.mean(embeddings, axis=0)
    print(f"\nMean embedding for the entire sequence (shape: {sequence_embedding.shape}):")
    print(f"  {sequence_embedding[:10]} ...")

    print("-" * 30)

    # ==============================================================================
    #  核心输出 2: 预测分数 (Logits)
    # ==============================================================================
    print("Inspecting Prediction Scores (Logits)...\n")
    
    # `outputs.logits` 直接给出了预测分数
    logits = outputs.logits.squeeze(0).cpu() # 同样移除批次维度并移到 CPU
    
    print(f"Shape of logits matrix: {logits.shape}")
    print(f" (Sequence Length, Vocabulary Size)")
    print(f"Vocabulary Size is {logits.shape[1]}\n")
    
    # 让我们来玩一个“完形填空”的游戏
    # 找出在第二个位置 ('K')，模型认为最有可能的令牌是什么
    position_to_predict = 2 # The token is 'K'
    
    # 获取该位置的所有预测分数
    scores_at_position = logits[position_to_predict]
    
    # 找出分数最高的令牌的 ID
    predicted_token_id = torch.argmax(scores_at_position).item()
    
    # 将 ID 转换回令牌
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
    
    print(f"At position {position_to_predict} (original token is '{tokens[position_to_predict]}'):")
    print(f"The model's top prediction is: '{predicted_token}'")
    
    # 我们还可以看看前5名的预测
    top_5_indices = torch.topk(scores_at_position, 5).indices
    top_5_tokens = tokenizer.convert_ids_to_tokens(top_5_indices)
    top_5_scores = scores_at_position[top_5_indices].numpy()

    print("\nTop 5 predictions for that position:")
    for token, score in zip(top_5_tokens, top_5_scores):
        print(f"  - Token: {token:<5} | Logit Score: {score:.2f}")