import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.coevolution_analysis import get_categorical_jacobian, contact_to_dataframe
from src.plotting import create_figure_matplotlib, create_subset_heatmap


# --- 1. 配置区域 ---
# 将所有可变参数放在这里，方便修改
MODEL_PATH = "/media/Data/pengpai/models/gLM2_650M"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "./results"

# 输入序列
La_TranC = "MSVSFSLNAKKIRLENYAMKMRLYPSPTQAEQMDKMFLALRLAYNMTFHEVFQQNPAVCGDPDEDGNVWPSYKKMANKTWRKALIDQNPAIAEAPAAAITTNNGLFLSNGQKAWKTGMHNLPANKADRKDFRFYSLSKPRRSFAVQIPPDCIIPSDTNQKVARIKLPKIDGAIKARGFNRKIWFGPDGKHTYEEALAAHELSNNLTVRVSKDTCGDYFICITFSQGKVKGDKPTWEFYQEVRVSPIPEPIGLDVGIKDIAILNTGTKYENKQFKRDRAATLKKMSRQLSRRWGPANSAFRDYNKNIRAENRALEKAQQDPGSSGVGPEAPVLKSVAQPSRRYLTIQKNRAKLERKIARRRDTYYHQVTAEVAGKSSLLAVETLRVKNMLQNHRLAFALSDAAMSDFISKLKYKARRIQVPLVAIGTFQPSSQTCSVCGSINPAVKNLSIRVWTCPNCGTRHNRDINAAKNILAIAQNMLEKKVPFADEALPDEKPPAAPVKKAARKPRDAVFPDHPDLVIRFSKELTQLNDPRYVIVNKATNQIVDNAQGAGYRSAAKAKNCYKAKLAWSSKTNK"
La_array = "GCAGATATCAAATCTCAAAGTGGTGGGAGGTCTGTCCCCACCATGGGGTGCGAACCTTGTGTGCTCATCATTGCCGTGAGCGTTCGCACGTCCAAACGACCATATCATCGTTGCCCTGCGACCATTCAGCGGCAATCAAGACGCAGGCATGATATGTAACCATGCATTTCTGTGAACTGCTTCCAAAACGCACTGCTTCATTATAATCCTCCCTGTGCAGTTCTGCCGAAGCACTTCACAGAAGTGCATGGAGCAGAAGCTCCTATCTTAGGCGCGCTTAATGCGCTTGAGCTGAAGCACTTCACAGAAGTGCATGGAGCAGAAGAGCAATTCGACGGTTTCCCCCTGGATTTTTGGGGGGAAGCACTTCACAGAAGTGCATGGAGCAGAAGATCGCCACGTACCAGCGGGGGGAGATCAATAGTTGAAGCACTTCACAGAAGTGCATGGAGCAGAAGGTAGATGAAGATGCTCCCCGGCATGGTTCTGTCCGAAGCACTTCACAGAAGTGCATGGAGCAGAAGTGCATTTTCCGCCGGATAGATTCCGGGGTGACTGTAGAAGCACTTCACAGAAGTGCATGGAGCAGAAGGGTCTCAAGATGGACTCCCCCTCCGCTGCTGAGGAAGCACTTCACAGAAGTGCATGGAGCAGAAG"

def main():
    # --- 2. 准备工作 ---
    print(f"使用的设备: {DEVICE}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型和 Tokenizer
    print("正在加载模型...")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("模型加载完毕。")

    # 准备序列
    combined_sequence = f"<+>{La_TranC}<+>{La_array.lower()}"

    # --- 3. 执行核心分析 ---
    print("开始计算雅可比矩阵和接触图...")
    _, contact, tokens = get_categorical_jacobian(
        sequence=combined_sequence, 
        model=model, 
        tokenizer=tokenizer,
        device=DEVICE,
        fast=True
    )
    df = contact_to_dataframe(contact)
    print("计算完成。")
    
    # --- 4. 生成并保存结果 ---
    # 定义输出文件名
    full_map_path = os.path.join(OUTPUT_DIR, "TranC_contact_map_full.png")
    subset_map_path = os.path.join(OUTPUT_DIR, "TranC_contact_map_subset.png")

    # 绘制并保存全尺寸接触图
    create_figure_matplotlib(df, tokens, full_map_path)

    # 绘制并保存子矩阵热力图
    # 注意：序列长度可能变化，这里的范围需要根据实际情况确认
    # La_TranC 长度为 576，La_array 长度为 657
    # 索引从1开始，所以范围是 蛋白质(1-576) vs 核酸(578-1234)
    create_subset_heatmap(df, subset_map_path, row_range=(1, 576), col_range=(578, 1234))

    print("\n全部分析完成！结果已保存在 'results' 文件夹中。")

if __name__ == "__main__":
    main()