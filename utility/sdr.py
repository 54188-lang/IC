import torch
from itertools import permutations


def calc_si_sdr(est_sig: torch.Tensor, ref_sig: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算单个音源对的SI-SDR
    输入:
        est_sig: 估计信号，形状为 (B, T)
        ref_sig: 参考信号，形状为 (B, T)
    输出:
        每个样本的SI-SDR值，形状为 (B,)
    """
    device = est_sig.device
    # 去直流分量
    est_sig = est_sig - torch.mean(est_sig, dim=-1, keepdim=True)
    ref_sig = ref_sig - torch.mean(ref_sig, dim=-1, keepdim=True)
    
    # 计算最优缩放因子
    dot_product = torch.sum(est_sig * ref_sig, dim=-1, keepdim=True)
    ref_power = torch.sum(ref_sig ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / ref_power
    
    # 计算目标分量和误差分量
    target = alpha * ref_sig
    error = est_sig - target
    
    # 计算SI-SDR
    target_power = torch.sum(target ** 2, dim=-1) + eps
    error_power = torch.sum(error ** 2, dim=-1) + eps
    
    return 10 * torch.log10(target_power / error_power).to(device)


def batch_si_sdr(estimation: torch.Tensor,
                 origin: torch.Tensor,
                 eps: float = 1e-8,
                 verbose: bool = True) -> torch.Tensor:
    """
    带调试信息的 PIT-SI-SDR 计算。
    verbose=True 时打印调试信息（仅在训练阶段打开，测试阶段可设为 False）。
    """
    batch_size, num_sources, num_samples = estimation.shape
    assert origin.shape == (batch_size, num_sources, num_samples), "输入形状不匹配"
    device = estimation.device

    # 1. 计算 SI-SDR 矩阵 (B, S, S)
    si_sdr_matrix = torch.zeros((batch_size, num_sources, num_sources), device=device)
    for i in range(num_sources):
        for j in range(num_sources):
            si_sdr_matrix[:, i, j] = calc_si_sdr(estimation[:, i], origin[:, j], eps)

    # 2. 遍历所有排列，记录最优解
    permutations_list = list(permutations(range(num_sources)))
    max_si_sdr = torch.full((batch_size,), fill_value=-1000.0, device=device)
    best_perm_indices = torch.zeros(batch_size, num_sources, dtype=torch.long, device=device)

    for idx, perm in enumerate(permutations_list):
        total = torch.stack([si_sdr_matrix[:, i, perm[i]] for i in range(num_sources)], dim=1).sum(dim=1)
        mask = total > max_si_sdr
        max_si_sdr = torch.maximum(max_si_sdr, total)
        best_perm_indices[mask] = torch.tensor(perm, device=device)

    # # 3. 调试打印：只打印第一条样本即可，避免刷屏
    # if verbose and batch_size > 0:
    #     sample_id = 0  # 只看第 1 条样本
    #     print(f"\n[DEBUG] Sample {sample_id} SI-SDR matrix (rows=est, cols=ref):")
    #     print(si_sdr_matrix[sample_id].cpu().numpy().round(2))

    #     print(f"[DEBUG] Best permutation for sample {sample_id}: "
    #           f"{best_perm_indices[sample_id].cpu().tolist()} "
    #           f"(SI-SDR sum = {max_si_sdr[sample_id].item():.2f})\n")

    # # 4. 统计每个声源在 batch 中被选中的次数（可选）
    # if verbose:
    #     count = torch.zeros(num_sources)
    #     for b in range(batch_size):
    #         for s in range(num_sources):
    #             count[best_perm_indices[b, s]] += 1
    #     print(f"[DEBUG] Batch 中各声源被匹配次数: {count.int().tolist()}")

    return max_si_sdr / num_sources


class negative_SI_SDR(torch.nn.Module):
    """
    负SI-SDR损失函数，用于多音源分离训练
    最小化此损失等价于最大化SI-SDR
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimation: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """
        输入:
            estimation: 模型输出的分离信号，形状为 (B, S, T)
            origin: 参考干净信号，形状为 (B, S, T)
        输出:
            批次平均的negative SI-SDR损失
        """
        si_sdr_values = batch_si_sdr(estimation, origin, self.eps)
        return -torch.mean(si_sdr_values)


