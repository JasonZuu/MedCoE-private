from torch import nn
from peft import LoraConfig, get_peft_model


def set_lora_on_layers(
    model: nn.Module,
    layer_idx: list[int] | None = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    train_kv_only: bool = False
) -> nn.Module:
    """
    对已加载的 HF 模型，在指定的 Transformer 层注入 LoRA，并冻结其余所有参数。
    - layer_idx 为 None 时，默认在所有层注入 LoRA；否则只在列表中指定的层注入。
    - train_kv_only=True 时，仅对 K、V 投影注入 LoRA；否则注入该层所有 Linear。

    Args:
        model: 已加载的 transformers 模型实例（如 Qwen2Model）。
        layer_idx: 要注入 LoRA 的层索引列表（从 0 开始），默认为 None（所有层）。
        r: LoRA 内部秩。
        alpha: LoRA alpha 超参。
        dropout: LoRA dropout。
        train_kv_only: 是否仅注入 K 和 V 投影。

    Returns:
        包装了 LoRA、并只训练 LoRA 参数的 PEFT 模型。
    """
    # 定位 Transformer 层列表（ModuleList）
    layers = None
    prefix = None
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            layers, prefix = child, name
            break
    if layers is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                layers, prefix = module, name
                break
    if layers is None:
        raise ValueError("无法定位 Transformer 层，请检查模型结构。")

    num_layers = len(layers)
    # 计算要注入的层索引
    if layer_idx is None:
        target_idxs = list(range(num_layers))
    else:
        # 支持负数索引
        target_idxs = [i if i >= 0 else num_layers + i for i in layer_idx]

    # 构造 target_modules
    target_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # 提取该 Linear 所属层的索引前缀
        for idx in target_idxs:
            layer_prefix = f"{prefix}.{idx}"
            if name.startswith(layer_prefix):
                if train_kv_only:
                    # 只注入 k_proj 和 v_proj
                    if "k_proj" in name or "v_proj" in name:
                        target_modules.append(name)
                else:
                    target_modules.append(name)
                break

    if not target_modules:
        raise ValueError("未找到符合条件的 Linear 子模块，请检查 layer_idx 或模型结构。")

    # 配置 LoRA
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )

    # 包装模型
    peft_model = get_peft_model(model, lora_cfg)

    # 打印可训练参数占比
    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Trainable params after adding LoRA: {trainable}/{total} = {100 * trainable / total:.4f}%")

    return peft_model
