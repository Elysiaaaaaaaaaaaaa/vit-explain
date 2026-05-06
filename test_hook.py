import torch
import torch.nn as nn

# 简单模型：包含 Dropout 层（p=0.0）
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.0)
    
    def forward(self, x):
        return self.dropout(x)

# 测试钩子是否触发
model = TestModel()
hook_triggered = False

def hook_fn(module, input, output):
    global hook_triggered
    hook_triggered = True
    print(f"钩子触发！输入: {input[0].shape}, 输出: {output.shape}")

model.dropout.register_forward_hook(hook_fn)

# 测试 train 和 eval 模式
print("=== Train 模式 ===")
model.train()
model(torch.randn(1, 3, 3))
print(f"钩子是否触发: {hook_triggered}")

print("\n=== Eval 模式 ===")
hook_triggered = False
model.eval()
model(torch.randn(1, 3, 3))
print(f"钩子是否触发: {hook_triggered}")