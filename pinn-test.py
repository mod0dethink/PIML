import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==== 真の初期条件関数 ====
# 実際に使った真の初期条件 u(x, 0) = sin(pi * x)
def true_u0(x):
    return np.sin(np.pi * x)

# ==== データ生成 ====
alpha = 0.1  # 熱伝導率（物理パラメータ）
x = np.linspace(0, 1, 100)  # 空間方向の点（0〜1を100分割）
t = np.linspace(0, 1, 100)  # 時間方向の点（0〜1を100分割）
X, T = np.meshgrid(x, t)  # 格子状データ（x,t）を生成

# PDEの解析解（順問題の正解データ）
# u(x, t) = exp(-pi^2 * alpha * t) * sin(pi * x)
U_true = np.exp(-np.pi**2 * alpha * T) * np.sin(np.pi * X)

# 1次元の訓練データに変換（Flatten）
x_train = X.flatten().reshape(-1, 1)
t_train = T.flatten().reshape(-1, 1)
u_train = U_true.flatten().reshape(-1, 1)

# ==== t > 0 の観測データのみ使用（t=0は推定対象）====
mask = t_train[:, 0] > 0.0  # t=0を除外
x_obs = torch.tensor(x_train[mask], dtype=torch.float32)
t_obs = torch.tensor(t_train[mask], dtype=torch.float32)
u_obs = torch.tensor(u_train[mask], dtype=torch.float32)

# ==== PINN モデル定義 ====
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),        # 入力層 → 中間層1
            nn.Linear(64, 64), nn.Tanh(),       # 中間層2
            nn.Linear(64, 1)                    # 出力層（uの予測）
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))  # xとtを結合して入力

# ==== PDEの残差（物理損失）を定義 ====
def pde_residual(model, x, t, alpha):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]   # ∂u/∂t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]   # ∂u/∂x
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]  # ∂²u/∂x²

    return u_t - alpha * u_xx  # PDEの残差（0に近いほど良い）

# ==== モデル学習 ====
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5000  # 学習エポック数

for epoch in range(epochs):
    optimizer.zero_grad()

    # データ損失：観測値との誤差
    u_pred = model(x_obs, t_obs)
    loss_data = nn.MSELoss()(u_pred, u_obs)

    # 物理損失：PDEの残差が小さくなるように
    x_phys = torch.rand(2000, 1)  # ランダムな空間点
    t_phys = torch.rand(2000, 1)  # ランダムな時間点
    f = pde_residual(model, x_phys, t_phys, alpha)
    loss_phys = torch.mean(f**2)

    # 総合損失：データ損失 + 物理損失
    loss = loss_data + loss_phys
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# ==== 初期条件の推定 ====
x_plot = torch.linspace(0, 1, 200).view(-1, 1)  # x座標（200点）
t0 = torch.zeros_like(x_plot)  # t=0 固定（初期状態）

with torch.no_grad():
    u0_pred = model(x_plot, t0).squeeze().numpy()  # 推定された u(x, 0)

# ==== 結果の描画 ====
plt.figure(figsize=(8, 4))
plt.plot(x_plot.numpy(), u0_pred, label="Predicted Initial Condition")  # 推定された初期条件
plt.xlabel("x")
plt.ylabel("u(x, t=0)")
plt.title("PIMLで推定された初期条件")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("predicted_initial.png", dpi=300)  # 結果を保存
plt.show()
