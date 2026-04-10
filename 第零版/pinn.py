import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 1. 定义 PINN 模型 -----------------
class FLakePINN(nn.Module):
    def __init__(self):
        super(FLakePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Softplus(),
            nn.Linear(50, 50),
            nn.Softplus(),
            nn.Linear(50, 1)
        )
        self.eta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    def forward(self, t, z):
        inputs = torch.cat([t, z], dim=1)
        return self.net(inputs)

# ----------------- 2. Loss 函数 -----------------
def compute_losses(model, t, z, I_s, t_obs, z_obs, T_obs, z_bottom):
    t.requires_grad_(True)
    z.requires_grad_(True)
    T = model(t, z)
    dT_dt = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dT_dz = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z, grad_outputs=torch.ones_like(dT_dz), create_graph=True)[0]

    K_H = 5e-7
    rho_cp = 4.18e6
    alpha = 0.07
    eta = model.eta

    dI_dz = -I_s * (1 - alpha) * eta * torch.exp(-eta * z)
    heating_term = -(1 / rho_cp) * dI_dz

    pde_residual = dT_dt - K_H * d2T_dz2 - heating_term
    L_PDE = torch.mean(pde_residual ** 2)

    T_bottom = model(t, z_bottom)
    dT_dz_bottom = torch.autograd.grad(
        T_bottom, z_bottom,
        grad_outputs=torch.ones_like(T_bottom),
        create_graph=True
    )[0]
    L_bc = torch.mean(dT_dz_bottom ** 2)

    T_pred_obs = model(t_obs, z_obs)
    L_data = torch.mean((T_pred_obs - T_obs) ** 2)

    return L_PDE, L_bc, L_data

# ----------------- 3. 训练函数 -----------------
def train():
    # 读取 ERA5 驱动
    df = pd.read_csv(r'E:\pycharm\ERA5_extracted_temp\ERA5_Alakol_Daily_2025_01.csv')
    df.columns = df.columns.str.strip()  # 去掉前后空格
    N_days = len(df)
    t_val = torch.tensor(np.arange(N_days).reshape(-1,1), dtype=torch.float32)
    Is_val = torch.tensor(df['Is_J_per_m2'].values.reshape(-1,1), dtype=torch.float32)

    # 读取真实观测 LST CSV
    lst_path = r'E:\pycharm\ERA5_extracted_temp\Alakol_LST_2025_01.csv'  # 直接指定文件
    df_obs = pd.read_csv(lst_path)
    df_obs.columns = df_obs.columns.str.strip()
    valid_obs = df_obs.dropna(subset=['LST_C'])
    t_obs_idx = valid_obs.index.values
    T_obs_real = valid_obs['LST_C'].values

    # 转换为张量
    t_obs_tensor = torch.tensor(t_obs_idx.reshape(-1,1), dtype=torch.float32)
    T_surface_tensor = torch.tensor(T_obs_real.reshape(-1,1), dtype=torch.float32)
    z_surface = torch.zeros((len(t_obs_idx),1), dtype=torch.float32)

    # 初始化模型和优化器
    model = FLakePINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lambda_data = 1.0
    lambda_pde = 0.1
    lambda_bc = 0.001
    epochs = 2000

    print(f"🚀 开始训练 PINN，共 {epochs} 轮...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        t_colloc = torch.rand((500,1)) * (N_days-1)
        z_colloc = torch.rand((500,1)) * 50.0
        z_bottom_colloc = torch.full((500,1), 50.0, dtype=torch.float32, requires_grad=True)
        t_index = torch.clamp(torch.round(t_colloc).long(), 0, N_days-1)
        Is_colloc = Is_val[t_index.squeeze()].reshape(-1,1)
        L_PDE, L_bc, L_data = compute_losses(
            model, t_colloc, z_colloc, Is_colloc,
            t_obs_tensor, z_surface, T_surface_tensor, z_bottom_colloc
        )
        Total_Loss = lambda_data * L_data + lambda_pde * L_PDE + lambda_bc * L_bc
        Total_Loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {Total_Loss.item():.4f} | "
                  f"L_data: {L_data.item():.4f} | L_PDE: {L_PDE.item():.4f} | "
                  f"L_bc: {L_bc.item():.4f} | eta: {model.eta.item():.4f}")

    print("🎉 训练完成！")
    return model, N_days

# ----------------- 4. 绘制全月热图 -----------------
def plot_full_month_heatmap(model, N_days, max_depth=50.0, n_depth_points=100):
    model.eval()
    z = torch.linspace(0,max_depth,n_depth_points).reshape(-1,1)
    T_month = []
    for day in range(N_days):
        t_day = torch.full_like(z, float(day))
        with torch.no_grad():
            T_day = model(t_day, z).numpy().flatten()
        T_month.append(T_day)
    T_month = np.array(T_month).T
    plt.figure(figsize=(12,6))
    im = plt.imshow(T_month, aspect='auto', origin='upper',
                    extent=[0,N_days-1,0,max_depth],
                    cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar(im, label='Temperature (°C)')
    plt.xlabel('Day')
    plt.ylabel('Depth (m)')
    plt.title('Alakol Lake Temperature Profile (PINN)')
    plt.show()

# ----------------- 5. 主程序 -----------------
if __name__ == "__main__":
    model, N_days = train()
    plot_full_month_heatmap(model, N_days)