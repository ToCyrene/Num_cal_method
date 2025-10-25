import random
from typing import Callable, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

#说明：
# 本代码实现了多种一维插值方法，并对它们在不同参数设置下的表现进行了比较。
# 目标函数为 f(x) = c*sin(d*x) + e*cos(f*x)，可调节参数 c, d, e, f 控制函数形态。
# 区间边界 a、b、采样节点数 n、测试点数 m、噪声水平 noise、随机数种子 seed均可调节。
# 代码包含以下插值方法：
    # 1. 范德蒙插值
    # 2. 拉格朗日插值 
    # 3. 牛顿插值
    # 4. 差分牛顿插值（均匀采样）
    # 5. 分段线性插值
    # 6. 分段三次埃尔米特插值 
# 采样点使用等距节点，以适配差分牛顿插值的要求。
# 三次埃尔米特插值可选是否为导数添加高斯噪声。
# 代码未使用除高斯消元法外的任何线性代数库函数，所有插值算法均手动实现。

#实验参数列表
param_list = [
        #无噪声，小样本数量
        dict(a=0, b=10, c=1.0, d=2.0, e=0.5, f=3.0, n=10, m=200, noise=0.0, seed=42),
        #有噪声，更密集采样
        dict(a=0, b=10, c=1.0, d=2.0, e=0.5, f=3.0, n=20, m=400, noise=1e-2, seed=42),
        #不同频率参数
        dict(a=0, b=10, c=1.0, d=3.0, e=0.5, f=5.0, n=15, m=300, noise=0.0, seed=42),
        #高频振荡，测试欠拟合
        dict(a=0, b=10, c=1.0, d=8.0, e=0.5, f=6.0, n=12, m=400, noise=0.0, seed=42),
        #大噪声，稀疏采样
        dict(a=0, b=10, c=1.0, d=2.0, e=0.5, f=3.0, n=8, m=200, noise=0.1, seed=42),
        #不对称区间 [-5,5]
        dict(a=-5, b=5, c=1.0, d=2.0, e=0.5, f=3.0, n=15, m=300, noise=0.0, seed=42),
        #小振幅，高噪声
        dict(a=0, b=10, c=0.2, d=2.0, e=0.1, f=3.0, n=15, m=300, noise=0.05, seed=42),
        #纯正弦函数
        dict(a=0, b=10, c=1.0, d=2.0, e=0.0, f=0.0, n=12, m=300, noise=0.01, seed=42),
        # 非常密集的采样，无噪声
        dict(a=0, b=10, c=1.0, d=2.0, e=0.5, f=3.0, n=50, m=500, noise=0.0, seed=42),
        #大区间，低频
        dict(a=0, b=20, c=1.0, d=0.5, e=0.5, f=1.0, n=25, m=500, noise=0.0, seed=42),
    ]

#----------------------------------
# 插值函数起始
#----------------------------------
#所有插值函数均接受两个numpy数组作为输入：x_samples（采样节点），y_samples（对应的函数值）
#需注意：三次埃尔米特插值还需要额外的导数值数组 y_deriv
#所有插值函数返回的对象均为一个函数 l(x)，该函数接受标量或numpy数组 x，返回插值结果

#范德蒙插值
#过程
# 1. 构造范德蒙矩阵 V
# 2. 使用高斯消元法求解线性方程 V * a = y_samples 得到系数 a
# 3. 构造插值多项式 l(x) = a[0] + a[1]*x + a[2]*x^2 + ... + a[n-1]*x^(n-1)      
# 高斯消元法使用 numpy.linalg.solve 实现
def vandermonde_interp(x_samples: np.ndarray, y_samples: np.ndarray):
    V = np.vander(x_samples, N=len(x_samples), increasing=True)
    a = np.linalg.solve(V, y_samples)

    def l(x):
        x = np.asarray(x)
        res = np.zeros_like(x, dtype=float)
        for i in range(len(a)):
            res += a[i] * x**i
        return res
    return l

#拉格朗日插值
#过程
# 1. 构造拉格朗日基函数 L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
# 2. 构造插值多项式 l(x) = Σ y_i * L_i(x)
def lagrange_interp(x_samples: np.ndarray, y_samples: np.ndarray):
    def l(x):
        x = np.asarray(x)
        res = np.zeros_like(x, dtype=float)
        for i in range(len(x_samples)):
            tmp = np.ones_like(x, dtype=float)
            for j in range(len(x_samples)):
                if i != j:
                    tmp *= (x - x_samples[j]) / (x_samples[i] - x_samples[j])
            res += y_samples[i] * tmp
        return res
    return l

#牛顿插值
#过程
# 1. 构造差商表 f[i][j]，其中 f[0][i] = y_samples[i]，f[i][j] = (f[i-1][j+1] - f[i-1][j]) / (x_samples[j+i] - x_samples[j])
# 2. 构造插值多项式 l(x) = f[0][0] + f[1][0]*(x - x_0) + f[2][0]*(x - x_0)*(x - x_1) + ... + f[n-1][0]*(x - x_0)*...*(x - x_{n-2})
# 节点不需要等距
def newton_interp(x_samples: np.ndarray, y_samples: np.ndarray):
    n = len(x_samples)
    f = [[0.0] * n for _ in range(n)]
    for i in range(n):
        f[0][i] = y_samples[i]
    for i in range(1, n):
        for j in range(n - i):
            f[i][j] = (f[i - 1][j + 1] - f[i - 1][j]) / (x_samples[j + i] - x_samples[j])

    def l(x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for i in range(n):
            term = f[i][0] * np.ones_like(x, dtype=float)
            for j in range(i):
                term *= (x - x_samples[j])
            result += term
        return result
    return l

#差分牛顿插值（等距节点）
#过程
# 1. 构造差分表 Δ^k y[i]，其中 Δ^0 y[i] = y_samples[i]，Δ^k y[i] = Δ^(k-1) y[i+1] - Δ^(k-1) y[i]
# 2. 构造插值多项式 l(x) = y_0 + (t)*Δ^1 y[0] + (t*(t-1)/2!)*Δ^2 y[0] + ... + (t*(t-1)*...*(t-n+1)/n!)*Δ^n y[0]
# 其中 t = (x - x_0) / h，h 为节点间距
# 节点必须等距
def diff_newton_uniform_interp(x_samples: np.ndarray, y_samples: np.ndarray):
    n = len(x_samples)
    h = x_samples[1] - x_samples[0]
    for i in range(2, n):
        if abs((x_samples[i] - x_samples[i - 1]) - h) > 1e-10: #浮点数比较不能直接用==，要考虑误差
            raise ValueError("The nodes are not equidistant, so Newton's divided difference interpolation cannot be used")

    diff_table = np.zeros((n, n))
    for i in range(n):
        diff_table[0, i] = y_samples[i]
    for i in range(1, n):
        for j in range(n - i):
            diff_table[i, j] = diff_table[i - 1, j + 1] - diff_table[i - 1, j]

    def l(x):
        x_arr = np.asarray(x, dtype=float)
        result = np.zeros_like(x_arr)
        t = (x_arr - x_samples[0]) / h
        result[:] = diff_table[0, 0]
        term = np.ones_like(t)
        for i in range(1, n):
            term *= (t - (i - 1)) / i
            result += term * diff_table[i, 0]
        return result
    return l

#分段线性插值
#过程
# 1. 对于每个区间 [x_i, x_{i+1}]，构造线性函数 L_i(x) = y_i + (y_{i+1} - y_i)/(x_{i+1} - x_i) * (x - x_i)
# 2. 插值函数 l(x) 在每个区间内使用对应的线性函数
def piecewise_linear_interp(x_samples: np.ndarray, y_samples: np.ndarray):
    n = len(x_samples)
    indices = np.argsort(x_samples)
    x_sorted = x_samples[indices]
    y_sorted = y_samples[indices]

    def l(x):
        x_arr = np.asarray(x, dtype=float)
        indices = np.searchsorted(x_sorted, x_arr) - 1
        indices = np.clip(indices, 0, n - 2)
        x0 = x_sorted[indices]
        x1 = x_sorted[indices + 1]
        y0 = y_sorted[indices]
        y1 = y_sorted[indices + 1]
        t = (x_arr - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    return l


#三次埃尔米特插值
#过程
# 1. 对于每个区间 [x_i, x_{i+1}]，构造三次埃尔米特多项式
#    H_i(x) = h00(t)*y_i + h10(t)*h*m_i + h01(t)*y_{i+1} + h11(t)*h*m_{i+1}
#    其中 t = (x - x_i)/h，h = x_{i+1} - x_i
#    h00(t) = 2t^3 - 3t^2 + 1
#    h10(t) = t^3 - 2t^2 + t 
#    h01(t) = -2t^3 + 3t^2
#    h11(t) = t^3 - t^2
def cubic_hermite_interp(x_samples: np.ndarray, y_samples: np.ndarray, y_deriv: np.ndarray):
    n = len(x_samples)
    x_sorted = np.asarray(x_samples)
    y_sorted = np.asarray(y_samples)
    dy_sorted = np.asarray(y_deriv)

    def l(x):
        x_arr = np.asarray(x, dtype=float)
        res = np.zeros_like(x_arr)
        indices = np.searchsorted(x_sorted, x_arr) - 1
        indices = np.clip(indices, 0, n - 2)
        x0 = x_sorted[indices]
        x1 = x_sorted[indices + 1]
        y0 = y_sorted[indices]
        y1 = y_sorted[indices + 1]
        m0 = dy_sorted[indices]
        m1 = dy_sorted[indices + 1]
        h = x1 - x0
        t = (x_arr - x0) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    return l

#----------------------------------
# 插值函数结束
#----------------------------------

#目标函数
def target_function_generator(c: float, d: float, e: float, f: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: c * np.sin(d * x) + e * np.cos(f * x)

#目标函数导数
def target_function_derivative_generator(c: float, d: float, e: float, f: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: c * d * np.cos(d * x) - e * f * np.sin(f * x)

#生成采样节点的函数
def sample_nodes(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, n + 1)

#为数值添加噪声的函数
def add_noise(y: np.ndarray, noise_level: float) -> np.ndarray:
    if noise_level <= 0:
        return y.copy()
    return y + np.random.normal(0.0, noise_level, size=y.shape)

#计算平均绝对误差
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

#执行单次实验
#过程
# 1. 根据参数生成目标函数和其导数
# 2. 采样节点并添加噪声
# 3. 在测试点上计算真实值
# 4. 对每种插值方法进行插值并计算预测值
def run_single_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    a, b, c, d, e, f_ = params["a"], params["b"], params["c"], params["d"], params["e"], params["f"]
    n, m, noise = params["n"], params["m"], params["noise"]
    seed = params.get("seed", None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    f_target = target_function_generator(c, d, e, f_)
    f_deriv = target_function_derivative_generator(c, d, e, f_)

    x_nodes = sample_nodes(a, b, n)
    y_nodes = add_noise(f_target(x_nodes), noise)
    y_deriv_exact = f_deriv(x_nodes)
    y_deriv = add_noise(y_deriv_exact, noise) #对导数添加噪声

    x_test = np.linspace(a, b, m)
    y_true = f_target(x_test)

    methods = [
        ("Vandermonde", vandermonde_interp),
        ("Lagrange", lagrange_interp),
        ("Newton", newton_interp),
        ("DiffNewtonUniform", diff_newton_uniform_interp),
        ("PiecewiseLinear", piecewise_linear_interp),
        ("CubicHermite", lambda x, y: cubic_hermite_interp(x, y, y_deriv_exact)), #使用精确的导数
        #("CubicHermite", lambda x, y: cubic_hermite_interp(x, y, y_deriv)), #使用带噪声的导数
    ]

    results = {}
    preds = {}

    for name, fn in methods:
        try:
            interp_fn = fn(x_nodes, y_nodes)
            y_pred = interp_fn(x_test)
            preds[name] = y_pred
            results[name] = mean_absolute_error(y_true, y_pred)
        except NotImplementedError:
            results[name] = None
            preds[name] = None
        except Exception as ex:
            print(f"[Warning] Method {name} failed: {ex}")
            results[name] = None
            preds[name] = None

    return dict(
        params=params,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        x_test=x_test,
        y_true=y_true,
        preds=preds,
        errors=results,
    )

#执行多次实验
def run_multiple_experiments(param_list: List[Dict[str, Any]], plot_each=True):
    all_results = []
    for i, p in enumerate(param_list):
        print(f"\n=== Experiment {i+1}/{len(param_list)} Parameters ===")
        print(p)
        res = run_single_experiment(p)
        all_results.append(res)
        print("Mean Absolute Errors:")
        for k, v in res["errors"].items():
            print(f"  {k:20s} : {v}")
        if plot_each:
            plot_results(res, title_suffix=f"#{i+1}")

    if len(all_results) > 1:
        plot_error_summary(all_results)
    return all_results

#绘图函数
def plot_results(res: Dict[str, Any], title_suffix=""):
    x_test, y_true = res["x_test"], res["y_true"]
    x_nodes, y_nodes = res["x_nodes"], res["y_nodes"]
    preds = res["preds"]

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_true, label="Target Function", lw=2)
    plt.scatter(x_nodes, y_nodes, c="black", s=30, label="Sample Points")

    for name, y_pred in preds.items():
        if y_pred is not None:
            plt.plot(x_test, y_pred, label=name)

    plt.legend()
    plt.title(f"Interpolation Comparison {title_suffix}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

#平均绝对误差汇总图
def plot_error_summary(results: List[Dict[str, Any]]):
    methods = list(results[0]["errors"].keys())
    n_exp = len(results)
    error_matrix = np.zeros((n_exp, len(methods)))

    for i, res in enumerate(results):
        for j, name in enumerate(methods):
            error_matrix[i, j] = res["errors"][name] if res["errors"][name] is not None else np.nan

    plt.figure(figsize=(10, 5))
    for j, name in enumerate(methods):
        plt.plot(range(n_exp), error_matrix[:, j], marker="o", label=name)

    plt.xticks(range(n_exp), [f"#{i+1}" for i in range(n_exp)])
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Experiment Index")
    plt.legend()
    plt.title("Error Comparison Across Experiments")
    plt.grid(True)
    plt.yscale("log") 
    plt.show()

def main():
    run_multiple_experiments(param_list, plot_each=True)

if __name__ == "__main__":
    main()
