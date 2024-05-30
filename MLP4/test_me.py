import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 미분 방정식 시스템 정의
def dydt(t, y):
    k = 1.0  # k 값을 설정, 예를 들어 1.0
    return [y[1], -k * y[0]]  # [y', u'] = [u, -ky]

# 초기 조건
y0 = [1.0, 0.0]  # y(0) = 1, y'(0) = 0 (초기 위치와 초기 속도)

# 시간 범위: 0에서 10초까지
t_span = (0, 10)

# solve_ivp로 미분 방정식 풀이
sol = solve_ivp(dydt, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 100))

# 결과 그래프 그리기
plt.plot(sol.t, sol.y[0], label='y(t) (Position)')
plt.plot(sol.t, sol.y[1], label='y\'(t) (Velocity)')
plt.title('Harmonic Oscillator Solution')
plt.xlabel('Time (t)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()