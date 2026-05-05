# Phân tích & Phê bình Thiết kế Hệ thống 2-Tầng Drone

---

## Tổng quan cấu trúc file

```
myAviary.py          ← Lớp nền vật lý dùng chung (không phải Env hoàn chỉnh)
env_highlevel.py     ← High-level GNN Env (CTDE) + Multi-Agent wrapper
env_lowlevel.py      ← Low-level DQN Env (single agent, discrete actions)
model_gnn.py         ← GNN Policy (GAT) cho high-level
train.py             ← Script train cả hai tầng
```

---

## PHẦN 1: Những điểm tốt trong thiết kế của cậu

### ✅ Phân tầng high/low rõ ràng
Đây là kiến trúc **hierarchical RL** chuẩn. High-level giải quyết bài toán
topology mạng (thời gian dài, tần suất thấp), low-level lo chuyển động (thời
gian ngắn, tần suất cao). Tách ra như vậy giúp mỗi tầng học được bài toán vừa
đủ phức tạp.

### ✅ Ý tưởng dùng cosine similarity làm reward signal
Rất tốt — cosine reward có **gradient mượt** hơn sparse reward (chỉ thưởng khi
tới đích). Nó hướng dẫn agent ngay từ bước đầu thay vì phải khám phá mù quáng.

### ✅ Dùng GNN để encode đồ thị
Đúng tool cho đúng việc. Đồ thị drone thay đổi topology theo thời gian (drone
vào/ra comm radius) → GNN xử lý tốt hơn MLP cố định.

### ✅ Ý tưởng CTDE (Centralized Training, Decentralized Execution)
Rất phù hợp với bài toán này. Lúc train thì critic thấy full state, lúc deploy
thì mỗi drone chỉ cần thông tin 1-hop → không cần server trung tâm khi bay thật.

---

## PHẦN 2: Các vấn đề thiết kế cần xem xét lại

### ⚠️ VẤN ĐỀ 1: Reward shaping ở High-level quá phức tạp cho DQN/PPO giai đoạn đầu

**Cậu đề xuất 4 thành phần reward:**
1. Độ phủ sóng tối thiểu
2. Băng thông (khoảng cách)
3. Đường dự phòng (failover)
4. Drone pin yếu về sạc

**Vấn đề:** Khi có nhiều thành phần reward cộng lại, agent thường bị **reward
hacking** — nó maximize một thành phần rất dễ và bỏ qua phần còn lại. Ví dụ:
agent có thể học cách luôn giữ kết nối A-B đơn giản bằng cách đứng im thành
hàng thẳng, bỏ qua hoàn toàn băng thông và failover.

**Khuyến nghị:** Bắt đầu với **chỉ 1 reward component** (kết nối A-B), train
đến hội tụ, sau đó mới thêm dần các thành phần khác. Đây gọi là **reward
curriculum**.

---

### ⚠️ VẤN ĐỀ 2: Action space của High-level quá liên tục và quá lớn

**Cậu đề xuất:** Action = (N, 3) continuous — delta xyz cho N drone.

**Vấn đề với DQN:** DQN chỉ hoạt động với **discrete action**. Với continuous
action (N*3 chiều), cậu phải dùng PPO/SAC. Nhưng PPO với output (3, 3) = 9
chiều continuous cũng khó hội tụ nếu reward thưa.

**Vấn đề với PPO:** Output 9 chiều continuous mean cậu cần nhiều sample hơn
để học, đặc biệt khi reward từ simulation PyBullet chậm và noisy.

**Khuyến nghị ngắn hạn:** Thay vì continuous xyz, dùng **discrete topology
action** cho high-level:
- Action 0: Dàn đều giữa A và B (equally spaced)
- Action 1: Tụ lại về phía A (tăng redundancy phía A)
- Action 2: Tụ lại về phía B
- Action 3: Hình tam giác (tăng failover)
- Action 4: Drone pin yếu về sạc, drone khác bù vào

Sau khi DQN hội tụ tốt với discrete, mới chuyển sang continuous PPO.

---

### ⚠️ VẤN ĐỀ 3: Temporal mismatch giữa 2 tầng

**Vấn đề:** High-level ra lệnh "đi tới (x, y, z)", nhưng drone cần **nhiều
bước low-level** để thực sự đến đó. Trong thời gian đó, topo graph thay đổi.
High-level không biết drone đang trên đường đi, nên có thể ra lệnh mới liên
tục → instability.

**Khuyến nghị:** High-level nên ra lệnh với **tần suất thấp** hơn nhiều:
- Low-level: mỗi step = 20 sim steps ≈ 0.08s (tốt)
- High-level: mỗi step = 50~100 low-level steps ≈ 4~8s (nên tăng lên)

Trong code đã thêm `HL_STEPS = 200`, nhưng cần đảm bảo mỗi HL step thực sự
chạy đủ nhiều LL steps để drone di chuyển đáng kể.

---

### ⚠️ VẤN ĐỀ 4: Drone về sạc làm vỡ đồ thị đột ngột

**Vấn đề:** Khi drone về sạc, nó biến mất khỏi đồ thị. Nếu đây là drone duy
nhất nối A-B, mạng sập ngay. High-level agent cần **biết trước** (lookahead)
rằng drone sắp về sạc để điều chỉnh vị trí các drone khác **trước** khi drone
đó rời đi.

**Giải pháp đề xuất:**
1. Battery level là feature trong state (đã có ✅)
2. Thêm reward penalty **sớm** khi pin < 30% nhưng chưa có drone thay thế vị trí
3. Hoặc: khi drone i về sạc, clone vị trí nó vào `targets` cho drone gần nhất
   để tự động bù đắp

---

### ⚠️ VẤN ĐỀ 5: Low-level reward tính trước khi thực hiện action

**Trong code tớ đã tính reward TRƯỚC khi step physics.** Đây là một lựa chọn
thiết kế có trade-off:

**Ưu điểm:** Đánh giá chất lượng lựa chọn action thuần túy, không bị noise của
physics làm nhiễu (gió, PID lag).

**Nhược điểm:** Agent không biết action thực sự đưa nó đến đâu. Có thể học
policy tốt về mặt hướng nhưng không chính xác về kết quả.

**Khuyến nghị:** Dùng **cả hai** — 50% cosine reward trước, 50% distance reward
sau khi di chuyển (delta distance đến goal). Điều này balance giữa direction và
outcome.

---

### ⚠️ VẤN ĐỀ 6: Recursion trong DFS (_graph_is_connected)

**Code gốc và tớ đều dùng DFS đệ quy.** Với N = 5~10 drone thì ổn, nhưng Python
có giới hạn recursion depth (~1000). Với N lớn hơn hoặc nested calls trong
training loop chạy hàng triệu lần, nên đổi sang BFS iterative.

```python
# Thay thế an toàn hơn:
from collections import deque
def _graph_is_connected_bfs(self, obs_matrix):
    nodes = [self.base_A, self.base_B] + [obs_matrix[i, :3] for i in range(self.num_drones)]
    N = len(nodes)
    visited = set([0])
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in range(N):
            if v not in visited:
                if np.linalg.norm(nodes[u] - nodes[v]) < COMM_RADIUS:
                    visited.add(v)
                    queue.append(v)
    return 1 in visited
```

---

### ⚠️ VẤN ĐỀ 7: `_count_backup_paths` (Ford-Fulkerson) rất chậm trong training loop

**Vấn đề:** Ford-Fulkerson chạy O(V*E) mỗi lần gọi. Trong training loop chạy
hàng triệu step, đây là bottleneck nghiêm trọng.

**Khuyến nghị giai đoạn đầu:** Thay thế bằng metric đơn giản hơn: **số lượng
neighbor của mỗi node**. Node có ≥ 2 neighbor = có redundancy tiềm năng.

```python
def _approx_redundancy(self, obs_matrix):
    # Nhanh hơn Ford-Fulkerson 100x
    degrees = np.zeros(self.num_drones)
    for i in range(self.num_drones):
        for j in range(self.num_drones):
            if i != j:
                d = np.linalg.norm(obs_matrix[i, :3] - obs_matrix[j, :3])
                if d < COMM_RADIUS:
                    degrees[i] += 1
    return float(np.mean(degrees))  # higher = more redundancy
```

---

## PHẦN 3: Vấn đề kỹ thuật với DQN cho High-level

**DQN KHÔNG phù hợp cho high-level vì:**

1. **Action space continuous:** High-level action là (N, 3) float — DQN không
   handle được. DQN chỉ dùng cho discrete.

2. **Multi-agent:** DQN gốc là single-agent. MADDPG (multi-agent DQN-style) phức
   tạp hơn nhiều.

3. **Non-Markovian:** Trạng thái đồ thị thay đổi chậm, nhưng reward có delay
   (drone cần thời gian di chuyển đến đích mới ảnh hưởng kết nối). DQN với
   Q-learning cần Markov assumption chặt.

**Khuyến nghị rõ ràng:**

| Tầng       | Algorithm | Lý do                                              |
|------------|-----------|----------------------------------------------------|
| Low-level  | DQN ✅    | Discrete action, single agent, reward dense        |
| High-level | PPO ✅    | Continuous action, multi-agent, reward có thể sparse|

---

## PHẦN 4: Lộ trình train khuyến nghị

```
Giai đoạn 1: Train Low-level DQN độc lập
  → Target: random mỗi episode
  → Điều kiện hội tụ: success rate > 70% (tới target trong 300 steps)
  → Thời gian ước tính: 100k~200k steps trên laptop CPU

Giai đoạn 2: Train High-level PPO (chỉ reward kết nối A-B)
  → Low-level bị freeze (không train thêm)
  → High-level chỉ output targets, low-level execute
  → Chỉ 1 reward: connected? +10 : -20
  → Điều kiện: connectivity rate > 80%

Giai đoạn 3: Fine-tune High-level với full reward
  → Thêm dần bandwidth, backup path, battery rewards
  → Load checkpoint từ giai đoạn 2

Giai đoạn 4: Joint fine-tune (optional)
  → Unfreeze low-level, train cả hai cùng lúc với nhỏ lr
```

---

## PHẦN 5: Những thứ còn thiếu cần implement sau

1. **Charging station logic:** Drone về sạc cần thực sự ở trạm (teleport hoặc
   landed), không chỉ set `charging=True`. Hiện tại PID vẫn cố bay đến (0,0,0.5)
   mỗi step — tốn tính toán vô ích.

2. **Obstacle avoidance:** Low-level hiện tại chỉ tránh drone khác, chưa có
   obstacle tĩnh (tòa nhà, cây cối). Cần thêm raycast PyBullet nếu muốn thực tế.

3. **Communication delay simulation:** Trong thực tế, drone không nhận thông tin
   tức thì. Thêm delay 1~2 steps vào obs có thể làm policy robust hơn.

4. **Evaluation pipeline:** Chưa có script test riêng với metrics đầy đủ
   (uptime, latency estimate, failover time).

5. **Normalization:** Observation chưa được normalize. Với pos có thể lên đến
   [0~8m] và vel [-2~2 m/s], các feature có scale rất khác nhau → ảnh hưởng
   tốc độ hội tụ. Cần thêm `VecNormalize` wrapper khi dùng SB3.
```