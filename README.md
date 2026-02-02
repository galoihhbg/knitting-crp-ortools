Dưới đây là tài liệu kỹ thuật tóm tắt về **Capacity Planning Engine** mà bạn đang xây dựng. Tài liệu này được chuẩn hóa dựa trên các đoạn code Golang và Python chúng ta đã trao đổi, giúp bạn dễ dàng tra cứu và bàn giao cho team.

---

# 📘 Capacity Planning Engine Documentation

## 1. Kiến trúc Tổng quan (Architecture)

Hệ thống hoạt động theo mô hình **Pre-calculation in Go** và **Optimization in Python**.

* **Input (Raw Data):** Đơn hàng, Danh sách máy, Lịch làm việc (Shift), Định mức (Routing).
* **Golang Service (The "Brain"):**
* Tính toán Duration chính xác cho từng task.
* Chia nhỏ task lớn (Slicing) hoặc gộp task nhỏ (Batching).
* Sinh ra danh sách Resource ảo (Worker) và Resource thật (Machine).
* Tính toán lịch nghỉ (Unavailability) dựa trên ca làm việc.


* **Python Service (The "Solver"):**
* Nhận dữ liệu đã "sạch".
* Dùng Google OR-Tools (CP-SAT) để tìm phương án xếp lịch tối ưu.


* **Output:** Danh sách Assignment (Task nào, Máy nào, Bắt đầu, Kết thúc).

---

## 2. Quy ước Thuật ngữ & Dữ liệu (Terminology)

### A. Thời gian (Time)

* **BaseTime:** Mốc thời gian `0` của hệ thống (thường là 00:00 ngày bắt đầu lịch).
* **Horizon:** Tổng thời gian lập lịch (tính bằng phút). Ví dụ: 7 ngày = 10,080 phút.
* **Time Unit:** Đơn vị nhỏ nhất là **Phút (Int)**.
* Ví dụ: `Start: 60` nghĩa là bắt đầu sau BaseTime 1 tiếng.



### B. Tài nguyên (Resources)

Trong hệ thống này, "Resource" bao gồm cả máy móc và con người.

| Loại Resource | ID Convention | Mô tả | Constraint OR-Tools |
| --- | --- | --- | --- |
| **Physical Machine** | `L01`, `SK01`... | Máy thật (Knitting, Linking...). Có trong danh sách `machines` đầu vào. | `NoOverlap` (Serial) |
| **Virtual Worker** | `W_{OP}_{INDEX}` | Công nhân ảo (Iron, Packing...). Được sinh ra từ `daily_allocation`. | `NoOverlap` (Serial) |
| **Batch Machine** | `W_WASHING_XX` | Máy giặt hoặc các công đoạn xử lý theo mẻ. | `Cumulative` (Batch) |

### C. Công việc (Tasks)

| Thuật ngữ | Key JSON (Snake_case) | Mô tả |
| --- | --- | --- |
| **Task ID** | `task_id` | ID duy nhất. Quy ước: `{OrderID}_b{BatchIdx}` hoặc `{OrderID}_p{SliceIdx}`. |
| **Duration** | `duration` | Thời gian thực hiện (phút). **Không được bằng 0**. |
| **Compatible IDs** | `compatible_resource_ids` | Danh sách ID các Resource **có thể** làm task này. (Quan trọng nhất). |
| **Priority** | `priority` | Độ ưu tiên (1 = Cao nhất, Urgent). Dùng để tính điểm phạt trễ hạn. |

---

## 3. Logic & Ràng buộc (Business Logic Constraints)

### 1. Logic "Âm bản" (Negative Availability)

Thay vì định nghĩa "Khi nào máy chạy", hệ thống định nghĩa **"Khi nào máy nghỉ"** (`unavailability`).

* **Unavailability bao gồm:**
* Thời gian trước khi dự án bắt đầu (Quá khứ).
* Thời gian nghỉ giữa các ca (Ví dụ: 12:00-13:00).
* Thời gian đóng cửa xưởng (Ví dụ: 17:00-08:00 sáng hôm sau).


* **Cơ chế Solver:** `Task` không được chồng lấn lên `Unavailability`.

### 2. Logic Chọn Máy (Assignments)

* Một Task có list `compatible_resource_ids` (VD: `['L01', 'L02']`).
* Solver tạo các biến Bool: `Task_on_L01`, `Task_on_L02`.
* **Ràng buộc:** `Sum(Bool) == 1` (Bắt buộc chọn đúng 1 máy).

### 3. Logic Task Gộp (Batching - Washing)

* Nhiều Task nhỏ (cùng loại Washing) được gộp thành 1 Task lớn (Batch Task).
* **SubTasks:** Các task nhỏ nằm trong trường `sub_tasks`.
* **Xử lý:** Solver chỉ xếp lịch cho Task cha (Batch). Khi Task cha có lịch, các Task con tự động nhận lịch đó.

### 4. Logic Task Chia (Slicing - Knitting)

* Một Order quá lớn (VD: 1000 hàng) được chia thành nhiều Slice nhỏ (`p1`, `p2`...).
* **Internal Dependency:** `p2` phải bắt đầu sau khi `p1` kết thúc (`Start_p2 >= End_p1`).
* **Slice Consistency:** `p2` bắt buộc phải chạy **cùng máy** với `p1` (để tránh chuyển đổi máy).

---

## 4. Hàm Mục tiêu (Objective Function)

Hệ thống chấm điểm phương án dựa trên công thức sau (Minimize Cost):

Trong đó:

1. **Makespan (W1 = 100):** Thời điểm task cuối cùng hoàn thành. Mục tiêu: Kéo ngắn thời gian dự án.
2. **Lateness (W2 = 1000 - 5000):** Tổng thời gian trễ Deadline.
* Trọng số phụ thuộc `Priority`. Task Priority 1 bị phạt nặng hơn Task Priority 3.


3. **ASAP Strategy (W3 = 1):** Tổng thời điểm kết thúc của mọi Task.
* Mục tiêu: Ép Solver làm mọi việc **Sớm Nhất Có Thể** (tránh tình trạng "nước đến chân mới nhảy").



---

## 5. Các Vấn đề Thường gặp & Cách Debug

| Triệu chứng | Nguyên nhân có thể | Cách kiểm tra |
| --- | --- | --- |
| **Assignments Rỗng** (`[]`) | Lệch tên key JSON (Pascal vs Snake) hoặc sai ID Resource. | Check log `DATA FORENSICS` trong Python. Check `compatible_resource_ids`. |
| **Start Time == End Time** | `Duration` đầu vào bằng 0. | Check logic `calculateTotalDuration` trong Golang. |
| **Máy Knitting làm việc Linking** | Logic `mapCompatibleResources` trong Go bị lỏng lẻo. | Kiểm tra xem `L01` có lọt vào list resource của Task Knitting không. |
| **Xếp lịch vào 2h sáng** | Logic tính ca đêm (`Unavailability`) bị sai. | Kiểm tra hàm `getShiftTimes` (xử lý `End < Start`). |
| **Task bị dồn hết về cuối** | Thiếu thành phần ASAP trong hàm mục tiêu. | Thêm `objective_terms.append(tv["end"])`. |

---

### Bạn muốn tôi bổ sung chi tiết nào vào tài liệu này không? (Ví dụ: Cấu trúc JSON API chi tiết?)