Dưới đây là **Master Plan** chi tiết để bạn triển khai hệ thống Capacity Planning với các tính năng Simulation, Overload Analysis và Workcenter Management.

Tôi chia làm **5 Giai đoạn (Phases)** để bạn dễ cuốn chiếu, làm đến đâu chắc đến đó.

---

# 📅 Capacity Planning System Implementation Plan

## 📌 Phase 1: Database Restructuring (Nền móng)

**Mục tiêu:** Tạo các bảng mới để hỗ trợ lưu trữ phân cấp (Job -> Order -> Task), số liệu thống kê (Stats) và cấu hình mô phỏng (Draft Input).

* [ ] **1.1. Refactor bảng `CapacityPlanningJob**`
* Thêm field `Status` (DRAFT, PROCESSING, APPROVED).
* Thêm field `SummaryMetrics` (JSON) để lưu KPI tổng quan.


* [ ] **1.2. Tạo bảng Result & Detail (Thay thế cấu trúc cũ)**
* Tạo bảng `CPOrderResult`: Lưu kết quả trễ/sớm, nguyên nhân trễ của từng đơn hàng.
* Update bảng `CPTaskAssignment`:
* Chuyển `StartTime`/`EndTime` sang `time.Time`.
* Thêm `ParentTaskId` (cho Batching).
* Thêm `OrderId`, `MachineId` (có Index).


* Tạo bảng `CPWorkcenterStat`: Lưu `TotalCapacity`, `UsedCapacity`, `Status` theo ngày/tuần.


* [ ] **1.3. Tạo bảng Simulation Config (Lớp "Override")**
* Tạo bảng `CPSimulationResourceConfig`: Lưu thay đổi về ca kíp, số lượng máy/người cho Job cụ thể.
* Tạo bảng `CPSimulationOrderConfig`: Lưu thay đổi về DueDate, Priority cho Job cụ thể.


* [ ] **1.4. Tạo bảng Master Data**
* Tạo bảng `WorkcenterMapping`: Định nghĩa nhóm máy (WC_KNIT_7G gồm máy nào, WC_LINKING gồm operation nào).



---

## ⚙️ Phase 2: Engine & Preprocessing Upgrade (Trái tim xử lý)

**Mục tiêu:** Sửa lỗi lệch múi giờ, lấn giờ nghỉ, và xử lý logic Batching/Setup Time chính xác.

* [ ] **2.1. Nâng cấp Golang `Preprocessor**`
* [ ] Thêm `TimezoneOffset` vào struct.
* [ ] Sửa hàm `getShiftTimes`: Trừ Offset để đồng bộ giờ Local -> UTC.
* [ ] Sửa hàm `calculateTotalDuration`: Dùng `math.Ceil`, **tách SetupTime** ra khỏi Duration.
* [ ] Implement `calculateMachineUnavailability` & `calculateWorkerUnavailability` theo logic mới (Tự động điền Gap).
* [ ] **Batching:** Sửa hàm `Process` để trả về Map `BatchID -> [SubTasks]`. Lưu Map này vào **Redis** (Key: `job:{id}:batch_map`).


* [ ] **2.2. Nâng cấp Python Engine**
* [ ] Nhận `TimezoneOffset` và áp dụng khi parse input (nếu cần).
* [ ] Implement logic **Sequence Dependent Setup**: Thêm ràng buộc Gap giữa 2 task khác Design trên cùng 1 máy.
* [ ] Implement logic `_analyze_why_late` và gom nhóm kết quả (`OrderMetrics`) trước khi trả về.



---

## 🎮 Phase 3: Simulation Workflow (Tính năng cốt lõi)

**Mục tiêu:** Cho phép người dùng chỉnh sửa input (Draft) và chạy mô phỏng mà không ảnh hưởng dữ liệu thật.

* [ ] **3.1. API: Get Standard Config**
* API trả về lịch làm việc hiện tại và danh sách đơn hàng gốc để FE hiển thị form chỉnh sửa.


* [ ] **3.2. API: Run Simulation (The Merger)**
* Nhận JSON thay đổi từ FE (Tăng ca, đổi DueDate).
* Tạo `CapacityPlanningJob` (Status: DRAFT).
* Lưu thay đổi vào bảng `CPSimulation...Config`.
* **Logic Merge:**
* Load Real Data (Orders, Machines).
* Apply Override Data (Ghi đè DueDate, Append Overtime Shift).


* Gửi dữ liệu đã Merge sang Python.


* [ ] **3.3. Webhook Receiver (Golang)**
* Nhận payload từ Python.
* Lấy Batch Map từ Redis -> Bung task con cho `CPTaskAssignment`.
* Tính toán Aggregation (Xem Phase 4).
* Batch Insert vào DB (`CPOrderResult`, `CPTaskAssignment`, `CPWorkcenterStat`).
* Update Job Status -> `COMPLETED`.



---

## 📊 Phase 4: Aggregation & Visualization (Hiển thị)

**Mục tiêu:** Tính toán số liệu để hiển thị Dashboard nhanh tức thì.

* [ ] **4.1. Logic Aggregation (Chạy ngay sau khi nhận Webhook)**
* Viết hàm `CalculateWorkcenterStats`:
* Input: List Assignments + Simulation Config (để lấy Total Capacity mới).
* Output: List `CPWorkcenterStat` (Load %, Status: IDLE/OVERLOAD).




* [ ] **4.2. API: Dashboard Overview**
* Query `CPWorkcenterStat` group by Week/Workcenter.
* Query `CPOrderResult` đếm số lượng đơn trễ.


* [ ] **4.3. API: Order Analysis**
* Query `CPOrderResult` lấy danh sách đơn hàng, sort theo `DelayMinutes`.


* [ ] **4.4. API: Daily Schedule & Detail**
* Query `CPWorkcenterStat` (theo ngày).
* Query `CPTaskAssignment` (filter theo ngày & machine) để vẽ Gantt/List.



---

## ✅ Phase 5: Approval & Integration (Về đích)

**Mục tiêu:** Chốt phương án và áp dụng vào thực tế.

* [ ] **5.1. API: Approve Job**
* User chọn 1 Job (Simulation) ưng ý nhất -> Bấm Approve.
* Backend thực hiện:
* Update Job Status -> `APPROVED`.
* (Optional) Tạo Ticket yêu cầu tăng ca thật dựa trên `CPSimulationResourceConfig`.
* (Optional) Update DueDate thật vào bảng `Order` dựa trên `CPSimulationOrderConfig`.




* [ ] **5.2. Cleanup Worker**
* Viết Cronjob xóa các Job `DRAFT` cũ quá 7 ngày và dữ liệu liên quan (Tasks, Results) để sạch DB.



---

### 💡 Gợi ý thứ tự thực hiện:

1. Làm **Phase 1** (DB) trước tiên.
2. Làm **Phase 2.1** (Golang Preprocessor) để fix lỗi múi giờ.
3. Làm **Phase 2.2** (Python) để fix lỗi Setup Time.
4. Test chạy luồng cũ xem dữ liệu vào bảng mới có đúng không.
5. Làm **Phase 3** (Simulation Logic) để bắt đầu tính năng "What-if".
6. Làm **Phase 4** để hiển thị lên Dashboard.

Bạn có thể copy nội dung này vào file `PLAN.md` trong project để theo dõi tiến độ! Chúc bạn code "mượt" không bug! 🚀