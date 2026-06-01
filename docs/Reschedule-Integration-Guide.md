# Re-schedule Integration Guide (Go ↔ Python Solver)

**Audience:** Go backend team integrating the new stability-preserving re-schedule flow.
**Status:** Python side merged; Go side needs to adopt the changes described below.
**Owner:** OR-tools team. Questions → kimiaki@…

---

## 1. Why this exists

Mỗi lần chạy solver từ trước đến nay đều trả về lịch khác (multi-worker LNS + payload thay đổi giữa hai lần solve → search tree khác). Người vận hành thấy lịch "nhảy" liên tục, không tin được hệ thống.

Python solver vừa thêm cơ chế **warm-start hint + minimum-perturbation penalty** trên CP-SAT. Khi Go gửi sang lịch cũ kèm payload mới, solver:

1. Dùng lịch cũ làm warm-start (`AddHint`).
2. Thêm penalty vào objective: lệch start-time → phạt nhẹ; đổi máy → phạt nặng.

Kết quả: với cùng input, keep-rate máy lên ≥ 95%, Σ\|Δstart\| → 0. Với input nhiễu (đơn mới / đổi due-date), chỉ những task buộc-phải-dời mới dời.

Quan trọng: đây là **soft constraint**. Khi có conflict thực sự (deadline mới gấp, máy hỏng), solver vẫn dời được — penalty chỉ là tie-breaker giữa các nghiệm gần-tối-ưu.

---

## 2. Hai endpoint, hai use case

### 2.1. `POST /api/v1/solve` — cold-start

Dùng cho:
- Tạo plan đầu ngày / sau reset / khi không có lịch trước.
- Operator yêu cầu replan toàn bộ.

**Không cần gì mới.** Nếu Go lỡ gửi kèm `reschedule_hint`, Python **chủ động bỏ** (set `None` trước khi enqueue). Backward-compatible 100%.

### 2.2. `POST /api/v1/re-schedule` — stability-preserving re-solve

Dùng cho:
- Báo cáo tiến độ định kỳ (mỗi giờ / mỗi ca).
- Có đơn mới chen / due-date đổi / quantity đổi.
- Bất kỳ situation nào "nên giữ lịch gần như cũ".

**Yêu cầu cứng**: `reschedule_hint.previous_assignments` non-empty. Thiếu → HTTP **400**, không enqueue.

Endpoint cũ `/api/re-schedule` (không có prefix `/v1`) vẫn hoạt động làm alias deprecated. **Migrate sang `/api/v1/re-schedule` trong sprint tới**; alias sẽ bị xóa khi tất cả callers đã chuyển.

---

## 3. Schema thay đổi

### 3.1. `SolverPayload` — thêm 1 field optional

```jsonc
{
  "job_id": "...",
  "config": { /* unchanged */ },
  "machines": [ /* unchanged */ ],
  "resources": [ /* unchanged */ ],
  "tasks": [ /* unchanged */ ],
  "material_capacities": { /* unchanged */ },

  // MỚI — optional. Bỏ qua / null → behavior cũ.
  "reschedule_hint": null
}
```

### 3.2. `RescheduleHint`

```jsonc
{
  "previous_assignments": [
    /* see 3.3 */
  ],

  // Trọng số đã calibrated. Không nên override trừ khi đo lường thấy cần.
  "stability_weight_time_per_min": 500,
  "stability_weight_machine_swap": 20000,

  // Khi true: task có task_id mới nhưng cùng original_order_id với 1 prev
  // → vẫn được hint máy (không hint thời gian). Để xử lý slicing/rename.
  "match_by_order_fallback": true
}
```

Default đã chốt qua đo thực tế (xem [§5](#5-trọng-số-calibration-tại-sao-là-500--50-000)). Go chỉ cần gửi `previous_assignments`; ba field còn lại có thể omit và Pydantic sẽ điền default.

### 3.3. `PreviousAssignment`

```jsonc
{
  "task_id":            "K1-ORDER_017",      // bắt buộc
  "machine_id":         "KM_03",             // bắt buộc
  "start_time":         1240,                // bắt buộc, virtual minutes
  "end_time":           1480,                // bắt buộc
  "original_order_id":  "ORDER_017"          // optional nhưng nên gửi
}
```

5 field. `original_order_id` không bắt buộc nhưng **gửi nó** để hỗ trợ fallback khi task_id đổi (xem §6 về slicing).

### 3.4. Go struct gợi ý

```go
type PreviousAssignment struct {
    TaskID           string `json:"task_id"`
    MachineID        string `json:"machine_id"`
    StartTime        int    `json:"start_time"`
    EndTime          int    `json:"end_time"`
    OriginalOrderID  string `json:"original_order_id,omitempty"`
}

type RescheduleHint struct {
    PreviousAssignments       []PreviousAssignment `json:"previous_assignments"`
    StabilityWeightTimePerMin int                  `json:"stability_weight_time_per_min,omitempty"`
    StabilityWeightMachineSwap int                 `json:"stability_weight_machine_swap,omitempty"`
    MatchByOrderFallback      *bool                `json:"match_by_order_fallback,omitempty"`
}

// Add to existing SolverPayload (do NOT rename existing fields)
type SolverPayload struct {
    // ... existing fields ...
    RescheduleHint *RescheduleHint `json:"reschedule_hint,omitempty"`
}
```

Dùng pointer + `omitempty` để gửi `null` khi không có hint (tương đương không có field). Cả hai dạng đều được Python chấp nhận.

---

## 4. Luồng integration

### 4.1. Lưu snapshot mỗi lần callback

Mỗi lần Python POST webhook `assignments` về Go, Go phải lưu snapshot mới:

```sql
CREATE TABLE schedule_snapshots (
  job_id      TEXT NOT NULL,
  version     INT  NOT NULL,
  assignments JSONB NOT NULL,   -- toàn bộ assignments array
  created_at  TIMESTAMP NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, version)
);
```

`assignments` lưu nguyên field `assignments` từ `SolverResponse`. Khi build hint, lấy `task_id`, `machine_id`, `start_time`, `end_time` từ đây cộng `original_order_id` tra cứu từ payload gốc.

### 4.2. Build payload re-schedule

> ⚠️ **Pitfall đã ghi nhận trong production**: lần đầu Go triển khai chỉ gửi
> hint cho Linking + Washing + Iron + Packing, BỎ MẤT Knitting Phase 1 (190
> BATCH tasks). Kết quả: Phase 1 hoàn toàn tự do mỗi lần solve → end_time đổi
> → start_lb của các phase sau đổi → KHÔNG phase nào ổn định được, dù 4 phase
> sau đều có hint. Quy tắc: **đưa HẾT assignments từ snapshot, không filter
> theo operation**.

```go
func BuildReschedulePayload(ctx context.Context, jobID string) (*SolverPayload, error) {
    prev := LoadLatestSnapshot(ctx, jobID)
    if prev == nil {
        // Chưa có snapshot → đây thực ra là cold-start, dùng /solve
        return BuildSolvePayload(ctx, jobID)
    }

    payload := BuildBasePayload(ctx, jobID)  // tasks, machines, resources hiện tại

    // Map task_id → original_order_id từ tasks hiện tại (nguồn truth)
    orderByTask := map[string]string{}
    for _, t := range payload.Tasks {
        orderByTask[t.TaskID] = t.OriginalOrderID
    }

    prevAssignments := make([]PreviousAssignment, 0, len(prev.Assignments))
    for _, a := range prev.Assignments {
        // KHÔNG FILTER theo operation — gửi mọi assignment, gồm cả BATCH_* knitting.
        // Pinned tasks ở payload mới Python tự skip; gửi thừa không hại.
        prevAssignments = append(prevAssignments, PreviousAssignment{
            TaskID:          a.TaskID,
            MachineID:       a.MachineID,
            StartTime:       a.StartTime,
            EndTime:         a.EndTime,
            OriginalOrderID: orderByTask[a.TaskID],  // có thể empty nếu task không còn
        })
    }

    payload.RescheduleHint = &RescheduleHint{
        PreviousAssignments: prevAssignments,
        // Để trống ba weight field → Python dùng default đã calibrated
    }
    return payload, nil
}
```

**Sanity check ngay sau khi build hint** (đề nghị thêm log/metric ở Go):

```go
// Đếm hint coverage theo operation. Nếu Knitting < total - pinned thì có bug
// trong logic build hint — xem pitfall ở trên.
opCount := map[string]int{}
for _, a := range prev.Assignments {
    op := opByTask[a.TaskID]
    opCount[op]++
}
log.Infof("reschedule hint coverage: %v", opCount)
```

### 4.3. Pin tasks đã chạy

Logic không đổi: nếu task đã hoàn thành / đang chạy, Go set `is_pinned=true` + `pinned_machine_id` + `pinned_start_time` + `pinned_end_time` trong `tasks` như hiện tại.

**KHÔNG cần** đưa pinned task vào `previous_assignments` — Python đã skip pinned task khỏi hint (không có biến để hint). Nếu bạn vẫn gửi, không có hại, Python sẽ ignore.

### 4.4. Khi nào dùng route nào

| Trường hợp | Route |
|---|---|
| Plan lần đầu trong ngày | `/api/v1/solve` |
| Có task mới chen vào | `/api/v1/re-schedule` |
| Đổi due-date / quantity | `/api/v1/re-schedule` |
| Báo cáo tiến độ định kỳ | `/api/v1/re-schedule` |
| Operator request "Reset toàn bộ" | `/api/v1/solve` |
| Test/benchmark | hoặc cũng được, tùy mục đích |

---

## 5. Trọng số calibration — tại sao là 500 / 50 000

Đã đo từ baseline thật trên CP-SAT của project này (priority=3 default):

- **Lateness coeff** = `10**(6-priority) * 100` = 100 000 / phút trễ / task.
- **Start tie-breaker** = `max(1, 10**(6-priority) // 100)` = 10 / phút start / task.
- **w_time** = 500 → 50× tie-breaker (đủ để giữ vị trí) và 200× < lateness (không bao giờ đẩy task quá deadline chỉ để ổn định).
- **w_machine** = 50 000 = 100× w_time (calibrated lại sau khi đo production 732-task payload với 60s search; w_machine cũ 20 000 cho keep_rate 86% — chưa đủ với multi-worker LNS).

Cộng với `repair_hint=True` (đã bật cứng trong `make_solver`), CP-SAT actively cố repair hint thay vì discard khi gặp conflict. Hai cái cộng lại đẩy keep_rate Phase 1 ≈ 99%, Phase 2 ≈ 98% trên payload sản xuất.

**Tổng penalty stability < 10% base objective** trong fixture symmetric → chỉ làm tie-breaker, không lật nghiệm chính. Vẫn < lateness coeff (100 000) nên không bao giờ chọn "ổn định thay vì kịp deadline".

Nếu sau khi triển khai bạn thấy keep-rate thấp hơn kỳ vọng:

- Phase 1 (Knitting) / Phase 2 (Linking) < 95%: tăng `stability_weight_machine_swap` lên 100 000.
- Phase 3 (Washing) / Phase 4 (Iron, Packing) drift: đây là vấn đề kiến trúc của batching model (xem §11). Không nên tăng weight quá cao — chỉ tốn objective.

Đừng tự nâng `w_time` lên ngưỡng `lateness` (= 100 000) — sẽ làm task hi-priority bị stuck ở vị trí cũ dù đã trễ.

### 5.1. Khuyến nghị về `max_search_time` và `max_deterministic_time`

Trên payload 700+ task với 100+ máy + 5 phase, `max_search_time=60s` là **quá ngắn** — solver kết thúc trong trạng thái timeout-feasible (không hội tụ), mỗi lần dừng ở nhánh khác.

Khuyến nghị: với payload sản xuất hàng ngày, nâng `max_search_time` lên 120-180s. Hint + repair giúp solver tới optimum nhanh hơn, không phải chạy nhiều hơn — phần thời gian thêm chủ yếu dùng để hội tụ.

### 5.2. Nguồn-1 (multi-worker race) — đã fix bằng `num_search_workers=1` cưỡng bức

**Empirical finding** (probe trên payload 732-task production):

| Cấu hình | 3 lần solve identical |
|---|---|
| `workers=8 wall_only` | ❌ FALSE |
| `workers=8 max_deterministic_time=240` | ❌ FALSE |
| `workers=1 wall_only` | ✅ TRUE |

OR-Tools 9.8+ docs nói `max_deterministic_time` đảm bảo reproducibility multi-worker, nhưng async bound-sharing giữa workers tạo race condition mà det_time không khử được. Single-worker là cách DUY NHẤT đảm bảo determinism trên payload thật.

**Hành vi `make_solver` hiện tại**:
- Khi `has_hint=True` (path /api/v1/re-schedule) → **ép `num_search_workers=1`** bất kể Go cấu hình. Wall-clock cap tăng 4× để bù chậm.
- Khi `has_hint=False` (path /api/v1/solve cold) → giữ nguyên `num_search_workers` Go cấu hình (mặc định 8). Tốc độ ưu tiên hơn reproducibility vì không có lịch cũ cần ổn định với.

**Trade-off**: single-worker chậm ~2-3× so với 8-worker. Với payload 60-180s, chấp nhận tăng lên 120-360s đổi lấy lịch ổn định. Đây là quyết định ngầm bên Python — Go không cần biết.

**Field `max_deterministic_time` vẫn được set best-effort** cho cả hai path để output đỡ phụ thuộc tốc độ máy chủ, nhưng đảm bảo determinism cho /re-schedule đến từ `workers=1`, không phải det_time.

---

## 6. Slicing / rename — fallback theo order

Production code đôi khi rename task_id khi quantity thay đổi (vd `K1-ORDER_017_SLICE_3` → `K1-ORDER_017_SLICE_3_v2` sau khi điều chỉnh qty).

Python xử lý qua **fallback theo `original_order_id`**:

1. Task hiện tại có task_id KHÔNG khớp prev nào exact.
2. Nhưng có `original_order_id` khớp với ≥ 1 prev.
3. Python chọn prev có `machine_id` thuộc `compatible_resource_ids` của task hiện tại.
4. Áp **chỉ machine hint + machine_swap penalty**, KHÔNG áp time hint (vì N slice mới có thể có start khác nhau, ép cùng prev_start sẽ phá no-overlap).

**Hành động cho Go**: luôn gửi `original_order_id` trong `PreviousAssignment` ngay cả khi nghĩ task_id không đổi. Nó cost-free và bảo vệ trường hợp rename.

---

## 7. Máy hỏng — quy ước bắt buộc

Khi máy `KM_05` hỏng, task đang chạy trên đó cần dời sang máy khác. Quy ước:

1. **Bỏ `is_pinned`** của task đó (Go phát hiện hỏng → unpin).
2. **Đưa task vào `previous_assignments`** (giữ thông tin lịch cũ).
3. **VÀ một trong hai:**
   - **(Ưu tiên)** Loại `KM_05` khỏi `compatible_resource_ids` của task đó.
   - Hoặc thêm `unavailability` window cover khoảng thời gian máy hỏng vào resource `KM_05`.

**Không được chỉ bỏ pin mà giữ máy nguyên trong `compatible_resource_ids`.** Lý do: machine_swap penalty sẽ chống lại việc đổi máy ⇒ solver có thể vẫn giữ task trên máy hỏng → infeasible / wrong.

Tóm tắt: **gỡ máy khỏi tập compatible TRƯỚC khi gửi sang Python**.

---

## 8. Validation / error responses

| Tình huống | HTTP status | body |
|---|---|---|
| `/api/v1/solve` — body hợp lệ | 200 | `{"celery_task_id", "job_id"}` |
| `/api/v1/re-schedule` — body hợp lệ | 200 | `{"celery_task_id", "job_id"}` |
| `/api/v1/re-schedule` — thiếu `reschedule_hint` | 400 | `{"detail": "re-schedule requires a non-empty reschedule_hint.previous_assignments…"}` |
| `/api/v1/re-schedule` — `previous_assignments: []` | 400 | same as above |
| Schema sai (vd thiếu `machine_id`) | 422 | Pydantic validation error |

---

## 9. Test plan đề xuất bên Go

1. **Unit**: serialize `RescheduleHint` → JSON, gửi qua HTTP mock, đảm bảo Python trả 200.
2. **Integration**: chạy 1 lần solve, lưu snapshot, build hint từ snapshot, gọi re-schedule, so sánh `assignments` mới với cũ — assert ≥ 95% machine match.
3. **Edge case**: snapshot rỗng → fallback sang `/solve` (Go-side logic), không tự build hint rỗng rồi gửi /re-schedule.
4. **Backward-compat**: gửi payload cũ (không có `reschedule_hint`) → vẫn 200, vẫn solve bình thường.

---

## 10. Roll-out plan đề xuất

1. **Sprint N**: Go merge schema struct + DB migration cho `schedule_snapshots`. Chưa gọi `/re-schedule`.
2. **Sprint N+1**: Go bật `/re-schedule` cho 1 job-id pilot trong staging. Quan sát log Python: dòng `🎯 Phase{N} stability_stats: ...` cho biết match-rate, n_hinted, penalty terms.
3. **Sprint N+2**: roll-out production, monitoring keep-rate qua snapshot diff.
4. **Sprint N+3**: deprecate `/api/re-schedule` (no-v1 alias).

---

## 11. Liên hệ + observability

Mỗi lần solve, Python log structured (qua `model.py` + per-phase):

```
🎯 reschedule_hint received: 47 previous assignments
🎯 Phase1 stability_stats: total_previous=23 matched_exact=23 matched_via_order=0 n_hinted=23 time_terms=23 machine_terms=46
🎯 Phase2 stability_stats: total_previous=15 matched_exact=15 matched_via_order=0 n_hinted=15 time_terms=15 machine_terms=22
```

`matched_via_order > 0` → có slicing rename xảy ra (Go nên xem lại có bug rename không). `n_hinted < total_previous` → có prev không khớp task hiện tại (task đã xong / bị xóa).

Khi có vấn đề: copy snapshot + payload + log Python → gửi OR-tools team.

---

## 12. Known limitation — Phase 3 batching stability

Phase 3 (Washing) là **batching solver**: tasks được group theo (color, substance), mỗi group là 1 model CP-SAT riêng với biến `batch_starts[k]` (slot start time). Khi task được assign vào slot k, `task.start == batch_starts[k]`.

`batch_starts[k]` là biến NEW mỗi lần build model → **không có trong hint**. Hint chỉ áp lên `task.start` (= `batch_starts[k]` qua ràng buộc), nhưng vì batch_starts có thể đổi → task.start dời theo dù task được giữ trên cùng slot logically.

Hệ quả đo trên production: Washing/Iron/Packing keep_rate ≈ 75-84%, Knitting/Linking 97-99% (Phase 4 đỉa Phase 3 nên dời theo).

Kế hoạch (PHA 5 nếu cần): build `prev_batch_id → batch_starts[k]` mapping từ snapshot, AddHint cho `batch_starts[k]` tương tự task vars. Chưa implement vì:
- Đòi hỏi snapshot lưu thêm batch metadata (Go phải fetch từ `assignment.batch_slot_id`).
- 75-84% keep-rate Phase 3 chấp nhận được tạm thời nếu mục tiêu chính là Phase 1+2 (output lệch nhau ngắn không gây hỗn loạn sản xuất nhiều như Phase 1 đổi máy).

Nếu Phase 3/4 stability là showstopper cho Go team thì raise và OR-tools team sẽ ưu tiên PHA 5.
