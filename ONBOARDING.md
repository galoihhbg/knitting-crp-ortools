# ONBOARDING — Knitting-CRP-Ortools Scheduler (CP-SAT)

> Tài liệu đào tạo dev mới cho scheduler CP-SAT công đoạn dệt:
> **knitting → linking → washing → ironing → packing**.
> Mọi sự thật dưới đây được trích từ **code / git log / test thật** và đã verify line-ref (mở từng dòng).
> Soạn 2026-06-30 trên `dev` @ `f7682e3`.

---

## 0. ĐỌC GÌ TRƯỚC — và cảnh báo nguồn

**Quy tắc nguồn (bắt buộc):**
- ✅ **Đáng tin:** code trong `app/**`, `git log`/commit diffs, `tests/**`, `logs/solver_*_CP_*.json`.
- ⚠️ **KHÔNG tin (chỉ dùng để dò chỗ, phải verify lại bằng code):** mọi file `.md` mô tả hành vi — `docs/`, `agent_docs/`, `part1..4-*.md`, `specs/`, kể cả `CLAUDE.md`/`AGENTS.md`. Chúng đã lâu không cập nhật. Ví dụ cụ thể về sự lệch xem §8.7.

**Cảnh báo branch (rất quan trọng):**
> `main` và `origin/main` đứng yên ở `5036447` (**2026-04-11**) — baseline cũ.
> **Toàn bộ scheduler hiện tại nằm trên `dev`** (HEAD, 56 commit phía trước main; `git log dev..main` rỗng).
> Trong tài liệu này, **`dev` = production-of-record**. Đừng nhầm "chưa có trên main" thành "chưa ship".

**Chạy dự án** (theo `CLAUDE.md` — kiểm lại trước khi tin):
```bash
# Test (dùng python của venv)
/home/anya/anya/crp-ortools/env/bin/python -m pytest tests/ -v
/home/anya/anya/crp-ortools/env/bin/python -m pytest tests/ -v -m "not slow"   # bỏ test 200-task chậm
# API dev
uvicorn app.main:app --host 0.0.0.0 --port 8083 --reload
# Worker dev
celery -A app.core.celery_app worker --loglevel=info --concurrency=1
# Docker
docker compose up --build
```

---

## 1. BỨC TRANH LỚN

Hệ thống nhận danh sách **task** (mỗi task = một công đoạn của một đơn) + **máy**, rồi xếp lịch lên máy bằng OR-Tools CP-SAT, tôn trọng tiền-đề-công-đoạn, sức chứa máy, ca làm việc, task đang chạy dở (pinned). Trả về `assignments` (task↔máy↔thời gian) + `overloads` (task trễ kèm nguyên nhân).

**Pipeline 5 phase tuần tự** — end-time của phase trước thành `start_lb` (lower-bound) của phase sau, nên **một phase mất tính tất định là phá hỏng mọi phase sau**:

```
knitting → linking → washing → ironing → packing
(phase1)   (phase2)   (phase3)  (phase4-iron) (phase4-pack)
```

---

## 2. VÒNG ĐỜI MỘT REQUEST

```
Go backend
  └─ POST /api/v1/solve  (hoặc /api/v1/re-schedule)        app/api/v1/solver_route.py:9 / :27
       └─ optimize_schedule.delay(data)   (Celery)          solver_route.py:19
            └─ solver_task: Engine(payload).solve()         app/tasks/solver_task.py
                 └─ pipeline.run  → 5 phase                 app/engine/pipeline.py
            └─ DOUBLE-SOLVE (cold): pass1 → hint → pass2     solver_task.py:133-141
            └─ POST WEBHOOK_URL  {assignments, overloads} → Go    .env:2
```

**Cold-solve vs Re-schedule:**
- Payload **không có** `reschedule_hint` → **cold**: chạy đủ mọi post-pass.
- Cold xong, hệ thống **tự** lấy assignments pass-1 làm `reschedule_hint` rồi solve lại (pass-2) và trả pass-2 cho UI — vì pass-2 mới là lịch ổn định (xem §8.2). Bật/tắt qua `ENABLE_DOUBLE_SOLVE` (mặc định true, `.env:8`).
  > `solver_task.py:133-141`: *"Lịch lần 1 (cold) khác lần 2 một chút vì lần 2 thực chất là re-schedule của cold … từ lần 2 trở đi ổn định."*
- Payload **có** `reschedule_hint` sẵn → re-schedule trực tiếp; **mọi post-pass bị tắt** (`if not self.reschedule_hint`, xem §8.4).

**3 endpoint** (`solver_route.py`): `/api/v1/solve` (:9), `/api/v1/re-schedule` (:27), `/api/re-schedule` legacy (:56).

---

## 2.1 LUỒNG CHẠY ĐẦY ĐỦ MỘT PIPELINE (end-to-end → kết quả cuối)

Toàn bộ nằm trong `TaskModel.solve()` (`model.py:104`) → `Pipeline.run()` (`pipeline.py`) → dyelot post-pass (`model.py:128`). Trình tự **chính xác** (cold-solve):

```
model.solve()                                            model.py:104
│
├─ Pipeline(...).run()                                   model.py:112 → pipeline.py
│   │
│   ├─ compute_global_horizon  (1 horizon chung mọi phase, giữ determinism)   pipeline.py:108
│   │
│   ├─ ① PHASE 1 — KNITTING        solve_knitting()       pipeline.py:121
│   │     └─ fail → return _phase_failure_result          :132   (FEASIBLE/empty mới đi tiếp)
│   │   ┌─ _improve_knitting()  [COLD, MONOTONE, không re-solve]   :139 / :265
│   │   │    spread_cold_knitting → left_shift_cold_knitting       :273-275
│   │   └─   (chạy TRƯỚC linking để linking xếp trên knitting đã kéo sớm)
│   │
│   ├─ ② PHASE 2 — LINKING         solve_linking()        pipeline.py:144   (start_lb = end-time knitting)
│   │     └─ fail → return failure                        :156
│   │   ┌─ _tighten_linking()  [COLD, MONOTONE]            :162 / :280
│   │   └─   left_shift_cold_linking  (chạy TRƯỚC phase 3-5)         :290-293
│   │
│   ├─ _solve_phases_3_to_5()                             pipeline.py:165 / :299
│   │   ├─ ③ PHASE 3 — WASHING     solve_washing()        :317   (start_lb = P1+P2 end-times)
│   │   │     └─ fail → return failure                    :331
│   │   ├─   washing FLUSH cuối ca  [COLD, chỉ-dịch-sớm]   :340-347   (§6.3)
│   │   ├─   washing LEFT-SHIFT     [COLD, monotone]       :353-360
│   │   └─ _solve_phases_4_5()                            :363 / :366
│   │       ├─ ④ PHASE 4 — IRONING  solve_downstream()    :382   (start_lb = …+washing)
│   │       └─ ⑤ PHASE 5 — PACKING  solve_downstream()    :401   (start_lb = …+ironing)
│   │
│   ├─ KNITTING RELAYOUT refine  [COLD, KHÔNG monotone → RE-SOLVE phase 2-5 + accept-if-no-regression]
│   │     parallel-PO + contiguity, một verify-pass chung      pipeline.py:186-194
│   ├─ SAME-QTY RE-LINK refine   [COLD, default OFF qua schema → RE-SOLVE 2-5, Pareto caps]   :202-211
│   │
│   ├─ gộp all_assignments (P1..P5)                        :214-217
│   ├─ TOUCH-UP cuối [COLD]:  balance_linking_load (relabel máy, GIỮ NGUYÊN thời gian)   :229-230
│   │                          + left_shift_cold_washing FINAL pass                       :235-239
│   └─ return {status, assignments, overloads, objective_value, solve_time}   :257-263
│
├─ allocate_dyelots()  (post-pass cấp dyelot; đọc knitting assignments +      model.py:128
│     main_yarn_consumption + dyelot_stock; bỏ qua nếu payload không có stock)
└─ result.update(dyelot_out) → return                                         model.py:132
```

**Ba loại bước, phân biệt rõ:**
1. **Solve phase** (①–⑤): mỗi phase một model CP-SAT riêng; handoff giữa phase là **end-times nguyên (int)**, không share biến CP-SAT. End-time phase trước = `start_lb` phase sau.
2. **Forward post-pass** (`_improve_knitting`, `_tighten_linking`, washing flush/left-shift): **monotone — chỉ kéo SỚM** → chạy *xen giữa* các phase, hạ nguồn xếp trên kết quả đã kéo sớm và "đi theo", **không cần re-solve** (§8.5).
3. **Verified refinement** (knitting relayout, same-qty relink): **KHÔNG monotone** → chạy *sau* khi đã có đủ P1–P5, **re-solve lại phase 2–5** với budget rẻ và chỉ chấp nhận nếu tổng lateness không tăng (§8.5).

**Trên đường re-schedule** (`reschedule_hint` có sẵn): **mọi bước loại 2 & 3 đều bị bỏ** (`if not self.reschedule_hint`, và các helper return sớm) → chỉ còn 5 lần solve phase với hint ổn định. Đây là lý do double-solve (§2) làm pass-2 = reschedule để trả lịch ổn định.

**Điều kiện dừng giữa chừng:** phase nào trả status ∉ {`feasible`,`empty`} thì `run()` trả ngay `_phase_failure_result` (`:132/:156/:331`), không chạy tiếp.

---

## 3. ĐẶC TẢ YÊU CẦU

### 3.1 Chức năng
| Yêu cầu | Nguồn |
|---|---|
| Xếp 5 công đoạn tuần tự, tôn trọng precedence `final_depends_on` | `phase4_downstream.py:139` |
| Chỉ xếp lên máy tương thích `compatible_resource_ids` | `shared.py:831` |
| Tôn trọng capacity + cửa sổ bảo trì `unavailability` | `shared.py:840`, `:854/:857` |
| Washing không vắt qua ranh ca `shift_ends_min` | `request_schema.py:107-110` |
| Giữ nguyên task pinned / in-progress | `shared.py:561` (`normalize_pinned_window`) |
| Smart washing batching (gom vào slot, sức chứa `washing_batch_capacity`=10) | `phase3_batching.py:356`, `request_schema.py:101` |
| Dyelot/creel allocation (post-pass riêng) | `app/engine/dyelot_allocator.py` |
| Re-schedule giữ ổn định từ `previous_assignments` | `request_schema.py:303` |
| Phân biệt đơn thường (`is_normal`) vs đơn gấp | `request_schema.py:56-60` |
| Báo `overloads` kèm `root_cause_code` | `response_schema.py:17-23` |

### 3.2 Phi-chức-năng (ẩn — rút từ solver/config/test)
| Thuộc tính | Cơ chế | Nguồn |
|---|---|---|
| **Determinism** (cùng input → output byte-identical) | 1 worker **+** `PYTHONHASHSEED=0` **+** sort task theo `task_id` | `shared.py:480`, `Dockerfile:12`, commit `90b8bf8`; test `test_determinism.py`, `test_input_order_determinism.py` |
| **Stability** (lần-1 ≈ lần-2; reschedule giữ kế hoạch cũ) | double-solve + hard-keep | `solver_task.py:133`, commit `be46a8e`; test `test_reschedule_stability.py` |
| **Reproducibility / replay** | log input/output + `MOCK_RESPONSE_FILE` | `solver_task.py:114`, `logs/solver_*_CP_*.json` |
| **Termination** (luôn dừng, không phụ thuộc tốc độ máy) | chỉ dừng theo `max_deterministic_time` | `shared.py:445-449` |
| **int-only** | mọi bound/penalty CP-SAT là `int` | quy ước (xem §9 / §11) |

---

## 4. SCHEMA WIRE-CONTRACT (Go ↔ Python)

> **Tên field = hợp đồng wire với Go. KHÔNG đổi tên.** (Field dùng `alias` Pydantic.)

**`SolverTask`** (`request_schema.py:42`): `task_id`, `original_order_id`, `group_id`, `order_group_id` (khóa gom dyelot, :46-51), `operation`, `qty`/`total_qty`, `priority`, `is_normal` (:56-60), `final_depends_on`, `start_after_min`, `due_at_min`, `duration`, `is_batch`/`sub_tasks`, `design_item_id`/`color_config`, `compatible_resource_ids`, `wait_offsets`, `is_slice`/`slice_index`/`parent_task_id`, `is_pinned`/`pinned_machine_id`/`pinned_start_time`/`pinned_end_time`, `demand`, `material_demands`, `main_yarn_consumption`.

**`SolverConfig`** (`:90`): `horizon_minutes`(57600), `max_search_time`(300), `random_seed`(42), `num_search_workers`(8 — **xem cảnh báo dưới**), `max_deterministic_time`(None), `washing_batch_capacity`(10), `shift_ends_min`, + nhiều cờ `enable_*`.

**`RescheduleHint`** (`:303`), **response** `Assignment`/`Overload`/`SolverResponse` (`response_schema.py`).

### Bảng cờ config & 3 mâu thuẫn schema↔runtime (CHỦ ĐÍCH, test-validated 2026-06-30 — KHÔNG phải bug)
| Cờ / field | Default schema | Giá trị **hiệu dụng** | Ghi chú |
|---|---|---|---|
| `num_search_workers` | `8` (`:95`) | **luôn = 1** | `make_solver` ép `effective_workers = 1` (`shared.py:480`) — bắt buộc cho determinism. Giá trị schema là **dead**. |
| `enable_sameqty_relink` | `False` (`:121`) | OFF qua API; **pipeline `.get(...,True)`** | Khác nhau theo đường vào (`pipeline.py:205`). Qua schema = OFF (đúng quyết định "shipped OFF"). |
| `enable_knitting_contiguity_reorder` | `True` (`:187`) | pipeline `.get(...,False)` | Effective default phụ thuộc đường vào (`pipeline.py:188/542`). |

> Các giá trị **hiệu dụng** ở trên là cái đã được test thấy tốt hơn → coi là chủ đích. **Đừng "sửa cho khớp" bằng cách đổi logic.**

**Cờ đáng chú ý khác (KHÔNG mâu thuẫn — bật/tắt như nhau ở mọi đường vào):**
- `enable_fifo_linking_floor` (default **True**, `:223`): cách linking biết panel nào "đủ điều kiện ráp".
  Bật = đổi-panel-cùng-PO (FIFO); tắt = ghép cứng theo số thứ tự. **Giải thích đầy đủ + ví dụ ở §6.2.**

---

## 5. TỪ ĐIỂN THUẬT NGỮ

### Domain
- **panel / component PO** — 1 sản phẩm gồm nhiều panel; mỗi panel là 1 PO knitting (vd front `0-641`, back `0-642`). `pipeline.py:175-178`.
- **order / `original_order_id` / `order_group_id`** — đơn bán hàng; `order_group_id` là **khóa gom dyelot**: mọi batch/panel cùng order_group_id phải dùng chung 1 dyelot (tránh lệch màu khi ghép). `request_schema.py:46-51`.
- **slice** (`is_slice`, `slice_index`) — lát cắt của task linking để link song song nhiều panel.
- **dyelot** — mẻ nhuộm. **creel** — giàn cone sợi trên **từng máy** (per-machine, không share). `dyelot_allocator.py`.
- **batch / co-location / `batch_slot_id`** — nhóm task giặt chung 1 mẻ máy giặt.
- **fungibility / FIFO-by-PO** — panel cùng `(component, qty)` là **thay-thế-được** (panel nào cũng như nhau); slice thứ k lấy panel thứ-k-**dệt-xong** của bucket, **không** cứng theo số thứ tự (index). Mặc định bật qua `enable_fifo_linking_floor`. **Xem giải thích dài + ví dụ ở §6.2.** `phase2_linking.py:566` (`compute_sameqty_start_lb`).
- **shift / `shift_ends_min`** — thời-gian-ảo (backend đã cắt giờ nghỉ); washing không được vắt qua mốc. `request_schema.py:107-110`.
- **pinned / in-progress** — task có máy/giờ cố định; in-progress chỉ 1 đầu thì suy đầu kia. `shared.py:561`.

### Kỹ thuật
- **FEASIBLE vs OPTIMAL** — solver có thể dừng ở FEASIBLE trong det-budget → **term phụ chưa được tối ưu** (xem §8.3).
- **deterministic-time** — đơn vị dừng tất định, không phải wall-clock; trên model khó có thể chậm ~10-15× wall. `shared.py:459-461`.
- **post-pass** — bước chỉnh **tất định** sau solve (left-shift, spread, flush, contiguity-reorder, parallel-PO, relink).
- **warm-start / AddHint** — gợi ý lời giải, **không thêm ràng buộc/objective**. **EDD** = earliest-due-date.
- **relative_gap** — dung sai tối ưu; mặc định `0.01`, đặt `0.0` cho phase rẻ cần đóng cải thiện <1%. `shared.py:501-503`.
- **reschedule_hint / hard-keep / stability** — cơ chế giữ kế hoạch cũ để lịch không "nhảy".

---

## 6. CATALOG RÀNG BUỘC THEO PHASE  ← phần lõi

Mỗi mục: **(a)** diễn giải — **(b)** `file:line` + code — **(c)** giải quyết vấn đề gì.

### 6.0 Shared — `build_resource_model` (`shared.py:627`) — phủ linking, downstream, phần resource mọi phase
- (a) **Thời lượng cố định**: end = start + duration. (b) `shared.py:718` `model.Add(end_var == start_var + duration_val)`. (c) ràng buộc cơ bản của một task.
- (a) **Release**: start ≥ thời điểm sẵn sàng / `start_after`. (b) `shared.py:722` `model.Add(start_var >= effective_lb)`; per-máy `:805` `…>= available_at).OnlyEnforceIf(is_selected)`. (c) máy/vật liệu chưa sẵn sàng thì chưa chạy.
- (a) **Routing — mỗi task đúng 1 máy**. (b) `shared.py:831` `model.AddExactlyOne(literals)`. (c) chọn 1 trong các máy tương thích.
- (a) **Không chồng / sức chứa**: máy serial không chồng, máy batch theo capacity. (b) `shared.py:857` `model.AddNoOverlap(ivs)` (cap=1) / `:854` `model.AddCumulative(ivs, demands, cap)` (cap>1). (c) chống 2 task cùng máy cùng lúc; giới hạn mẻ.
- (a) **Bảo trì máy chiếm trọn capacity** trong cửa sổ `unavailability`. (b) `shared.py:840` + interval thêm vào. (c) không xếp việc lúc máy bảo trì.

### 6.1 Phase 1 — Knitting (`phase1_knitting.py`)
- (a) **Tổng năng lực máy dệt**. (b) `:1803` `model.AddCumulative(knitting_intervals, demands, MAX_MACHINES)`. (c) không vượt số máy dệt đồng thời.
- (a) **Slot exactly-one + per-slot no-overlap**. (b) `:1837` `AddExactlyOne(slot_bools)`, `:1846` `AddNoOverlap(slot_intervals[s])`. (c) gán workforce/slot dệt.
- (a) **Material/creel cumulative** — *(ĐANG DISABLED ở caller, xem §9 G5)*. (b) `:1908` `model.AddCumulative(intervals, demands, int(capacity))` nhưng `_apply_material_constraints` bị comment tại `:448-451`. (c) đáng lẽ giới hạn sợi/creel; hiện không enforce.
- (a) **PO-active reify** phục vụ parallel-PO + contiguity. (b) `:1967`/`:2037` `AddMaxEquality(po_active/po_on_m, lits)`. (c) biết PO nào "động"/chạm máy nào.

### 6.2 Phase 2 — Linking (`phase2_linking.py`)
- Dùng `build_resource_model` (§6.0). Objective tại `:465`.

#### FIFO-by-PO floor — giải thích thường (đọc cái này trước, công thức ở dưới)

**Bối cảnh.** Một sản phẩm (vd cái áo) gồm nhiều **panel** (thân trước, thân sau…). Mỗi panel
được dệt bằng **một PO knitting riêng** — ví dụ thân-trước = PO `643`, thân-sau = PO `644`.
Công đoạn **linking** là ráp các panel lại với nhau; mỗi **slice** (một lát việc linking) cần
**một panel của MỖI PO** thì mới ráp được (1 thân-trước + 1 thân-sau).

**Vấn đề của cách cũ ("ghép theo số thứ tự" / index pairing).** Phía Go đánh số và ghép cứng:
slice số **k** phải đợi đúng **panel số k của 643 VÀ panel số k của 644**. Nhưng:
- các panel **cùng một PO là giống hệt nhau, đổi cho nhau được** (thân-trước nào cũng như nhau);
- máy dệt **không** dệt xong theo đúng thứ tự số.

→ Hậu quả: slice số 1 ngồi chờ "panel 643 *số 1*"; lỡ đúng panel đó dệt xong muộn (vd phút 1076)
thì slice kẹt ở đó, **dù** panel 643 *số 4* đã dệt xong từ phút 255 và hoàn toàn dùng thay được.

**Cách mới (FIFO-by-PO).** Slice số k chỉ cần **panel thứ-k-DỆT-XONG** của mỗi PO (bất kể nó
mang số mấy), thay vì đúng "panel số k". → các slice ở giữa được chạy ngay khi đủ panel, không
phải chờ một panel-số-cụ-thể về muộn. **Slice cuối cùng vẫn chờ panel cuối cùng**, nên
**giờ-hoàn-thành của đơn KHÔNG đổi** — chỉ các slice giữa được khởi động sớm hơn.
("FIFO" = first-in-first-out: panel nào dệt xong trước thì được dùng trước.)

**Ví dụ thật** (order `WLJPELMsPp`, log `1782783`, 2 PO `643`/`644`):

| | thời điểm các slice linking bắt đầu |
|---|---|
| cách cũ (index) | `255, 407, 407, `**`1076`**`, 2756` — lúc phút 255 **chỉ 1 slice** chạy được |
| cách mới (FIFO) | `255, `**`255`**`, 407, 407, 2756` — **2 slice** chạy lúc 255; slice "1076" được kéo về 255 |

Slice cuối ở cả hai vẫn là `2756` (panel `644` thật sự dệt xong muộn) ⇒ đơn xong cùng lúc.

#### Chi tiết kỹ thuật

- (a) **Cài đặt:** floor này áp qua `start_lb` (= giới-hạn-dưới của thời điểm slice được phép bắt đầu).
  **Mặc định BẬT** — cờ `enable_fifo_linking_floor=True` (`request_schema.py:223`).
  (b) `phase2_linking.py:566` `def compute_sameqty_start_lb` tính floor; chọn dùng nó tại `:397`
  (`elif config.get("enable_fifo_linking_floor", True) and all_pipeline_tasks is not None`).
  (c) slice không bị kẹt sau một panel-số về muộn trong khi panel anh-em (cùng PO) đã sẵn sàng.
- (a) **Khi nào quay về floor cũ (index):** nếu **tắt cờ**, HOẶC không truyền `all_pipeline_tasks`
  (thiếu metadata `group_id`/`qty` của knitting để biết panel nào cùng PO). (b) nhánh `else` tại `:402`
  gọi `_compute_start_lb` (`:506`, ghép cứng theo số thứ tự). (c) an toàn ngược: tắt cờ → kết quả
  giống y hệt cách cũ (byte-identical).
- (a) **QUAN TRỌNG — floor này chạy trong CẢ 2 lượt của double-solve (§2.1/§8.4).** Trước đây việc
  "đổi panel" chỉ được làm bằng **post-pass** (left-shift), mà post-pass **bị tắt ở lượt 2** (lượt 2 là
  re-schedule, §8.4) → lượt 2 dựng lại floor cũ → **lịch trả về UI mất lợi ích FIFO**. Đưa FIFO thẳng
  vào solver (qua `start_lb`) làm nó "sống" ở cả lượt 1 lẫn lượt 2. (c) vá đúng lỗ hổng "post-pass chỉ
  chạy lượt cold" gặp double-solve. Xem memory `project_fifo_linking_floor_insolver`
  (bản post-pass cũ: `project_linking_fifo_po_floor`).
- **Vì sao an toàn 100%:** floor FIFO **luôn ≤** floor index (chỉ nới sớm, không bao giờ đẩy muộn) ⇒
  miền khả thi của solver chỉ rộng ra ⇒ lời giải chỉ tốt-hơn-hoặc-bằng. Test: `test_slice_interleaving.py::TestFifoLinkingFloor`.

### 6.3 Phase 3 — Washing (`phase3_batching.py`)
- (a) **Mỗi washing task vào đúng 1 slot**. (b) `:356` `AddExactlyOne(x[t_id])`.
- (a) **Batch-active reify**. (b) `:372` `AddMaxEquality(batch_active[k], …)`.
- (a) **Slot ↔ máy tương đương** (slot chạy trên đúng 1 máy, cho phép task co-located song song). (b) `:490-491` `AddImplication(...)` (hai chiều). (c) commit `f185c3c`: mỗi batch-slot chạy trên đúng 1 máy.
- (a) **Máy không chồng batch**. (b) `:639` `AddNoOverlap(machine_batch_ivs)`.
- (a) **Co-location reify** (2 task cùng slot ⇒ giặt chung). (b) `:883-887` `AddBoolAnd/AddBoolOr … AddMaxEquality(co, same_slots)`.
- (a) **Flush cuối ca (post-pass đã ship)**: gom bán-thành-phẩm linking đã xong nhưng bị xếp giặt ca sau, kéo vào batch **kết thúc đúng mốc hết ca** (start = T − duration); không kịp thì để ca/hôm sau. (b) `:1296` `def flush_unwashed_end_of_shift`, docstring `:1303-1307` *"…pulled into a flush batch that ENDS exactly at that boundary … cứ cuối ca là đem đi giặt, không vắt qua giờ nghỉ."* (c) đồ không nằm chờ qua đêm. An toàn vì chỉ-dịch-sớm (§8.5).

### 6.4 Phase 4 — Downstream: Ironing + Packing (`phase4_downstream.py`)
- Dùng `build_resource_model` (§6.0); gọi `solve_downstream` **2 lần** (iron rồi pack).
- (a) **Precedence chuỗi**: task sau bắt đầu ≥ task trước kết thúc. (b) `:139` `model.Add(task_vars[t_id]["start"] >= task_vars[dep_id]["end"])`. (c) pack sau iron, iron sau wash.

### (tham khảo) `TaskModelBuilder` monolithic (`builder.py`)
Builder cũ vẫn còn (AddCumulative `:545/:886/:920`, AddNoOverlap `:601/:924`, AddExactlyOne `:591/:706/:1237`, AddImplication `:1335` batch monotonic, Minimize `:1420`). Code mới đi qua `phases/*` + `shared.py`.

---

## 7. OBJECTIVE THEO PHASE

**Dạng: tổng-trọng-số** — mỗi phase đúng **một** `model.Minimize(sum(obj_terms))`:
`phase1:617`, `phase2:465`, `phase3:923/:950`, `phase4:157`, builder `:1420`.
**KHÔNG lexicographic / không tiered.** Ưu tiên thể hiện qua **trọng số** (vd `knitting_contiguity_mult`, `stability_weight_machine_swap=50000`, calibration window ở `request_schema.py:303-320`).

**Các term (helper trong `shared.py`)**: lateness/soft-deadline (`:866`), order-flow (`:940`), order-cluster (`:1006`), earliness (`:1152`), slice-sync (`:1193`), panel-sync (`:1334`), stability (`:58`), affinity (trong `build_resource_model`, `use_affinity`).

**Warm-start (AddHint, không phải objective)**: stability/reschedule (`shared.py:159/:277`), batch warm (`phase3:823/:837/:853`), EDD knitting hint.

**Validate**: mỗi lever có test riêng — `test_panel_sync.py`, `test_short_term_deadline.py`, `test_linking_balance.py`, `test_slice_interleaving.py`, `test_normal_order_lateness.py`.

---

## 8. MÔ HÌNH TƯ DUY CỐT LÕI (điều không tự suy ra từ code)

**8.1 Determinism = (1 worker) + (`PYTHONHASHSEED=0`) — một cặp đôi.**
> `shared.py:432-443`: *"num_search_workers > 1 shares bounds … by WALL-CLOCK timing … 155 vs 129 late orders across runs at 8 workers; byte-identical at 1 worker"*; *"PYTHONHASHSEED=0 … without it, even 1 worker is non-deterministic because the MODEL itself differs run-to-run."* + `Dockerfile:12`, commit `3945bc3`.

**8.2 Determinism ≠ Stability.** Determinism = cùng input → cùng output (lo bởi 1-worker). Stability = lịch lần-1 ≈ lần-2 / sau reschedule (lo bởi double-solve + hard-keep). Hai chuyện khác nhau.
> `solver_task.py:133-141`; commit `be46a8e`.

**8.3 Solver kẹt FEASIBLE → dùng post-pass, KHÔNG tăng trọng số.** Khi solver dừng ở FEASIBLE trong det-budget, term phụ (contiguity, earliness, left-shift) **không bao giờ được tối ưu**. Sửa bằng post-pass tất định.
> `pipeline.py:180-182`: *"the solver stalls at FEASIBLE so its secondary contiguity term never optimises; re-sequence each machine…"*; `pipeline.py:231`: *"washing itself stalls at FEASIBLE…"*

**8.4 Mọi post-pass là COLD-only (`if not reschedule_hint`).** Đường reschedule phải giữ ổn định; post-pass sẽ tái-tạo drift. Vì double-solve biến pass-2 thành reschedule, post-pass chỉ chạy ở pass-1.
> Gating lặp lại: `pipeline.py:186`, `:202-205`, `:340` `if not self.reschedule_hint`.

**8.5 An toàn post-pass = "chỉ-dịch-sớm ⇒ miền khả thi là superset".** Transform chỉ kéo task **sớm hơn** (flush, left-shift) → release hạ nguồn chỉ NỚI → optimum không thể xấu đi → **không cần re-solve**. Transform **không** monotone (parallel-PO, contiguity-reorder) thì phải **re-solve + accept-if-lateness-không-tăng**.
> `phase3_batching.py:1309-1312`: *"flush ONLY moves washing EARLIER … the phase 4–5 feasible region is a superset → its optimum cannot get worse"*; tương phản `pipeline.py:183-185`: *"Neither is monotone … accepted ONLY if total pipeline lateness does not increase."*

**8.6 Budget là deterministic-time, KHÔNG phải wall-time.** `max_deterministic_time` là tiêu chí dừng **duy nhất**; `max_time_in_seconds` cố tình bỏ trống (wall cap = non-deterministic, fire ở node phụ thuộc tốc-độ-máy).
> `shared.py:445-449`, `:470-471`.

**8.7 Tài liệu `.md` đã lệch code — luôn verify.** Ví dụ: `CLAUDE.md` ghi *"No CP-SAT calls outside builder.py"* nhưng thực tế CP-SAT có ở `shared.py` + cả 4 `phases/*.py`. Đừng tin mô tả trong `.md`; mở code.

---

## 9. DECISION LOG — "TẠI SAO KHÔNG LÀM X" (chặn đi lại đường cũ)

**G1. KHÔNG bật `enable_identical_symmetry_break`** (default OFF). Tăng tốc phase-1 nhưng **phá hạ nguồn ~6×**: iron/packing tardiness **13.523 → 79.107** trên payload 240-đơn-giống-nhau (wall chỉ giảm 54→29 phút). Lý do: pipeline tuần tự — mỗi batch "giống nhau" feed một đơn KHÁC.
> `phase1_knitting.py:355-366`, commit `4af5699`.

**G2. KHÔNG tăng `num_search_workers` > 1.** 8 workers cho **155 vs 129 late orders** giữa các lần chạy (cùng seed); byte-identical ở 1 worker. `make_solver` ép `=1`.
> `shared.py:432-436`, `:480`, commit `3945bc3`.

**G3. KHÔNG set `repair_hint=True`.** Gây `Check failed: heuristics.fixed_search != nullptr` SIGABRT trên CP-SAT 9.8+. AddHint warm-start đã đủ.
> `shared.py:507-510`.

**G4. KHÔNG set `max_time_in_seconds` (wall cap).** Dừng theo wall-clock tái-tạo drift ngay cả khi chỉ là "safety cap".
> `shared.py:445-449`, `:470-471`.

**G5. KHÔNG re-enable material/creel enforcement phase-1 nguyên trạng.** Đang tắt có chủ đích: material capacity=0 (vs demand>0) làm `AddCumulative` INFEASIBLE; và nó chặn mục tiêu parallel-knitting. 3 test enforcement đã `skip` (không xoá). Re-enable cần thêm guard `capacity<=0`.
> Code comment `phase1_knitting.py:448-451` *"Temporarily disabled per user request"*; `tests/test_material_constraints.py:24-32`; commit `0b9019c`.

**G6. KHÔNG trông cậy `enable_sameqty_relink` để có lợi** (schema default `False`, effective OFF qua API). Đo: 0/9 chấp nhận, tốn 16-21% wall-time; EDD hint + panel-sync đã làm pass-1 linking đủ tốt nên pass-2 chỉ regress.
> `request_schema.py:116-121`; ⚠️ pipeline `.get(...,True)` (`:205`) — khác đường vào (xem §4).

**G7. Flush làm bằng post-pass, KHÔNG in-solver.** Vì chỉ-dịch-sớm nên không cần re-solve (§8.5). *(Lưu ý: MEMORY có ghi "in-solver flush thử & revert" nhưng KHÔNG truy được git — `git log --all` chỉ có 1 commit flush, là post-pass — nên không khẳng định ở đây; xem §12.)*
> `phase3_batching.py:1294-1315`.

---

## 10. DEBUG ĐẶC THÙ DỰ ÁN

- **Replay log thật:** `logs/solver_input_CP_*.json` + `solver_output_CP_*.json` (input/output từng job). Đây là payload thật để tái hiện & đo.
- **`MOCK_RESPONSE_FILE`:** set env → bỏ qua solver, trả thẳng file. `solver_task.py:114`. Dùng để test tích hợp Go mà không solve lại.
- **`tests/reproduce_issue.py`:** khung repro tối thiểu, dựng `TaskModelBuilder`, `num_search_workers:1`.
- **Kiểm determinism:** chạy 2 lần với **1 worker + `PYTHONHASHSEED=0`**, diff output. Test sẵn: `test_determinism.py`, `test_input_order_determinism.py`.
- **Sweep det-budget:** chỉnh `max_deterministic_time` để phân biệt "kẹt FEASIBLE" (post-pass) vs "cần thêm thời gian". Thư mục thí nghiệm: `det_study/`.

---

## 11. BẤT BIẾN PHẢI-GIỮ Ở MỌI THAY ĐỔI (verify từ code/test, KHÔNG từ `.md` stale)

1. **int-only**: mọi bound/penalty CP-SAT là `int`, không `float`.
2. **CP-SAT giới hạn trong tầng engine** (`builder.py` + `shared.py` + `phases/*.py`). *(Sửa lại từ câu stale "chỉ builder.py" của CLAUDE.md — xem §8.7.)*
3. **Tên field schema = Go wire contract** (alias) — KHÔNG đổi tên (`request_schema.py`, `response_schema.py`).
4. **OR-Tools pin** `ortools==9.8.3296` (`requirements.txt:6`) — KHÔNG đổi.
5. **`make_solver` ép 1 worker** — KHÔNG nâng (determinism).
6. **Post-pass phải monotone (chỉ-dịch-sớm) HOẶC re-solve + accept-if-no-regression** (§8.5).
7. **Comment tiếng Việt trong `builder.py`/`phase*.py`** là tài liệu domain — KHÔNG xoá.
8. **Regression guard (chạy trước khi merge):** `test_machine_no_overlap.py`, `test_frozen_pin_overlap.py`, `test_pinned_vs_free.py`, `test_reschedule_stability.py`, `test_determinism.py`.

---

## 12. VIỆC ĐANG MỞ / NỢ KỸ THUẬT

- **3 param mismatch schema↔runtime** (§4) — đã xác nhận là **chủ đích, test-validated** (2026-06-30), không phải bug; ghi lại để dev mới khỏi "sửa cho khớp".
- **Material/creel enforcement phase-1 DISABLED** + 3 test `skip` (§9 G5). Re-enable cần guard `capacity<=0`.
- **`CLAUDE.md` invariant "No CP-SAT outside builder.py" đã STALE** (§8.7, §11.2) — nên cập nhật CLAUDE.md (ngoài phạm vi tài liệu này).
- **Docstring `RescheduleHint` mâu thuẫn code:** `request_schema.py:317` nói *"with solver `repair_hint=True` in make_solver…"* nhưng `make_solver` KHÔNG set `repair_hint` (SIGABRT, `shared.py:507-510`). Docstring stale.
- **G7 — claim "in-solver washing flush thử & revert"** (từ MEMORY) **không truy được git/code** → chưa xác nhận; chỉ ghi nhận.
- **Branch:** toàn bộ scheduler ở `dev`, **chưa merge `main`** (main đứng ở baseline 2026-04-11). Việc merge/đồng bộ main là việc mở.
- **Re-schedule phía Go:** theo `docs/Reschedule-Integration-Guide.md` (DOC — chưa tin), *"Go side needs to adopt"* — **chưa verify được từ code**, cần hỏi đội Go.

---

*Hết. Mọi line-ref trong tài liệu đã được mở & đối chiếu nguyên văn (Gate VERIFY, 0 dòng lệch). Khi nghi ngờ: tin code + git, không tin `.md`.*
