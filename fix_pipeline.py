import re

with open("app/engine/pipeline.py", "r") as f:
    content = f.read()

# 1. Remove left_shift_cold_knitting in _try_knitting_relayout
content = re.sub(
    r"(\s*)if left_shift_cold_knitting\(p1\.assignments, self\.tasks, self\.config\):"
    r"(\s*)# Keep p1's time maps.*?"
    r"(\s*)for a in p1\.assignments:"
    r"(\s*)p1\.start_times\[a\[\"task_id\"\]\] = a\[\"start_time\"\]"
    r"(\s*)p1\.end_times\[a\[\"task_id\"\]\] = a\[\"end_time\"\]",
    "",
    content,
    flags=re.DOTALL
)

# 2. Remove left_shift_cold_washing block in run()
content = re.sub(
    r"(\s*)# Washing left-shift, FINAL pass[^\n]*\n"
    r"(?:\s*#[^\n]*\n)*"
    r"(\s*)if not self\._stabilize_pass and self\.config\.get\(\"enable_washing_left_shift\", True\):"
    r".*?(?=\s*# Iron/packing worker load-balance)",
    "\n",
    content,
    flags=re.DOTALL
)

# 3. Remove hole-closing left-shift AFTER the balance in run()
content = re.sub(
    r"(\s*)# Hole-closing left-shift AFTER the balance:.*?"
    r"(\s*)if changed:"
    r".*?(?=\s*all_overloads =)",
    "\n",
    content,
    flags=re.DOTALL
)

# 4. Remove left_shift_cold_knitting in _improve_knitting
content = re.sub(
    r"(\s*)left_shift_cold_knitting\(p1r\.assignments, self\.tasks, self\.config\)",
    "",
    content
)

# 5. Empty _tighten_linking
content = re.sub(
    r"(def _tighten_linking\([^)]+\)\s*->\s*None:\n"
    r"(?:\s*(?:\"\"\"[^\"]*\"\"\"|#[^\n]*)\n)*"
    r")"
    r"(?:\s*if not self\._apply_cold_passes.*?)(?=\s*def _solve_phases_3_to_5)",
    r"\1        pass\n\n",
    content,
    flags=re.DOTALL
)

# 6. Remove flush_unwashed_end_of_shift and left_shift_cold_washing in _solve_phases_3_to_5
content = re.sub(
    r"(\s*)# ── End-of-shift washing flush.*?if not self\._stabilize_pass and self\.config\.get\(\"enable_washing_left_shift\", True\):.*?p3\.end_times\[a\[\"task_id\"\]\] = a\[\"end_time\"\]\n",
    "\n",
    content,
    flags=re.DOTALL
)

# 7. Remove left_shift_cold_ironing and fifo_swap_ironing in _solve_phases_4_5
content = re.sub(
    r"(\s*)# Tighten ironing BEFORE packing solves:.*?"
    r"(\s*)if moved:"
    r"(\s*)p4\.end_times = \{a\[\"task_id\"\]: a\[\"end_time\"\] for a in p4\.assignments\}\n",
    "\n",
    content,
    flags=re.DOTALL
)

# 8. Remove left_shift_cold_packing in _solve_phases_4_5
content = re.sub(
    r"(\s*)# Tighten packing: same FEASIBLE-stall as ironing/linking.*?"
    r"(\s*)if moved:"
    r"(\s*)p5\.end_times = \{a\[\"task_id\"\]: a\[\"end_time\"\] for a in p5\.assignments\}\n",
    "\n",
    content,
    flags=re.DOTALL
)

# 9. Clean up imports (optional but good practice)
content = re.sub(r"\s*left_shift_cold_knitting,?\n", "\n", content)
content = re.sub(r"\s*left_shift_cold_linking,?\n", "\n", content)
content = re.sub(r"\s*left_shift_cold_washing,?\n", "\n", content)
content = re.sub(r"\s*flush_unwashed_end_of_shift,?\n", "\n", content)
content = re.sub(r"\s*left_shift_cold_ironing,?\n", "\n", content)
content = re.sub(r"\s*fifo_swap_ironing,?\n", "\n", content)
content = re.sub(r"\s*left_shift_cold_packing,?\n", "\n", content)

with open("app/engine/pipeline.py", "w") as f:
    f.write(content)
