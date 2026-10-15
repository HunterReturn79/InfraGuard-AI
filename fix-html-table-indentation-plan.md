# Fix: HTML Table Rendering as Raw Code Block in Tab 1

## Top-Level Overview

**Goal:** Fix the "Grid & Plant Infrastructure Health Matrix" HTML table in Tab 1 of `app.py`, which renders as a raw code block instead of a styled table because Streamlit's Markdown parser treats lines with 4+ leading spaces as a fenced code block.

**Scope:** The `render_executive_matrix()` function in `app.py` (lines 596–659). No other tabs or functions are in scope.

**Root Cause:** In `render_executive_matrix()`, the `rows_html` f-string (built per-row inside a loop) and the `full_html` f-string both carry Python indentation whitespace — each line starts with 8–16 spaces. When passed to `st.markdown(..., unsafe_allow_html=True)`, Markdown interprets any block of lines with 4+ leading spaces as a code block, so the raw HTML tags appear instead of the rendered table.

**Approach:** Strip the common leading indentation from `full_html` using `textwrap.dedent()` before passing it to `st.markdown()`. This is the minimal, surgical fix — no structural refactor, no switch to `st.dataframe()` needed.

---

## Sub-Tasks

### Sub-Task 1 — Add `textwrap` import
- **Intent:** `textwrap.dedent()` is the standard library function needed for the fix. It is not currently imported in `app.py`.
- **Expected Outcomes:** `import textwrap` appears in the import block at the top of `app.py`.
- **Todo List:**
  - [ ] Add `import textwrap` after the existing `import html` on line 21 of `app.py`.
- **Relevant Context:** `app.py` line 21 — `import html` is the last import in the stdlib block.
- **Status:** `[ ] pending`

---

### Sub-Task 2 — Fix `rows_html` per-row indentation in `render_executive_matrix`
- **Intent:** Each row's HTML string is built with 8-space leading indentation from Python. This indentation accumulates and contributes to the code-block triggering whitespace. Stripping it at construction time keeps the final HTML clean.
- **Expected Outcomes:** Each `<tr>...</tr>` block appended to `rows_html` starts at column 0 (no leading spaces), so when embedded into `full_html` it does not introduce problematic whitespace.
- **Todo List:**
  - [ ] Wrap the per-row f-string in `textwrap.dedent()` before appending to `rows_html` (line 616 of `app.py`), OR use a `textwrap.dedent()` call on the final `full_html` string (see Sub-Task 3 — doing it once on `full_html` is sufficient).
- **Relevant Context:** `app.py` lines 616–634 — the `rows_html +=` f-string inside the `for` loop.
- **Status:** `[ ] pending`

---

### Sub-Task 3 — Dedent `full_html` before passing to `st.markdown()`
- **Intent:** `full_html` is the complete HTML string (wrapper div + table + injected rows). All its lines carry Python indentation. Applying `textwrap.dedent(full_html)` removes the common leading whitespace from every line in one operation, eliminating the Markdown code-block misinterpretation.
- **Expected Outcomes:** `st.markdown(textwrap.dedent(full_html), unsafe_allow_html=True)` correctly renders the styled glassmorphism table instead of raw `<tr><td>...` text.
- **Todo List:**
  - [ ] Change line 659 in `app.py` from `st.markdown(full_html, unsafe_allow_html=True)` to `st.markdown(textwrap.dedent(full_html), unsafe_allow_html=True)`.
- **Relevant Context:** `app.py` line 659 — `st.markdown(full_html, unsafe_allow_html=True)`.
- **Status:** `[ ] pending`

---

### Sub-Task 4 — Verify `render_dispatch_matrix` does not have the same issue
- **Intent:** `render_dispatch_matrix()` (lines 661–705) uses the same pattern. Confirm whether it also suffers from the same indentation bug and apply the same fix if needed, to maintain consistency.
- **Expected Outcomes:** Both `render_executive_matrix` and `render_dispatch_matrix` use `textwrap.dedent()` on their `full_html` before passing to `st.markdown()`.
- **Todo List:**
  - [ ] Read `render_dispatch_matrix` and confirm the same indentation pattern exists.
  - [ ] Apply `textwrap.dedent()` to its `st.markdown()` call (line 705) if confirmed.
- **Relevant Context:** `app.py` lines 661–705 — `render_dispatch_matrix()` function.
- **Status:** `[ ] pending`
