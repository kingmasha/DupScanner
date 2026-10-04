#!/usr/bin/env python3
"""
Duplicate report for the RX 5700 XT.

Only JPG, JPEG, PNG, MOV, MP4, and ZIP are scanned. Images and videos
outside AppData are listed first, grouped by type. AppData copies are
a separate section at the bottom.
"""

import os
import sys
import string
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

SKIP_DIRS = {
    "$recycle.bin", "system volume information", "recovery",
    "windows", "program files", "program files (x86)",
    "proc", "sys", "dev", "run", "snap",
    ".git", "node_modules", "__pycache__",
}
WANTED_EXT = {".jpg", ".jpeg", ".png", ".mov", ".mp4", ".zip"}
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".mov", ".mp4"}
TYPE_ORDER = {".mp4": 0, ".mov": 1, ".jpg": 2, ".jpeg": 3, ".png": 4, ".zip": 5}
TYPE_LABELS = {
    ".mp4": "MP4",
    ".mov": "MOV",
    ".jpg": "JPG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".zip": "ZIP",
}
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)

PROBE_SRC = r"""
__kernel void probe(__global const int *a, __global const int *b, __global int *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i] + 7;
}
"""

SORT_SRC = r"""
__kernel void bitonic_pass(
    __global long *sizes,
    __global long *times,
    __global int *idx,
    int j,
    int k
) {
    int i = get_global_id(0);
    int ixj = i ^ j;
    if (ixj <= i) return;

    int ascending = ((i & k) == 0);
    int greater = sizes[i] > sizes[ixj]
        || (sizes[i] == sizes[ixj] && times[i] > times[ixj]);
    int do_swap = ascending ? greater : !greater;
    if (!do_swap) return;

    long ts = sizes[i]; sizes[i] = sizes[ixj]; sizes[ixj] = ts;
    long tt = times[i]; times[i] = times[ixj]; times[ixj] = tt;
    int ti = idx[i]; idx[i] = idx[ixj]; idx[ixj] = ti;
}
"""

MARK_SRC = r"""
__kernel void mark_dups(
    __global const long *sizes,
    __global const long *times,
    __global const int *idx,
    __global uchar *dup,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    int prev = (i > 0)
        && sizes[i] == sizes[i - 1]
        && times[i] == times[i - 1];
    int next = (i + 1 < n)
        && sizes[i] == sizes[i + 1]
        && times[i] == times[i + 1];
    if (prev || next)
        dup[idx[i]] = 1;
}
"""


def fail(msg):
    print(msg, file=sys.stderr)
    print("GPU is not working for this script. Stopping.", file=sys.stderr)
    sys.exit(1)


def require_gpu():
    try:
        import pyopencl as cl
        import numpy as np
    except ImportError as exc:
        fail(
            f"GPU check failed: {exc}.\n"
            "RX 5700 XT is AMD, so CuPy/CUDA will not see it.\n"
            "Install the AMD Adrenalin driver, then: pip install pyopencl numpy"
        )

    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        fail(f"OpenCL runtime failed: {exc}")
    if not platforms:
        fail("No OpenCL platforms. Install/repair the AMD Adrenalin driver.")

    gpus = []
    for platform in platforms:
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
        except Exception:
            continue
        for device in devices:
            gpus.append((platform, device))
    if not gpus:
        fail("No OpenCL GPU device.")

    def rank(item):
        _platform, device = item
        name = f"{device.vendor} {device.name}".lower()
        score = 0
        if "5700" in name or "gfx1010" in name:
            score += 4
        if "amd" in name or "advanced micro devices" in name:
            score += 2
        return score

    platform, device = max(gpus, key=rank)
    label = f"{device.name} | {device.vendor} | {platform.name} | OpenCL {device.version}"
    if "gfx1010" not in device.name.lower() and "5700" not in device.name.lower():
        fail(f"Refusing to run: selected device is not the RX 5700 XT ({label}).")

    try:
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        program = cl.Program(ctx, PROBE_SRC).build()
        a = np.arange(256, dtype=np.int32)
        b = np.arange(256, dtype=np.int32) * 3
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a.nbytes)
        program.probe(queue, a.shape, None, a_buf, b_buf, c_buf)
        out = np.empty_like(a)
        cl.enqueue_copy(queue, out, c_buf)
        queue.finish()
    except Exception as exc:
        fail(f"GPU was enumerated ({label}) but the probe kernel failed: {exc}")

    if not np.array_equal(out, a + b + 7):
        fail(f"GPU probe returned wrong results on {label}.")

    print(f"GPU working: {label}")
    print("Probe kernel passed on the device.")
    return ctx, queue


def gpu_duplicate_mask(ctx, queue, sizes, ctimes):
    import pyopencl as cl
    import numpy as np

    n = len(sizes)
    if n < 2:
        return [False] * n

    padded = 1
    while padded < n:
        padded <<= 1

    size_arr = np.full(padded, np.iinfo(np.int64).max, dtype=np.int64)
    time_arr = np.full(padded, np.iinfo(np.int64).max, dtype=np.int64)
    idx_arr = np.arange(padded, dtype=np.int32)
    size_arr[:n] = np.asarray(sizes, dtype=np.int64)
    time_arr[:n] = np.asarray(ctimes, dtype=np.int64)

    mf = cl.mem_flags
    size_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=size_arr)
    time_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=time_arr)
    idx_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=idx_arr)
    dup_buf = cl.Buffer(ctx, mf.READ_WRITE, n)
    cl.enqueue_copy(queue, dup_buf, np.zeros(n, dtype=np.uint8))

    sort_prog = cl.Program(ctx, SORT_SRC).build()
    mark_prog = cl.Program(ctx, MARK_SRC).build()
    k = 2
    while k <= padded:
        j = k // 2
        while j > 0:
            sort_prog.bitonic_pass(
                queue, (padded,), None,
                size_buf, time_buf, idx_buf,
                np.int32(j), np.int32(k),
            )
            j //= 2
        k *= 2
    mark_prog.mark_dups(
        queue, (n,), None,
        size_buf, time_buf, idx_buf, dup_buf, np.int32(n),
    )
    dup = np.empty(n, dtype=np.uint8)
    cl.enqueue_copy(queue, dup, dup_buf)
    queue.finish()
    return dup.astype(bool).tolist()


def get_roots():
    if sys.platform == "win32":
        return [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    return ["/"]


def creation_time(st):
    if hasattr(st, "st_birthtime"):
        return int(st.st_birthtime)
    return int(st.st_ctime)


def in_appdata(path):
    parts = os.path.normpath(path).lower().split(os.sep)
    return "appdata" in parts


def scan_tree(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in WANTED_EXT:
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path, follow_symlinks=False)
            except (OSError, PermissionError):
                continue
            found.append((name.lower(), st.st_size, ext, creation_time(st), path))
    return found


def find_duplicates(ctx, queue):
    roots = get_roots()
    print(f"Scanning {', '.join(roots)} for {', '.join(sorted(WANTED_EXT))}")
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_tree, root): root for root in roots}
        for fut in as_completed(futures):
            root = futures[fut]
            try:
                part = fut.result()
            except Exception as exc:
                print(f"[warn] {root}: {exc}")
                continue
            rows.extend(part)
            print(f"Finished {root} ({len(part)} matching files)")

    if not rows:
        return {}

    print(f"Matching files examined: {len(rows)}")
    print("Grouping on GPU.")
    try:
        mask = gpu_duplicate_mask(ctx, queue, [r[1] for r in rows], [r[3] for r in rows])
    except Exception as exc:
        fail(f"GPU grouping failed after the probe passed: {exc}")
    print("GPU grouped numeric keys (size, created).")

    groups = defaultdict(list)
    for row, keep in zip(rows, mask):
        if not keep:
            continue
        name, size, ext, ctime, path = row
        groups[(name, size, ext, ctime)].append(path)
    return {k: v for k, v in groups.items() if len(v) > 1}


def open_path(path):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - user asked to inspect their own files
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


class ReportApp(tk.Tk):
    def __init__(self, dupes):
        super().__init__()
        self.title("Duplicate media report")
        self.geometry("1040x720")
        self.dupes = dupes
        self.path_by_iid = {}
        self.type_filter = tk.StringVar(value="All types")

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.status = ttk.Label(top, text="")
        self.status.pack(side="left")
        ttk.Label(top, text="Show").pack(side="left", padx=(16, 4))
        pick = ttk.Combobox(
            top,
            textvariable=self.type_filter,
            values=("All types", "MP4", "MOV", "JPG", "JPEG", "PNG", "ZIP"),
            state="readonly",
            width=12,
        )
        pick.pack(side="left")
        pick.bind("<<ComboboxSelected>>", lambda _e: self.fill())
        ttk.Button(top, text="Inspect", command=self.inspect_selected).pack(side="right")
        ttk.Button(top, text="X Delete", command=self.delete_selected).pack(side="right", padx=(0, 6))

        cols = ("type", "name", "size", "created", "path")
        self.tree = ttk.Treeview(self, columns=cols, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Section")
        self.tree.column("#0", width=180, anchor="w")
        for col, title, width in (
            ("type", "Type", 60),
            ("name", "Name", 180),
            ("size", "Size", 90),
            ("created", "Created", 150),
            ("path", "Path", 420),
        ):
            self.tree.heading(col, text=title, command=lambda c=col: self.sort_by(c))
            self.tree.column(col, width=width, anchor="w")
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.tree.bind("<Double-1>", lambda _e: self.inspect_selected())
        self.tree.bind("<Delete>", lambda _e: self.delete_selected())
        self.fill()

    def wanted(self, ext):
        choice = self.type_filter.get()
        if choice == "All types":
            return True
        return TYPE_LABELS.get(ext, "") == choice

    def fill(self):
        self.tree.delete(*self.tree.get_children())
        self.path_by_iid.clear()
        media = []
        appdata = []
        for key, paths in self.dupes.items():
            name, size, ext, ctime = key
            if not self.wanted(ext):
                continue
            created = datetime.fromtimestamp(ctime).isoformat(sep=" ", timespec="seconds")
            for path in sorted(set(paths)):
                row = (TYPE_ORDER.get(ext, 9), TYPE_LABELS.get(ext, ext), name, size, created, path)
                if in_appdata(path):
                    appdata.append(row)
                elif ext in MEDIA_EXT:
                    media.append(row)
                else:
                    appdata.append(row)
        media.sort()
        appdata.sort()
        self.add_section("Media duplicates", media)
        self.add_section("AppData and non-media, not prioritized", appdata)
        self.refresh_status()

    def add_section(self, title, rows):
        parent = self.tree.insert("", "end", text=f"{title} ({len(rows)})", open=True)
        for _order, label, name, size, created, path in rows:
            iid = self.tree.insert(
                parent, "end",
                text="",
                values=(label, name, size, created, path),
            )
            self.path_by_iid[iid] = path

    def sort_by(self, col):
        for parent in self.tree.get_children(""):
            kids = list(self.tree.get_children(parent))
            kids.sort(key=lambda iid: str(self.tree.set(iid, col)).lower())
            for index, iid in enumerate(kids):
                self.tree.move(iid, parent, index)

    def selected(self):
        picked = self.tree.selection()
        if not picked or picked[0] not in self.path_by_iid:
            messagebox.showinfo("Nothing selected", "Select a file row first.")
            return None
        return picked[0]

    def inspect_selected(self):
        iid = self.selected()
        if iid is None:
            return
        path = self.path_by_iid[iid]
        if not os.path.exists(path):
            messagebox.showinfo("Missing", f"Already gone:\n{path}")
            return
        win = tk.Toplevel(self)
        win.title(os.path.basename(path))
        win.geometry("760x480")
        st = os.stat(path)
        created = datetime.fromtimestamp(creation_time(st)).isoformat(sep=" ", timespec="seconds")
        meta = f"{path}\nSize: {st.st_size} bytes\nCreated: {created}"
        ttk.Label(win, text=meta, justify="left", wraplength=720).pack(anchor="w", padx=8, pady=8)
        ttk.Button(win, text="Open with default app", command=lambda: open_path(path)).pack(anchor="w", padx=8)
        text = tk.Text(win, wrap="none")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("1.0", "Image, video, or zip. Use 'Open with default app'.")
        text.configure(state="disabled")

    def delete_selected(self):
        iid = self.selected()
        if iid is None:
            return
        path = self.path_by_iid[iid]
        if not os.path.exists(path):
            self.tree.delete(iid)
            self.path_by_iid.pop(iid, None)
            self.refresh_status()
            return
        if not messagebox.askyesno("Delete", f"Delete this file?\n\n{path}"):
            return
        try:
            os.remove(path)
        except OSError as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        self.tree.delete(iid)
        self.path_by_iid.pop(iid, None)
        self.refresh_status()

    def refresh_status(self):
        left = len(self.path_by_iid)
        self.status.configure(text=f"{len(self.dupes)} groups, {left} files shown")


def main():
    ctx, queue = require_gpu()
    dupes = find_duplicates(ctx, queue)
    print(f"Duplicate groups: {len(dupes)}")
    app = ReportApp(dupes)
    app.mainloop()


if __name__ == "__main__":
    main()
