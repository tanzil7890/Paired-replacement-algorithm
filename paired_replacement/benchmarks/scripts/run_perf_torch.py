#!/usr/bin/env python3
import argparse
import os
import platform
import shutil
import subprocess
import sys
import csv
import json
from datetime import datetime
from pathlib import Path


def have_perf() -> bool:
    return shutil.which("perf") is not None and platform.system().lower() == "linux"


def run_perf(events: list[str], cmd: list[str], env: dict | None = None) -> tuple[int, str, str]:
    if env is None:
        env = os.environ.copy()
    perf_cmd = [
        "perf",
        "stat",
        "-x",
        ",",
        "-e",
        ",".join(events),
        "--",
    ] + cmd
    p = subprocess.Popen(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def parse_perf_csv(err_text: str):
    # perf stat -x , produces CSV on stderr; lines: value,unit,event,run_time,*,*
    rows = []
    for line in err_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value, unit, event = parts[0], parts[1], parts[2]
        # Skip non-event lines
        try:
            # values can be like '<not supported>' or '<not counted>'
            v = float(value)
        except ValueError:
            rows.append({"event": event, "value": None, "unit": unit})
            continue
        rows.append({"event": event, "value": v, "unit": unit})
    return rows


def discover_imc_events() -> list[str]:
    """Best-effort discovery of IMC CAS read/write events via `perf list`.
    Returns a list like ["uncore_imc_0/cas_count_read/", ...]. Empty if none.
    """
    try:
        out = subprocess.check_output(["perf", "list"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []
    events: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if "uncore_imc_" in line and ("cas_count_read" in line or "cas_count_write" in line):
            # Extract token that looks like uncore_imc_X/cas_count_YYY/
            # Lines can contain extra annotations; split on spaces and pick the first token-like pattern
            parts = line.replace("(", " ").replace(")", " ").split()
            for p in parts:
                if p.startswith("uncore_imc_") and "/" in p and p.endswith("/"):
                    events.add(p)
    return sorted(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--ks", type=str, default="16,64,256,1024")
    ap.add_argument("--steps_per_k", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--events",
        type=str,
        default="cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,cycles,instructions,task-clock",
    )
    ap.add_argument("--bw_preset", type=str, choices=["none", "auto", "intel"], default="none",
                    help="Add uncore IMC CAS read/write events for bandwidth estimation")
    ap.add_argument("--bytes_per_cas", type=int, default=64, help="Bytes per CAS for IMC counters (typically 64)")
    ap.add_argument("--outfile", type=str, default=str((Path(__file__).parent.parent / "data" / "results_perf_torch.csv").resolve()))
    # Optional: load autotune JSON to set env vars for the C++ parallel knobs
    ap.add_argument("--autotune_json", type=str, default=str((Path(__file__).parent.parent / "configs" / "autotune_result.json").resolve()))
    ap.add_argument("--skip_autotune_json", action="store_true")
    args = ap.parse_args()

    if not have_perf():
        print("perf stat not available on this system (Linux required).", file=sys.stderr)
        sys.exit(1)

    ks = [int(x) for x in args.ks.split(",") if x]
    events = [e for e in args.events.split(",") if e]
    # Bandwidth event augmentation
    bw_events: list[str] = []
    if args.bw_preset == "auto":
        bw_events = discover_imc_events()
        if not bw_events:
            print("Warning: no IMC events discovered via perf list; bandwidth metrics will be unavailable")
    elif args.bw_preset == "intel":
        # Probe a reasonable number of IMCs (0..7). Missing ones will be 'not counted'.
        for i in range(8):
            bw_events.append(f"uncore_imc_{i}/cas_count_read/")
            bw_events.append(f"uncore_imc_{i}/cas_count_write/")
    # Ensure we collect elapsed time for BW normalization (perf prints seconds time elapsed anyway)
    all_events = events + bw_events

    script = str(Path(__file__).parent / "run_microbench_torch.py")

    # Prepare environment, optionally loading autotune JSON
    run_env = os.environ.copy()
    at_path = Path(args.autotune_json)
    autotune_source = None
    if not args.skip_autotune_json and at_path.exists():
        try:
            with open(at_path, "r") as f:
                data = json.load(f)
            g = data.get("grain")
            th = data.get("par_threshold")
            if g is not None and th is not None:
                run_env["WEIGHT_CACHE_GRAIN"] = str(int(g))
                run_env["WEIGHT_CACHE_PAR_THRESHOLD"] = str(int(th))
                autotune_source = str(at_path)
                print(f"Loaded autotune JSON for perf: grain={g}, threshold={th} from {at_path}")
        except Exception as e:
            print(f"Warning: failed to load autotune JSON {at_path}: {e}")

    # Collect rows per (mode, k)
    results = []
    for mode in ("paired", "rebuild"):
        for k in ks:
            cmd = [
                sys.executable,
                script,
                "--N",
                str(args.N),
                "--hidden_dim",
                str(args.hidden_dim),
                "--up_cols",
                str(args.up_cols),
                "--gate_cols",
                str(args.gate_cols),
                "--m",
                str(args.m),
                "--steps_per_k",
                str(args.steps_per_k),
                "--device",
                args.device,
                "--mode",
                mode,
                "--k",
                str(k),
            ]

            code, out, err = run_perf(events, cmd, env=run_env)
            if code != 0:
                print(err)
                raise SystemExit(f"perf run failed for mode={mode}, k={k}")
            rows = parse_perf_csv(err)
            row_dict = {"mode": mode, "k": k}
            for r in rows:
                if r["value"] is not None:
                    row_dict[r["event"]] = r["value"] / args.steps_per_k
                else:
                    row_dict[r["event"]] = None
            # Record parallel knobs if present
            row_dict["grain"] = run_env.get("WEIGHT_CACHE_GRAIN")
            row_dict["par_threshold"] = run_env.get("WEIGHT_CACHE_PAR_THRESHOLD")
            # Derived bandwidth metrics (per-step and per-second)
            # Aggregate CAS counts across IMCs
            read_cas = 0.0
            write_cas = 0.0
            total_seconds = None
            for r in rows:
                ev = r["event"]
                val = r["value"]
                if val is None:
                    continue
                if "cas_count_read" in ev:
                    read_cas += val / args.steps_per_k
                elif "cas_count_write" in ev:
                    write_cas += val / args.steps_per_k
                elif ev == "seconds time elapsed":
                    total_seconds = val
            # Compute bytes per step and BW if available
            dram_read_bytes_per_step = read_cas * args.bytes_per_cas
            dram_write_bytes_per_step = write_cas * args.bytes_per_cas
            row_dict["dram_read_bytes_per_step"] = dram_read_bytes_per_step if (read_cas > 0) else None
            row_dict["dram_write_bytes_per_step"] = dram_write_bytes_per_step if (write_cas > 0) else None
            if total_seconds and total_seconds > 0:
                total_bytes = (dram_read_bytes_per_step + dram_write_bytes_per_step) * args.steps_per_k
                row_dict["dram_bw_bytes_per_sec"] = total_bytes / total_seconds
            else:
                row_dict["dram_bw_bytes_per_sec"] = None
            results.append(row_dict)
            print(f"mode={mode} k={k} collected {len(rows)} counters")

    # Write wide CSV
    # Gather all event names
    all_events = set()
    for r in results:
        all_events.update([k for k in r.keys() if k not in ("mode", "k")])
    fieldnames = [
        "mode", "k", "grain", "par_threshold",
        "dram_read_bytes_per_step", "dram_write_bytes_per_step", "dram_bw_bytes_per_sec",
    ] + sorted(all_events)
    # Banner line with provenance
    banner_parts = [
        f"events={';'.join(events)}",
        f"device={args.device}",
        f"autotune_source={autotune_source}" if autotune_source else None,
        f"N={args.N}", f"hidden_dim={args.hidden_dim}", f"up_cols={args.up_cols}", f"gate_cols={args.gate_cols}", f"m={args.m}",
        f"steps_per_k={args.steps_per_k}", f"ks={','.join(map(str, ks))}",
        f"python={platform.python_version()}", f"platform={platform.platform()}",
        f"timestamp={datetime.utcnow().isoformat()}Z",
    ]
    with open(args.outfile, "w", newline="") as f:
        f.write("# " + ", ".join([p for p in banner_parts if p]) + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Sidecar JSON with metadata
    sidecar = {
        "events": events,
        "device": args.device,
        "autotune_source": autotune_source,
        "N": args.N,
        "hidden_dim": args.hidden_dim,
        "up_cols": args.up_cols,
        "gate_cols": args.gate_cols,
        "m": args.m,
        "steps_per_k": args.steps_per_k,
        "ks": ks,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        import torch  # type: ignore
        sidecar["torch"] = torch.__version__
    except Exception:
        sidecar["torch"] = None

    json_path = Path(args.outfile).with_suffix(".json")
    with open(json_path, "w") as jf:
        json.dump(sidecar, jf, indent=2)

    print(f"Saved {args.outfile} and {json_path}")


if __name__ == "__main__":
    main()
