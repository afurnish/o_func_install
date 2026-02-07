# -*- coding: utf-8 -*-
"""
TESTER: Shared 12-day tide validation plot (same time window for all gauges)
+ DEBUG: verifies Heysham and Liverpool are actually different

What this does:
- Uses ONE shared date window for *all* gauges (no per-gauge "best window")
- Legend only (no titles/stats on the figure)
- Clear line hierarchy + draw order:
    Tide Gauge (grey, thick, behind)
    IRENE (red, dashed, on top)
    UKC4 (blue, dotted, on top)
- Adds console-only diagnostics to confirm gauge 0 != gauge 1
- Preserves your original indexing logic: model[gauge_i][mk]

How to use in Spyder:
- Run this file (or paste into a cell).
- Then call:
    timeseries_plot_test_shared_window(self, list_of_data, variable_name, var_dict)

Assumptions:
- self.time_sliced is numpy datetime64 array
- self.tide_save is iterable (len = number of gauges)
- self.tide_loc_dict keys correspond to gauge order
- list_of_data has the same structure your original code expects
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================
# USER CONTROLS
# =========================
SHARED_START = "2013-11-21"   # inclusive
SHARED_END   = "2013-12-02"   # inclusive-ish (robustly indexed)
# =========================


def _as_np(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def _get_window_indices_from_dates(tt, start_date, end_date):
    """
    Return indices (s0, s1) spanning [start_date, end_date] using searchsorted.
    """
    tt = np.asarray(tt)
    t0 = np.datetime64(start_date)
    t1 = np.datetime64(end_date)

    s0 = int(np.searchsorted(tt, t0, side="left"))
    s1 = int(np.searchsorted(tt, t1, side="right")) - 1

    s0 = max(0, min(s0, len(tt) - 1))
    s1 = max(0, min(s1, len(tt) - 1))
    if s1 <= s0:
        s0, s1 = 0, len(tt) - 1
    return s0, s1


def _build_named_series_for_gauge(list_of_data, gauge_i, variable_name, debug=False):
    """
    Matches your naming logic:
      Tide Gauge, UKC4(PRIMEA)->IRENE, UKC4(ao)->UKC4

    Preserves your original indexing logic:
      mk = first key in model[gauge_i]
      y  = model[gauge_i][mk]
    """
    if variable_name != "surface_height":
        data_iter = list_of_data[1:]  # remove tide gauge
        name_map = [r"UKC4$_{\mathrm{PRIMEA}}$", r"UKC4$_{\mathrm{ao}}$"]
    else:
        data_iter = list_of_data
        name_map = ["Tide Gauge", r"UKC4$_{\mathrm{PRIMEA}}$", r"UKC4$_{\mathrm{ao}}$"]

    series = {}
    mk_debug = {}

    for kil, model in enumerate(data_iter):
        # original logic
        mk = list(model[gauge_i].keys())[0]
        y = model[gauge_i][mk]

        if name_map[kil] == r"UKC4$_{\mathrm{PRIMEA}}$":
            nm = "IRENE"
        elif name_map[kil] == r"UKC4$_{\mathrm{ao}}$":
            nm = "UKC4"
        else:
            nm = name_map[kil]

        series[nm] = y
        mk_debug[nm] = mk

    if debug:
        print(f"    [mk keys] gauge_i={gauge_i} -> " +
              ", ".join([f"{k}:{mk_debug[k]}" for k in mk_debug]))

    return series


def _fingerprint(x, n=200):
    """
    Console-only fingerprint to verify two gauges differ.
    Uses mean/std/min/max on first n samples.
    """
    a = _as_np(x).astype(float)
    a = a[: min(len(a), n)]
    return {
        "mean": float(np.nanmean(a)),
        "std":  float(np.nanstd(a)),
        "min":  float(np.nanmin(a)),
        "max":  float(np.nanmax(a)),
    }


def timeseries_plot_test_shared_window(self, list_of_data, variable_name, var_dict, debug=True):
    """
    Plot one figure per gauge using ONE shared time window for all gauges.
    No titles, no stats. Legend only.
    debug=True prints gauge name + fingerprints + mk keys to console.
    """
    tt = np.asarray(self.time_sliced)
    gauge_names = list(self.tide_loc_dict.keys())

    # shared window indices once (same xlim for all)
    s0, s1 = _get_window_indices_from_dates(tt, SHARED_START, SHARED_END)

    # store fingerprints to compare gauges at end
    fp_store = {}

    for i, _ in enumerate(self.tide_save):

        gname = gauge_names[i] if i < len(gauge_names) else f"Gauge {i}"
        series = _build_named_series_for_gauge(list_of_data, i, variable_name, debug=debug)

        if "Tide Gauge" not in series:
            raise ValueError("This tester expects variable_name='surface_height' so Tide Gauge is present.")

        if debug:
            print("----")
            print(f"Gauge index {i}: {gname}")
            fp_store[gname] = {
                "Tide Gauge": _fingerprint(series["Tide Gauge"]),
                "IRENE": _fingerprint(series.get("IRENE")),
                "UKC4": _fingerprint(series.get("UKC4")),
            }
            print("    [fingerprint] Tide Gauge:", fp_store[gname]["Tide Gauge"])
            if series.get("IRENE") is not None:
                print("    [fingerprint] IRENE     :", fp_store[gname]["IRENE"])
            if series.get("UKC4") is not None:
                print("    [fingerprint] UKC4      :", fp_store[gname]["UKC4"])

        # ---- Plot ----
        fig, ax = plt.subplots(dpi=300)
        fig.set_figheight(4.0)
        fig.set_figwidth(7.0)

        # Keep your colours: grey (gauge), red (IRENE), blue (UKC4)
        STYLE = {
            "Tide Gauge": dict(color="grey", lw=2.6 * 1.5, ls="-", alpha=0.75, zorder=2),
            "IRENE":      dict(color="red",  lw=2.0 * 1.2, ls="--", alpha=0.95, zorder=4),
            "UKC4":       dict(color="blue", lw=1.5,       ls=":",  alpha=0.95, zorder=5),
        }

        # same x-limits for all gauges
        ax.set_xlim([tt[s0], tt[s1]])

        # draw order: Tide Gauge behind, models on top
        for nm in ["Tide Gauge", "IRENE", "UKC4"]:
            if nm not in series:
                continue
            ax.plot(
                tt, series[nm],
                label=nm,
                **STYLE[nm],
                solid_capstyle="round"
            )

        # Legend only (no title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)

        # Cleaner ticks for ~12 days
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # every 2 days
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        # Labels
        if variable_name == "surface_height":
            var_label = "Surface Height"
        elif variable_name == "salinity":
            var_label = "Salinity"
        else:
            var_label = variable_name

        ax.set_ylabel(f"{var_label} [{var_dict[variable_name]['UNITS']}]")

        plt.tight_layout()
        plt.show()

    if debug and len(fp_store) >= 2:
        # quick check: are the Tide Gauge fingerprints identical?
        names = list(fp_store.keys())
        a = fp_store[names[0]]["Tide Gauge"]
        b = fp_store[names[1]]["Tide Gauge"]
        same = (a["mean"] == b["mean"] and a["std"] == b["std"] and a["min"] == b["min"] and a["max"] == b["max"])
        print("----")
        print("Sanity check (first two gauges Tide Gauge fingerprints identical?) ->", same)
        if same:
            print("If this says True, you are almost certainly plotting the same gauge data twice upstream.")


#%
# Example call:
timeseries_plot_test_shared_window(self, list_of_data, variable_name, var_dict, debug=True)
