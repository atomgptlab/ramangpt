"""
ir.py — Infrared (IR) Intensity Explorer.

Fetch and parse IR spectra from JARVIS-DFT DFPT calculations by reading
the JARVIS XML pages for <ir_intensity>, <max_ir_mode>, <min_ir_mode>.

Provides both raw peak data and Gaussian-broadened spectra.

Usage:
    python ir.py                  # run with default JID (silicon: JVASP-1002)
    python ir.py JVASP-1174       # run for a specific JARVIS ID
"""

import re
import sys
from typing import Optional

import numpy as np
import requests


# ────────────────────────────────────────────────────────────────
# XML sources
# ────────────────────────────────────────────────────────────────
JARVIS_XML_URLS = [
    "https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/{jid}.xml",
    "https://jarvis.nist.gov/jarvisdft/xml/{jid}.xml",
]


def _fetch_jarvis_xml(jid: str) -> Optional[str]:
    """Fetch the raw XML content for a given JARVIS ID."""
    for url_template in JARVIS_XML_URLS:
        url = url_template.format(jid=jid)
        try:
            print(f"  Trying {url}")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                print(f"  Got XML ({len(resp.text)} bytes)")
                return resp.text
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    return None


def _extract_xml_tag(xml_text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag> from XML text."""
    pattern = r"<{tag}>(.*?)</{tag}>".format(tag=re.escape(tag))
    match = re.search(pattern, xml_text, re.DOTALL)
    if match:
        return match.group(1).strip().strip("'\"")
    return None


def _parse_ir_from_xml(xml_text: str, sigma: float = 10.0) -> dict:
    """
    Parse IR intensity data from the XML.

    The <ir_intensity> format is:
        'freq1,freq2,...;intensity1,intensity2,...'
    Two arrays separated by semicolon.

    Returns raw peaks plus a Gaussian-broadened spectrum (cm^-1).
    """
    result = {
        "frequencies": None,
        "intensities": None,
        "peaks": None,
        "gaussian_x": None,
        "gaussian_y": None,
        "max_ir_mode": None,
        "min_ir_mode": None,
    }

    raw_ir = _extract_xml_tag(xml_text, "ir_intensity")
    if raw_ir and raw_ir.strip():
        try:
            parts = raw_ir.split(";")
            if len(parts) >= 2:
                freqs = [float(v.strip()) for v in parts[0].split(",") if v.strip()]
                intens = [float(v.strip()) for v in parts[1].split(",") if v.strip()]
                if freqs and intens:
                    result["frequencies"] = freqs
                    result["intensities"] = intens

                    n = min(len(freqs), len(intens))
                    result["peaks"] = [
                        {"frequency": round(freqs[i], 4), "intensity": intens[i]}
                        for i in range(n)
                    ]

                    # Gaussian broadening
                    freq_min = max(0.0, min(freqs) - 5 * sigma)
                    freq_max = max(freqs) + 5 * sigma
                    x_grid = np.linspace(freq_min, freq_max, 500)
                    y_grid = np.zeros_like(x_grid)
                    for i in range(n):
                        amp = intens[i]
                        if abs(amp) < 1e-30:
                            continue
                        y_grid += amp * np.exp(-0.5 * ((x_grid - freqs[i]) / sigma) ** 2)

                    result["gaussian_x"] = [round(v, 4) for v in x_grid.tolist()]
                    result["gaussian_y"] = [round(v, 8) for v in y_grid.tolist()]
        except (ValueError, IndexError) as e:
            print(f"  IR intensity parse error: {e}")

    raw_max = _extract_xml_tag(xml_text, "max_ir_mode")
    if raw_max:
        try:
            result["max_ir_mode"] = float(raw_max)
        except ValueError:
            pass
    raw_min = _extract_xml_tag(xml_text, "min_ir_mode")
    if raw_min:
        try:
            result["min_ir_mode"] = float(raw_min)
        except ValueError:
            pass

    return result


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────
def get_ir_data(jid: str, sigma: float = 10.0) -> dict:
    """
    Fetch and parse IR data for a JARVIS ID.

    Returns dict with keys: jid, peaks, gaussian_x, gaussian_y,
    max_ir_mode, min_ir_mode, formula (parsed from XML if present).
    Raises RuntimeError if XML cannot be fetched or IR data not found.
    """
    print(f"Fetching IR data for {jid}")
    xml_text = _fetch_jarvis_xml(jid)
    if xml_text is None:
        raise RuntimeError(f"Could not fetch XML data for {jid}")

    parsed = _parse_ir_from_xml(xml_text, sigma=sigma)
    if parsed["peaks"] is None:
        if "<ir_intensity>" in xml_text:
            raise RuntimeError(
                f"IR section found but could not parse intensity data for {jid}"
            )
        raise RuntimeError(f"No IR intensity data available for {jid}")

    formula = _extract_xml_tag(xml_text, "formula") or ""

    return {
        "jid": jid,
        "formula": formula,
        "peaks": parsed["peaks"],
        "gaussian_x": parsed["gaussian_x"],
        "gaussian_y": parsed["gaussian_y"],
        "max_ir_mode": parsed["max_ir_mode"],
        "min_ir_mode": parsed["min_ir_mode"],
    }


# ────────────────────────────────────────────────────────────────
# Local testing
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # JVASP-1002 = bulk diamond-cubic silicon
    jid = sys.argv[1] if len(sys.argv) > 1 else "JVASP-1002"

    try:
        data = get_ir_data(jid)
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    peaks = data["peaks"]
    active = [p for p in peaks if abs(p["intensity"]) > 1e-20]

    print(f"\n{'='*60}")
    print(f"  JID:          {data['jid']}")
    print(f"  Formula:      {data['formula'] or '(unknown)'}")
    print(f"  Total modes:  {len(peaks)}")
    print(f"  IR-active:    {len(active)}")
    print(f"  max_ir_mode:  {data['max_ir_mode']} cm^-1")
    print(f"  min_ir_mode:  {data['min_ir_mode']} cm^-1")
    print(f"{'='*60}\n")

    print("Top IR-active modes (by |intensity|):")
    top = sorted(peaks, key=lambda p: abs(p["intensity"]), reverse=True)[:10]
    print(f"  {'Frequency (cm^-1)':>20}  {'Intensity':>15}")
    print(f"  {'-'*20}  {'-'*15}")
    for p in top:
        print(f"  {p['frequency']:>20.4f}  {p['intensity']:>15.6g}")
