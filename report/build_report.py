"""Fill the {{tokens}} in report.md with real numbers from the pipeline.

Run order:
    python run_pipeline.py            # writes outputs/metrics/results.json
    python report/build_report.py     # writes report/report_final.md (+ docx/pdf if tools exist)

Any number not yet produced is shown as "(pending run)".
"""
import json, re, shutil, subprocess, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "outputs" / "metrics" / "results.json"
PENDING = "(pending run)"


def pct(x): return f"{x*100:.1f}%"
def f3(x): return f"{x:.3f}"


def build_tokens():
    if not RES.exists():
        print("No results.json yet. Run run_pipeline.py first; writing a draft.")
        return {}
    r = json.loads(RES.read_text())
    t = {}
    a = r.get("track_A", {})
    if a:
        t["A_BEST"] = a["best_model"]
        t["A_ACC"] = pct(a["test_metrics"]["accuracy"])
        t["A_F1"] = f3(a["test_metrics"]["f1_macro"])
        t["A_AUC"] = f3(a["macro_roc_auc"])
        t["A_PVAL"] = f"{a['paired_ttest_logreg_vs_best']['p_value']:.1e}"
    bm = r.get("track_B_models", {})
    b = r.get("track_B", {})
    if bm:
        if "linreg" in bm:
            t["B_LIN_R2"] = f3(bm["linreg"]["r2"])
            t["B_LIN_RMSE"] = f3(bm["linreg"]["rmse"])
            t["B_LIN_MAE"] = f3(bm["linreg"]["mae"])
        best = b.get("best_model")
        if best and best in bm:
            t["B_BEST"] = best
            t["B_BEST_R2"] = f3(bm[best]["r2"])
            t["B_BEST_RMSE"] = f3(bm[best]["rmse"])
            t["B_BEST_MAE"] = f3(bm[best]["mae"])
    return t


def render(md_name):
    """Best-effort md -> docx (pandoc) -> pdf (LibreOffice). Skips quietly if absent."""
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", md_name, "-o", "report.docx", "--resource-path=.:.."],
                       cwd=HERE, check=False)
        print("wrote report.docx")
        if shutil.which("soffice"):
            env = dict(os.environ, HOME="/tmp")
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", ".",
                            "report.docx"], cwd=HERE, check=False, env=env)
            print("wrote report.pdf")
        else:
            print("LibreOffice not found: open report.docx in Word and Save As PDF.")
    else:
        print("pandoc not found: export report_final.md to PDF in your editor.")


def main():
    tokens = build_tokens()
    src = (HERE / "report.md").read_text(encoding="utf-8")
    filled = re.sub(r"\{\{(\w+)\}\}", lambda m: str(tokens.get(m.group(1), PENDING)), src)
    (HERE / "report_final.md").write_text(filled, encoding="utf-8")
    print("wrote report_final.md")
    pend = sorted({m for m in re.findall(r"\{\{(\w+)\}\}", src) if m not in tokens})
    if pend:
        print("pending (run run_pipeline.py first):", pend)
    render("report_final.md")


if __name__ == "__main__":
    main()
