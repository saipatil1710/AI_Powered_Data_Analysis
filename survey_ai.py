# ===============================
# Rich AI Survey Pipeline (HTML + PDF)
# ===============================
import os, io, base64, math, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------- Helpers ----------
def b64_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def kish_neff(w):
    w = np.asarray(w, float)
    if w.size == 0 or np.all(w == 0) or np.any(~np.isfinite(w)):
        return np.nan
    return (w.sum() ** 2) / np.square(w).sum()

def weighted_mean_ci(x, w, alpha=0.05):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[m], w[m]
    if x.size == 0 or w.sum() == 0:
        return (np.nan, (np.nan, np.nan), np.nan)
    wm = np.average(x, weights=w)
    v = np.average((x - wm) ** 2, weights=w)
    neff = kish_neff(w)
    if not np.isfinite(neff) or neff <= 1:
        return (wm, (np.nan, np.nan), neff)
    se = math.sqrt(v / neff); z = stats.norm.ppf(0.975)
    return (wm, (wm - z * se, wm + z * se), neff)

def winsorize_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.clip(lo, hi)

# ---------- PDF Table Helper ----------
def pdf_table(pdf, df, col_widths=None, max_col_chars=30):
    """Render a pandas DataFrame as a table in FPDF."""
    if df is None or df.empty:
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 8, "No data", ln=True)
        return

    df2 = df.copy()
    df2 = df2.fillna("")
    # convert complex objects to string safely
    for c in df2.columns:
        df2[c] = df2[c].apply(lambda x: str(x))

    cols = df2.columns.tolist()
    n_cols = len(cols)
    page_width = pdf.w - 2 * pdf.l_margin
    if col_widths is None:
        col_widths = [page_width / n_cols] * n_cols
    else:
        # if provided but length mismatched, fall back to even widths
        if len(col_widths) != n_cols:
            col_widths = [page_width / n_cols] * n_cols

    # Header
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 8, str(col)[:max_col_chars], border=1, align="C", fill=True)
    pdf.ln(8)

    # Rows
    pdf.set_font("Arial", "", 8)
    for _, row in df2.iterrows():
        for i, col in enumerate(cols):
            text = str(row[col])
            # wrap/clip text to fit cell
            pdf.cell(col_widths[i], 6, text[:max_col_chars], border=1, align="C")
        pdf.ln(6)
    pdf.ln(4)


# ---------- Core ----------
class SurveyAI:
    def __init__(self, df, weight_col=None, rules=None,
                 num_impute="median", cat_impute="mode",
                 outlier="iqr", outlier_param=1.5):
        self.raw = df.copy()
        self.df = df.copy()
        self.weight_col = weight_col if (weight_col in df.columns) else None
        self.rules = rules or {}
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.outlier = outlier
        self.outlier_param = outlier_param
        self.log = {"shape_before": df.shape, "steps": []}
        self.assets = []  # list of {"title","src"}

    # --- cleaning ---
    def impute_missing(self):
        det = {}
        num = self.df.select_dtypes(include=np.number).columns
        cat = [c for c in self.df.columns if c not in num]
        if self.num_impute != "none":
            for c in num:
                if self.df[c].isna().any():
                    v = {"mean": self.df[c].mean(),
                         "median": self.df[c].median(),
                         "zero": 0}.get(self.num_impute, self.df[c].median())
                    self.df[c] = self.df[c].fillna(v)
            det["numeric"] = self.num_impute
        if self.cat_impute != "none":
            for c in cat:
                if self.df[c].isna().any():
                    if self.cat_impute == "mode":
                        m = self.df[c].mode(dropna=True)
                        v = m.iloc[0] if len(m) else "Missing"
                    else:
                        v = "Missing"
                    self.df[c] = self.df[c].fillna(v)
            det["categorical"] = self.cat_impute
        self.log["steps"].append({"Missing data": det})
        return self

    def handle_outliers(self):
        if self.outlier == "none":
            self.log["steps"].append({"Outliers": "skipped"}); return self
        det = {}
        for c in self.df.select_dtypes(include=np.number).columns:
            s = self.df[c].astype(float)
            before = s.copy()
            if self.outlier == "iqr":
                s2 = winsorize_iqr(s, self.outlier_param)
            else:  # zscore winsor
                thr = float(self.outlier_param)
                z = (s - s.mean()) / s.std(ddof=0)
                lo = s[z.abs() <= thr].min(); hi = s[z.abs() <= thr].max()
                s2 = s.clip(lo, hi)
            self.df[c] = s2
            det[c] = int((before != s2).sum())
        self.log["steps"].append({"Outliers changed": det})
        return self

    def enforce_rules(self):
        if not self.rules:
            self.log["steps"].append({"Rules": "none"}); return self
        mask = pd.Series(True, index=self.df.index)
        det = {}
        for col, rule in self.rules.items():
            if col not in self.df.columns:
                det[col] = "missing"; continue
            s = pd.to_numeric(self.df[col], errors="coerce")
            if "min" in rule:
                bad = s < rule["min"]; det[f"{col}<min"] = int(bad.sum()); mask &= ~bad.fillna(False)
            if "max" in rule:
                bad = s > rule["max"]; det[f"{col}>max"] = int(bad.sum()); mask &= ~bad.fillna(False)
            if "in" in rule:
                allowed = set(rule["in"])
                bad = ~self.df[col].isin(allowed); det[f"{col} invalid"] = int(bad.sum()); mask &= ~bad.fillna(False)
        before = len(self.df)
        self.df = self.df[mask].reset_index(drop=True)
        det["rows_removed"] = before - len(self.df)
        self.log["steps"].append({"Rules summary": det})
        return self

    def apply_weights(self):
        info = {"weight_col": self.weight_col}
        if self.weight_col is None:
            info["applied"] = False; self.log["steps"].append({"Weights": info}); return self
        w = pd.to_numeric(self.df[self.weight_col], errors="coerce").fillna(0).clip(lower=0)
        if w.sum() == 0:  # fallback if invalid weights
            self.df["_w"] = 1
            self.df["_w_norm"] = self.df["_w"] / len(self.df)
            info["applied"] = False
        else:
            self.df["_w"] = w
            self.df["_w_norm"] = w / w.sum()
            info["sum"] = float(w.sum())
            info["neff_kish"] = float(kish_neff(w))
            info["applied"] = True
        self.log["steps"].append({"Weights": info})
        return self

    # --- stats ---
    def stats_blocks(self, top_k=8):
        blk = {}
        df = self.df
        num = [c for c in df.select_dtypes(include=np.number).columns if c not in ["_w", "_w_norm"]]
        cat = [c for c in df.columns if c not in num + ["_w", "_w_norm"]]

        # Descriptives
        if len(num):
            desc = df[num].describe().T
            desc["skew"] = df[num].skew(numeric_only=True)
            desc["kurtosis"] = df[num].kurtosis(numeric_only=True)
            blk["descriptives"] = desc.reset_index().rename(columns={"index": "column"}).to_dict(orient="records")
        else:
            blk["descriptives"] = []

        # Missingness
        miss = df.isna().sum().sort_values(ascending=False)
        blk["missing"] = miss[miss > 0].to_dict()

        # Outlier counts (z>3)
        out = {}
        for c in num:
            s = df[c].dropna()
            if s.std(ddof=0) == 0 or len(s) == 0:
                out[c] = 0; continue
            z = ((s - s.mean()) / s.std(ddof=0)).abs() > 3
            out[c] = int(z.sum())
        blk["outliers"] = out

        # Correlations
        if len(num) >= 2:
            corr = df[num].corr()
            blk["correlation"] = corr.to_dict()
        else:
            blk["correlation"] = {}

        # Top categories
        topcats = {}
        for c in cat:
            topcats[c] = df[c].astype("string").value_counts().head(top_k).to_dict()
        blk["top_categories"] = topcats

        # Weighted means (if weights)
        if "_w_norm" in df.columns:
            wstats = []
            for c in num:
                m, (l, u), ne = weighted_mean_ci(df[c], df["_w_norm"])
                wstats.append({"column": c, "weighted_mean": m, "ci_low": l, "ci_high": u, "neff": ne})
            blk["weighted_means"] = wstats
        else:
            blk["weighted_means"] = []

        # Extra summary
        blk["row_count"] = len(df)
        blk["col_count"] = len(df.columns)
        blk["numeric_columns"] = num
        blk["categorical_columns"] = cat

        self.log["stats"] = blk
        return self

    # --- visuals (assets stored as base64) ---
    def visuals(self, max_num=6, max_cat=6):
        df = self.df.copy()
        # sample safeguard
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)

        num = [c for c in df.select_dtypes(include=np.number).columns if c not in ["_w", "_w_norm"]]
        cat = [c for c in df.columns if c not in num + ["_w", "_w_norm"]]

        # Histograms & KDE
        for c in num[:max_num]:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df[c].dropna(), bins=30, kde=True, ax=ax)
            ax.set_title(f"Histogram: {c}")
            self.assets.append({"title": f"Histogram: {c}", "src": b64_fig(fig)})

        # Boxplots
        for c in num[:max_num]:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            sns.boxplot(x=df[c], ax=ax)
            ax.set_title(f"Boxplot: {c}")
            self.assets.append({"title": f"Boxplot: {c}", "src": b64_fig(fig)})

        # Violin
        for c in num[:max_num]:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            sns.violinplot(x=df[c], ax=ax)
            ax.set_title(f"Violin: {c}")
            self.assets.append({"title": f"Violin: {c}", "src": b64_fig(fig)})

        # Correlation heatmap
        if len(num) >= 2:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            corr = df[num].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
            ax.set_title("Correlation heatmap")
            self.assets.append({"title": "Correlation heatmap", "src": b64_fig(fig)})

        # Missingness map
        sample_df = df if len(df) <= 2000 else df.sample(n=2000, random_state=42)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(sample_df.isna(), aspect="auto", interpolation="nearest", cmap="gray_r")
        ax.set_title(f"Missingness map (showing {len(sample_df)} rows)")
        ax.set_xlabel("Columns"); ax.set_ylabel("Rows")
        ax.set_yticks([]); ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
        self.assets.append({"title": "Missingness map", "src": b64_fig(fig)})

        # Top categorical bars
        for c in cat[:max_cat]:
            vc = df[c].astype("string").value_counts().head(12)
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            vc.iloc[::-1].plot(kind="barh", ax=ax)
            ax.set_title(f"Top categories: {c}"); ax.set_xlabel("Count")
            self.assets.append({"title": f"Top categories: {c}", "src": b64_fig(fig)})

        # Scatter (pairwise)
        if len(num) >= 2:
            top = (df[num].var().sort_values(ascending=False).head(min(4, len(num)))).index.tolist()
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    x, y = top[i], top[j]
                    fig, ax = plt.subplots(figsize=(5, 3.2))
                    ax.scatter(df[x], df[y], s=10, alpha=0.6)
                    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"Scatter: {x} vs {y}")
                    self.assets.append({"title": f"Scatter: {x} vs {y}", "src": b64_fig(fig)})
        return self

    # --- outputs ---
    def save_clean(self, path="cleaned.csv"):
        out = self.df.drop(columns=[c for c in ["_w", "_w_norm"] if c in self.df.columns])
        out.to_csv(path, index=False)
        self.log["clean_path"] = path
        return path

    def save_html(self, path="report.html", title="AI Survey Report"):
        css = """
        body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#111;}
        h1{margin:0 0 6px 0;} .muted{color:#555}
        .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin-top:6px;}
        .card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:14px 0;background:#fff;}
        table{border-collapse:collapse;width:100%;font-size:14px}
        th,td{border:1px solid #e5e7eb;padding:6px;text-align:left}
        th{background:#f7fafc}
        img{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px}
        code{background:#f6f8fa;padding:2px 6px;border-radius:6px}
        .muted{color:#555;font-size:13px;}
        .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0;}
        .stat-box{background:#ffffff;padding:14px;border-radius:8px;text-align:center;border:1px solid #e5e7eb;}
        .stat-box h3{margin:0;font-size:20px;color:#007bff;}
        .stat-box p{margin:5px 0 0;font-size:14px;color:#555;}
        """

        # build tables
        desc_records = self.log["stats"].get("descriptives", [])
        desc = pd.DataFrame(desc_records) if len(desc_records) else pd.DataFrame()
        wm = pd.DataFrame(self.log["stats"].get("weighted_means", []))

        def tbl(df_):
            return df_.to_html(index=False, float_format=lambda x: f"{x:,.4f}" if isinstance(x, (int, float, np.floating)) else x) if len(df_) else "<p class='muted'>No data</p>"

        total_outliers = sum(self.log["stats"].get("outliers", {}).values())
        missing_cols = len(self.log["stats"].get("missing", {}))

        steps_html = ""
        for s in self.log.get("steps", []):
            steps_html += f"<div class='card'><pre style='white-space:pre-wrap'>{textwrap.indent(str(s), ' ')}</pre></div>"

        imgs = "\n".join([f"<div class='card'><div class='muted'>{a['title']}</div><img src='{a['src']}'/></div>" for a in self.assets])

        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>
<h1>{title}</h1>
<p class="muted">Shape before: {self.log.get('shape_before')} &middot; after: {self.df.shape}</p>

<div class="stats-grid">
  <div class="stat-box"><h3>{self.log['stats'].get('row_count', 0)}</h3><p>Rows</p></div>
  <div class="stat-box"><h3>{self.log['stats'].get('col_count', 0)}</h3><p>Columns</p></div>
  <div class="stat-box"><h3>{missing_cols}</h3><p>Cols with Missing</p></div>
  <div class="stat-box"><h3>{total_outliers}</h3><p>Total Outliers</p></div>
</div>

<div class="card">
  <h3>Processing Steps</h3>
  {steps_html}
</div>

<div class="card">
  <h3>Numeric Descriptives</h3>
  {tbl(desc)}
</div>

<div class="card">
  <h3>Insights</h3>
  <pre>Missing: {self.log['stats'].get('missing', {}) if self.log['stats'].get('missing', {}) else "None"}</pre>
  <pre>Outliers: {self.log['stats'].get('outliers', {})}</pre>
  <pre>Top categories: {self.log['stats'].get('top_categories', {})}</pre>
</div>

<div class="card">
  <h3>Weighted Means & 95% CI</h3>
  {tbl(wm)}
</div>

<h3>Visuals</h3>
<div class="grid">
  {imgs}
</div>

<div class="card">
  <h3>Outputs</h3>
  <p>Cleaned dataset: <code>{self.log.get('clean_path','(not saved)')}</code></p>
</div>

</body></html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        self.log["html_path"] = path
        return path

    def try_pdf(self, html="report.html", pdf="report.pdf"):
        """
        Generate PDF. Prefers wkhtmltopdf or WeasyPrint.
        Fallback: FPDF with structured layout: stats boxes, insights (bulleted), 
        descriptives (neat table), weighted means (table), and charts in 2×2 grid.
        """
        # Try wkhtmltopdf/pdfkit
        try:
            import pdfkit
            pdfkit.from_file(html, pdf)
            return pdf
        except Exception:
            pass

        # Try WeasyPrint
        try:
            from weasyprint import HTML
            HTML(html).write_pdf(pdf)
            return pdf
        except Exception:
            pass

        # Fallback: FPDF
        try:
            from fpdf import FPDF
            pdfdoc = FPDF()
            pdfdoc.set_auto_page_break(auto=True, margin=15)

            # Title
            pdfdoc.add_page()
            pdfdoc.set_font("Arial", "B", 20)
            pdfdoc.cell(0, 15, "AI Survey Report", ln=True, align="C")
            pdfdoc.ln(10)

            stats = self.log.get("stats", {})

            # --- Stat Boxes ---
            pdfdoc.set_font("Arial", "B", 12)
            box_w, box_h = 45, 20
            spacing = 10
            values = [
                (str(stats.get("row_count", 0)), "Rows"),
                (str(stats.get("col_count", 0)), "Columns"),
                (str(len(stats.get("missing", {}))), "Cols Missing"),
                (str(sum(stats.get("outliers", {}).values())), "Outliers"),
            ]
            x_start = 15
            y_start = pdfdoc.get_y()
            for i, (val, label) in enumerate(values):
                x = x_start + i * (box_w + spacing)
                pdfdoc.set_xy(x, y_start)
                pdfdoc.set_fill_color(230, 230, 230)
                pdfdoc.rect(x, y_start, box_w, box_h, style="F")
                pdfdoc.set_xy(x, y_start + 4)
                pdfdoc.set_font("Arial", "B", 12)
                pdfdoc.cell(box_w, 6, val, align="C")
                pdfdoc.set_xy(x, y_start + 12)
                pdfdoc.set_font("Arial", "", 9)
                pdfdoc.cell(box_w, 6, label, align="C")
            pdfdoc.ln(box_h + 8)

            # --- Insights ---
            pdfdoc.set_font("Arial", "B", 14)
            pdfdoc.cell(0, 10, "Insights", ln=True)
            pdfdoc.set_font("Arial", "", 11)

            pdfdoc.multi_cell(0, 6, "• Missing:")
            for k, v in stats.get("missing", {}).items():
                pdfdoc.multi_cell(0, 6, f"    - {k}: {v}")

            pdfdoc.multi_cell(0, 6, "• Outliers:")
            for k, v in stats.get("outliers", {}).items():
                pdfdoc.multi_cell(0, 6, f"    - {k}: {v}")

            pdfdoc.multi_cell(0, 6, "• Top categories:")
            for k, v in stats.get("top_categories", {}).items():
                pdfdoc.multi_cell(0, 6, f"    - {k}: {v}")
            pdfdoc.ln(5)

            # --- Descriptive statistics table ---
            desc_records = stats.get("descriptives", [])
            desc_df = pd.DataFrame(desc_records) if len(desc_records) else pd.DataFrame()
            if not desc_df.empty:
                pdfdoc.set_font("Arial", "B", 14)
                pdfdoc.cell(0, 10, "Numeric Descriptives", ln=True)

                # round floats for readability
                for col in desc_df.select_dtypes(include=np.number).columns:
                    desc_df[col] = desc_df[col].round(3)

                # adjust col widths
                n_cols = len(desc_df.columns)
                page_width = pdfdoc.w - 2 * pdfdoc.l_margin
                col_widths = [page_width / n_cols] * n_cols

                pdf_table(pdfdoc, desc_df.head(10), col_widths)

            # --- Weighted means ---
            wm_list = stats.get("weighted_means", [])
            wm_df = pd.DataFrame(wm_list) if len(wm_list) else pd.DataFrame()
            if not wm_df.empty:
                pdfdoc.set_font("Arial", "B", 14)
                pdfdoc.cell(0, 10, "Weighted Means & 95% CI", ln=True)
                for col in wm_df.select_dtypes(include=np.number).columns:
                    wm_df[col] = wm_df[col].round(3)
                pdf_table(pdfdoc, wm_df)

            # --- Charts in 2×2 grid ---
            if self.assets:
                pdfdoc.add_page()
                pdfdoc.set_font("Arial", "B", 14)
                pdfdoc.cell(0, 10, "Charts", ln=True)

                x_positions = [15, 110]
                y_positions = [60, 160]
                col_w = 85
                row_h = 80
                chart_count = 0

                for i, a in enumerate(self.assets):
                    b = base64.b64decode(a["src"].split(",")[1])
                    tmp = f"_tmp_chart_{i}.png"
                    with open(tmp, "wb") as f:
                        f.write(b)

                    grid_pos = chart_count % 4
                    col = grid_pos % 2
                    row = grid_pos // 2
                    x = x_positions[col]; y = y_positions[row]

                    pdfdoc.image(tmp, x=x, y=y, w=col_w, h=row_h)
                    os.remove(tmp)

                    chart_count += 1
                    if chart_count % 4 == 0 and (i + 1) < len(self.assets):
                        pdfdoc.add_page()
                        pdfdoc.set_font("Arial", "B", 14)
                        pdfdoc.cell(0, 10, "Charts (cont.)", ln=True)

            pdfdoc.output(pdf)
            return pdf
        except Exception as e:
            print("⚠ PDF generation failed in all modes:", e)
            return None
