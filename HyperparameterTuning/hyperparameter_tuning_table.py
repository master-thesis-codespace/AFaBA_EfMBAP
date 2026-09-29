#!/usr/bin/env python3
import ast
import csv
import json
import os
import re

HEADING = "Hyperparameter Tuning - HBN withSD Dataset (OOF Cross-Validation)"
OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperparameter_tuning_table.html")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # workspace root

MODELS = [
    {
        "name": "XGBoost",
        "subdir": ".",
        "hp_cols": [
            ("n_estimators", "n_estimators"),
            ("learning_rate", "learning_rate"),
            ("max_depth", "max_depth"),
            ("gamma", "gamma"),
            ("early_stopping", "early_stopping_rounds"),
        ],
        "default_dir": "BRAIN_XB_beImpFeat-GS1",
        "run_dirs": [
            "BRAIN_XB_beImpFeat-GS1",
            "BRAIN_XB_beImpFeat-GS2",
            "BRAIN_XB_beImpFeat-GS3",
            #"BRAIN_XB_beImpFeat-GS4",
            #"BRAIN_XB_beImpFeat-GS5",
            #"BRAIN_XB_beImpFeat-GS6",
            #"BRAIN_XB_beImpFeat-GS7",
            #"BRAIN_XB_beImpFeat-GS8"
        ],
    },
    {
        "name": "ExtraTree",
        "subdir": ".",
        "hp_cols": [
            ("n_estimators", "n_estimators"),
            ("max_depth", "max_depth"),
            ("min_samples_split", "min_samples_split"),
            ("max_features", "max_features"),
            ("bootstrap", "bootstrap"),
            ("min_samples_leaf", "min_samples_leaf"),
            ("criterion", "criterion"),
        ],
        "default_dir": "BRAIN_ET_beImpFeat-GS1",
        "run_dirs": [
            "BRAIN_ET_beImpFeat-GS1",
            #"BRAIN_ET_beImpFeat-GS2",
            #"BRAIN_ET_beImpFeat-GS3",
            #"BRAIN_ET_beImpFeat-GS4",
            #"BRAIN_ET_beImpFeat-GS5",
            #"BRAIN_ET_beImpFeat-GS6",
            #"BRAIN_ET_beImpFeat-GS7",
            #"BRAIN_ET_beImpFeat-GS8"
        ],
    },
    {
        "name": "LightGBM",
        "subdir": ".",
        "hp_cols": [
            ("n_estimators", "n_estimators"),
            ("learning_rate", "learning_rate"),
            ("max_depth", "max_depth"),
            ("num_leaves", "num_leaves"),
            ("min_child_samples", "min_child_samples"),
            ("early_stopping", "early_stopping_rounds"),
        ],
        "default_dir": "BRAIN_LG_beImpFeat-GS1",
        "run_dirs": [
            "BRAIN_LG_beImpFeat-GS1",
            "BRAIN_LG_beImpFeat-GS2",
            "BRAIN_LG_beImpFeat-GS3",
            "BRAIN_LG_beImpFeat-GS4",
            "BRAIN_LG_beImpFeat-GS5",
            "BRAIN_LG_beImpFeat-GS6",
            #"BRAIN_LG_beImpFeat-GS7",
            #"BRAIN_LG_beImpFeat-GS8"
        ],
    },
    {
        "name": "CatBoost",
        "subdir": ".",
        "hp_cols": [
            ("iterations", "iterations"),
            ("learning_rate", "learning_rate"),
            ("depth", "depth"),
            ("l2_leaf_reg", "l2_leaf_reg"),
            ("min_data_in_leaf", "min_data_in_leaf"),
            ("early_stopping", "early_stopping_rounds"),
        ],
        "default_dir": "BRAIN_CB_beImpFeat-GS1",
        "run_dirs": [
            "BRAIN_CB_beImpFeat-GS1",
            "BRAIN_CB_beImpFeat-GS2",
            "BRAIN_CB_beImpFeat-GS3",
            #"BRAIN_CB_beImpFeat-GS4",
            #"BRAIN_CB_beImpFeat-GS5",
            #"BRAIN_CB_beImpFeat-GS6",
            #"BRAIN_CB_beImpFeat-GS7",
            #"BRAIN_CB_beImpFeat-GS8"
        ],
    },
]

def find_log_file(folder_path, folder_name):
    if os.path.isdir(folder_path):
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(('.out', '.log')) or fname.endswith('_log.txt'):
                return os.path.join(folder_path, fname)
    parent = os.path.dirname(folder_path)
    for suffix in ('_log.txt', '_training_log.txt'):
        cand = os.path.join(parent, folder_name + suffix)
        if os.path.exists(cand):
            return cand
    return None

def parse_default_params(log_path):
    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'RUN 1 - DEFAULT HYPERPARAMETERS' in line and i + 1 < len(lines):
            m = re.search(r'\{.+\}', lines[i + 1])
            if m:
                try:
                    return ast.literal_eval(m.group())
                except Exception:
                    pass
    return {}

def parse_grid_params(log_path):
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'Grid:\s*(\{.+\})', line)
            if m:
                try:
                    return ast.literal_eval(m.group(1))
                except Exception:
                    pass
    return {}

def parse_total_fits(log_path):
    """Total grid search fits from 'totalling N fits' line."""
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'totalling\s+(\d+)\s+fits', line)
            if m:
                return int(m.group(1))
    return None

def load_cv_metrics(folder_path, run_tag):                                                          #(mean_rmse, mean_mae)
    csv_path = os.path.join(folder_path, run_tag, 'some path/file ...', 'some path/file ...')
    if not os.path.exists(csv_path):
        return None, None
    rmse_vals, mae_vals = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rmse_vals.append(float(row['RMSE']))
            mae_vals.append(float(row['MAE']))
    if not rmse_vals:
        return None, None
    return sum(rmse_vals) / len(rmse_vals), sum(mae_vals) / len(mae_vals)

def load_test_metrics(folder_path, run_tag):                                                        #(rmse, mae) from .../test_metrics.json
    json_path = os.path.join(folder_path, run_tag, 'some path/file ...', 'some path/file ...')
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('RMSE'), data.get('MAE')

def coerce(raw):                                                                                    #Normalise a best_search_params.json value to its Python type
    if not isinstance(raw, str):
        return raw
    if raw == 'None': return None
    if raw == 'True': return True
    if raw == 'False': return False
    try: return int(raw)
    except ValueError: pass
    try: return float(raw)
    except ValueError: pass
    return raw

def to_str(v):                                                                                      #Display string for any parameter value
    return 'None' if v is None else str(v)

def eq(a, b):
    return to_str(a) == to_str(b)

def build_default_row(folder_path, folder_name, log_path):
    params = parse_default_params(log_path)
    rmse, mae = load_cv_metrics(folder_path, 'default')
    test_rmse, test_mae = load_test_metrics(folder_path, 'default')
    return {
        'folder': folder_name + '/default',
        'is_default': True,
        'params': params,
        'rmse': rmse,
        'mae': mae,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'gs_duration': None,
        'gs_fits': None,
    }


def build_best_row(folder_path, folder_name, log_path, best_json_path):
    grid = parse_grid_params(log_path)
    fits = parse_total_fits(log_path)
    with open(best_json_path, encoding='utf-8') as f:
        raw = json.load(f)
    best = {k: coerce(v) for k, v in raw.items()}
    rmse, mae = load_cv_metrics(folder_path, 'best')
    test_rmse, test_mae = load_test_metrics(folder_path, 'best')
    return {
        'folder': folder_name,
        'is_default': False,
        'grid': grid,
        'best': best,
        'rmse': rmse,
        'mae': mae,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'gs_duration': best.get('grid_search_duration_s'),
        'gs_fits': fits,
    }

def annotate_metric_flags(rows):
    def _flags(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        min_val = min(vals) if vals else None
        prev = None
        for r in rows:
            v = r[key]
            r[key + '_best'] = v is not None and min_val is not None and abs(v - min_val) < 1e-9
            r[key + '_worse'] = v is not None and prev is not None and v > prev
            if v is not None:
                prev = v
    for key in ('rmse', 'mae', 'test_rmse', 'test_mae'):
        _flags(rows, key)
    return rows

def render_hp_default(value):                                                                       #Single green value for default row
    return f'[<span class="best-param">{to_str(value)}</span>]'

def render_hp_grid(values, best_value):                                                             #Full grid range; best value coloured green
    if not values:
        if best_value is not None:
            return f'[<span class="best-param">{to_str(best_value)}</span>]'
        return '-'
    parts = []
    for v in values:
        s = to_str(v)
        parts.append(f'<span class="best-param">{s}</span>' if eq(v, best_value) else s)
    return '[' + ', '.join(parts) + ']'

def render_metric(value, is_best, is_worse):                                                        #Render an RMSE or MAE value with bold/dark-red CSS classes
    if value is None:
        return '-'
    classes = []
    if is_best: classes.append('metric-best')
    if is_worse: classes.append('metric-worse')
    text = f'{value:.4f}'
    if classes:
        return f'<span class="{" ".join(classes)}">{text}</span>'
    return text

def format_duration(seconds, fits):
    text = f'{seconds / 60 / 60:.2f}&nbsp;hrs'
    if fits:
        text += f'({fits:,}&nbsp;fits)'
    return text

def model_table_html(model_cfg, rows):
    hp_cols = model_cfg['hp_cols']
    col_headers = ([model_cfg['name']] + [d for d, _ in hp_cols] + ['OOF RMSE', 'OOF MAE', 'Test RMSE', 'Test MAE', 'GS duration'])
    ths = ''.join(f'<th>{h}</th>' for h in col_headers)

    body_rows = []
    for row in rows:
        cls = ' class="default-row"' if row['is_default'] else ''
        cells = [f'<td class="folder">{row["folder"]}</td>']

        for _, key in hp_cols:
            if row['is_default']:
                if key in row['params']:
                    cells.append(f'<td>{render_hp_default(row["params"][key])}</td>')
                else:
                    cells.append('<td>-</td>')
            else:
                grid_vals = row['grid'].get(key, [])
                best_val = row['best'].get(key)
                cells.append(f'<td>{render_hp_grid(grid_vals, best_val)}</td>')

        cells.append(f'<td>{render_metric(row["rmse"], row["rmse_best"], row["rmse_worse"])}</td>')
        cells.append(f'<td>{render_metric(row["mae"], row["mae_best"], row["mae_worse"])}</td>')
        cells.append(f'<td>{render_metric(row["test_rmse"], row["test_rmse_best"], row["test_rmse_worse"])}</td>')
        cells.append(f'<td>{render_metric(row["test_mae"], row["test_mae_best"], row["test_mae_worse"])}</td>')

        dur = row.get('gs_duration')
        cells.append(f'<td>{"-" if dur is None else format_duration(dur, row.get("gs_fits"))}</td>')
        body_rows.append(f'<tr{cls}>{"".join(cells)}</tr>')

    return (f'<table>\n<thead><tr>{ths}</tr></thead>\n<tbody>\n' + '\n'.join(body_rows) + '\n</tbody>\n</table>')

_CSS = """
* { box-sizing: border-box; }
body {
    font-family: Arial, sans-serif;
    font-size: 13px;
    margin: 16px 20px;
    color: #111;
}
h1 {
    font-size: 16px;
    margin-bottom: 8px;
}
table {
    border-collapse: collapse;
    white-space: nowrap;
    margin-bottom: 2px;
}
th, td {
    border: 1px solid #bbb;
    padding: 5px 10px;
    text-align: center;
    vertical-align: middle;
}
th { background: #e8e8e8; font-weight: bold; }
td.folder {
    text-align: left;
    font-family: monospace;
    font-size: 11.5px;
    color: #333;
}
tr.default-row td { background: #f3fff3; }
.best-param  { color: #006400; font-weight: bold; }
.metric-best  { font-weight: bold; }
.metric-worse { color: #8b0000; }
.metric-best.metric-worse { color: #8b0000; font-weight: bold; }
"""

def build_page(heading, tables_html):
    body = '\n'.join(tables_html)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        f'<title>{heading}</title>\n'
        f'<style>{_CSS}</style>\n'
        '</head>\n<body>\n'
        f'<h1>{heading}</h1>\n'
        f'{body}\n'
        '</body>\n</html>\n'
    )

def build_model_rows(model_cfg):
    subdir = model_cfg['subdir']
    model_root = os.path.join(BASE_DIR, subdir)
    default_dir = model_cfg.get('default_dir')
    rows = []

    if default_dir:                                                                             #Default row
        fp = os.path.join(model_root, default_dir)
        log = find_log_file(fp, default_dir)
        if log:
            rows.append(build_default_row(fp, default_dir, log))
        else:
            print(f'Achtung [{subdir}]: no log for default_dir "{default_dir}" - skipped')

    for run_dir in model_cfg['run_dirs']:                                                       #Grid-searched rows
        fp = os.path.join(model_root, run_dir)
        if not os.path.isdir(fp):
            print(f'Achtung [{subdir}]: folder not found "{run_dir}" - skipped')
            continue
        log = find_log_file(fp, run_dir)
        if not log:
            print(f'Actung [{subdir}]: no log for "{run_dir}" - skipped')
            continue
        best_json = os.path.join(fp, 'best_search_params.json')
        if not os.path.exists(best_json):
            print(f'Achtung [{subdir}]: best_search_params.json missing in "{run_dir}" - skipped')
            continue
        rows.append(build_best_row(fp, run_dir, log, best_json))
    return annotate_metric_flags(rows)

def main():
    tables_html = []
    for model_cfg in MODELS:
        print(f'Processing {model_cfg["name"]} ...')
        rows = build_model_rows(model_cfg)
        tables_html.append(model_table_html(model_cfg, rows))

    html = build_page(HEADING, tables_html)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nSaving {OUTPUT_HTML}...')

if __name__ == '__main__':
    main()