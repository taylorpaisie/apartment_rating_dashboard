"""
Apartment Tour Scorer – Plotly Dash App (post-tour survey, fancy UI)
-------------------------------------------------------------------
Purpose: After each tour, log quick ratings and details. The app stores entries,
lets you weight what matters, and visualizes matches — now with a sleek UI.

Run locally:
    pip install dash plotly pandas numpy dash-bootstrap-components
    python apartment_chooser_dash_app.py

Render start command:
    gunicorn apartment_chooser_dash_app:server
"""
from __future__ import annotations

import os, io, base64, datetime as dt
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

# ---------------------------- Defaults ----------------------------
DEFAULT_WEIGHTS = {
    "Price": 0.22,
    "Commute": 0.16,
    "Vibe": 0.16,
    "Sunlight": 0.10,
    "Safety": 0.12,
    "Cleanliness": 0.10,
    "Amenities": 0.10,
    "Noise": 0.04,
}
AMENITIES = ["pets_ok", "parking", "laundry_in_unit", "gym", "pool", "storage", "balcony"]
BOOL_TRUE = {"true", "t", "yes", "y", "1", "x"}

# Global Plotly look & feel (pairs well with dark Bootswatch themes)
pio.templates.default = "plotly_dark"

def _scale(s: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    vmin, vmax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - vmin) / (vmax - vmin)
    return 1 - out if invert else out

# ---------------------------- Scoring ----------------------------
def compute_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    d = df.copy()
    # type coercion
    for c in ["rent","sqft","commute_min","vibe","sunlight","noise","smell","safety_seen","cleanliness","building_age","floors"]:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in AMENITIES:
        if c in d: d[c] = d[c].apply(lambda x: (str(x).strip().lower() in BOOL_TRUE) if isinstance(x, str) else bool(x))

    # amenity score
    amen_cols = [c for c in AMENITIES if c in d]
    d["amenity_score"] = d[amen_cols].mean(axis=1) if amen_cols else 0

    # scaled components
    d["price_scaled"]   = _scale(d.get("rent",         pd.Series([np.nan]*len(d))), invert=True)
    d["commute_scaled"] = _scale(d.get("commute_min",  pd.Series([np.nan]*len(d))), invert=True)
    d["vibe_scaled"]    = _scale(d.get("vibe",         pd.Series([np.nan]*len(d))))
    d["sun_scaled"]     = _scale(d.get("sunlight",     pd.Series([np.nan]*len(d))))
    d["safety_scaled"]  = _scale(d.get("safety_seen",  pd.Series([np.nan]*len(d))))
    d["clean_scaled"]   = _scale(d.get("cleanliness",  pd.Series([np.nan]*len(d))))
    d["noise_scaled"]   = _scale(d.get("noise",        pd.Series([np.nan]*len(d))), invert=True)
    d["amen_scaled"]    = d["amenity_score"].fillna(0.0)

    w = weights
    d["score"] = (
        w["Price"]      * d["price_scaled"]   +
        w["Commute"]    * d["commute_scaled"] +
        w["Vibe"]       * d["vibe_scaled"]    +
        w["Sunlight"]   * d["sun_scaled"]     +
        w["Safety"]     * d["safety_scaled"]  +
        w["Cleanliness"]* d["clean_scaled"]   +
        w["Amenities"]  * d["amen_scaled"]    +
        w["Noise"]      * d["noise_scaled"]
    )
    return d

# ---------------------------- App ----------------------------
# Dark theme: SLATE (swap to LUX/FLATLY for light)
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = app.server

# Health check (Render-friendly)
@server.route("/healthz")
def health():
    return "OK"

# --------------- Layout (Navbar + Cards + Tabs + Spinners) ---------------
app.layout = dbc.Container([
    dcc.Store(id="tour-store", storage_type="local"),
    dcc.Download(id="dl-csv"),

    # Navbar
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Apartment Tour Scorer", className="fw-semibold"),
        ]),
        color="primary", dark=True, sticky="top", className="mb-3 rounded"
    ),

    html.Div("Log each tour, tweak weights, and compare like a data-driven adulting wizard 🧙‍♂️.",
             className="text-muted mb-3"),

    dbc.Row([
        # Left: Survey card
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Post-tour survey", className="fw-semibold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Name of property"),
                            dbc.Input(id="name", placeholder="e.g., Midtown Square"),
                            dbc.FormText("Required"),
                        ], md=12),
                        dbc.Col([
                            dbc.Label("Date toured"),
                            dcc.DatePickerSingle(
                                id="date_toured",
                                display_format="YYYY-MM-DD",
                                date=pd.Timestamp.today().date()
                            ),
                        ], md=12, className="mt-2"),
                        dbc.Col([
                            dbc.Label("Unit (optional)"),
                            dbc.Input(id="unit", placeholder="#504"),
                        ], md=6, className="mt-2"),
                        dbc.Col([
                            dbc.Label("Neighborhood (optional)"),
                            dbc.Input(id="neighborhood", placeholder="Corktown / Midtown / …"),
                        ], md=6, className="mt-2"),
                        dbc.Col([
                            dbc.Label("Rent ($/mo)"),
                            dbc.Input(id="rent", type="number", min=0, step=25, placeholder="1900"),
                        ], md=4, className="mt-2"),
                        dbc.Col([
                            dbc.Label("Size (sqft)"),
                            dbc.Input(id="sqft", type="number", min=0, step=10),
                        ], md=4, className="mt-2"),
                        dbc.Col([
                            dbc.Label("Commute (min)"),
                            dbc.Input(id="commute_min", type="number", min=0, step=1),
                        ], md=4, className="mt-2"),
                    ]),

                    html.Hr(),
                    html.Div("Quick ratings (0=bad, 10=chef’s kiss)", className="text-muted mb-1"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Overall vibe"), dcc.Slider(0,10,1, value=7, id="vibe")], md=6),
                        dbc.Col([dbc.Label("Sunlight"), dcc.Slider(0,10,1, value=7, id="sunlight")], md=6),
                        dbc.Col([dbc.Label("Noise (higher=worse)"), dcc.Slider(0,10,1, value=4, id="noise")], md=6, className="mt-3"),
                        dbc.Col([dbc.Label("Smell (higher=worse)"), dcc.Slider(0,10,1, value=2, id="smell")], md=6, className="mt-3"),
                        dbc.Col([dbc.Label("Safety seen"), dcc.Slider(0,10,1, value=7, id="safety_seen")], md=6, className="mt-3"),
                        dbc.Col([dbc.Label("Cleanliness"), dcc.Slider(0,10,1, value=7, id="cleanliness")], md=6, className="mt-3"),
                    ], className="gx-3"),

                    html.Details([
                        html.Summary("Amenities"),
                        dcc.Checklist(
                            id="amenities",
                            options=[
                                {"label":"Pets OK","value":"pets_ok"},
                                {"label":"Parking","value":"parking"},
                                {"label":"Laundry in unit","value":"laundry_in_unit"},
                                {"label":"Gym","value":"gym"},
                                {"label":"Pool","value":"pool"},
                                {"label":"Storage","value":"storage"},
                                {"label":"Balcony","value":"balcony"},
                            ],
                            value=["pets_ok","parking"], inline=True
                        ),
                    ], className="mt-3"),

                    dbc.Label("Notes", className="mt-2"),
                    dcc.Textarea(id="notes", style={"width":"100%","height":"80px"},
                                 placeholder="Flooring, odors, neighbor vibes, leasing agent tea…"),

                    dbc.ButtonGroup([
                        dbc.Button("Add entry", id="btn-add", color="primary"),
                        dbc.Button("Update selected", id="btn-update", color="secondary", outline=True, className="ms-2"),
                        dbc.Button("Delete selected", id="btn-delete", color="danger", outline=True, className="ms-2"),
                    ], className="mt-3"),

                    # Toast (status messages)
                    dbc.Toast(
                        id="form-msg",
                        header="Status",
                        is_open=False,
                        dismissable=True,
                        duration=4000,
                        icon="info",
                        className="mt-3"
                    ),
                ])
            ]),
            md=4
        ),

        # Right: Weights + table + visuals in tabs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Weights", className="fw-semibold"),
                dbc.CardBody([
                    html.Div(id="weight-sum", className="text-muted mb-2"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Price"),      dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Price"],      id="w_price")],   md=6),
                        dbc.Col([dbc.Label("Commute"),    dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Commute"],    id="w_commute")], md=6),
                        dbc.Col([dbc.Label("Vibe"),       dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Vibe"],       id="w_vibe")],    md=6, className="mt-2"),
                        dbc.Col([dbc.Label("Sunlight"),   dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Sunlight"],   id="w_sun")],     md=6, className="mt-2"),
                        dbc.Col([dbc.Label("Safety"),     dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Safety"],     id="w_safe")],    md=6, className="mt-2"),
                        dbc.Col([dbc.Label("Cleanliness"),dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Cleanliness"],id="w_clean")],  md=6, className="mt-2"),
                        dbc.Col([dbc.Label("Amenities"),  dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Amenities"],  id="w_amen")],    md=6, className="mt-2"),
                        dbc.Col([dbc.Label("Noise"),      dcc.Slider(0,1,0.02,value=DEFAULT_WEIGHTS["Noise"],      id="w_noise")],   md=6, className="mt-2"),
                    ]),
                    html.Hr(),
                    dbc.Button("Export CSV", id="btn-export", color="success"),
                    dbc.Button("Import CSV", id="btn-import", outline=True, color="primary", className="ms-2"),
                    dcc.Upload(
                        id="uploader",
                        children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
                        multiple=False,
                        style={"border":"1px dashed #999","padding":"8px","marginLeft":"8px","display":"inline-block"}
                    ),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Your tours", className="fw-semibold"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="tour-table",
                        page_size=8,
                        row_selectable="single",
                        sort_action="native",
                        style_table={"overflowX":"auto"},
                        style_cell={"fontFamily":"system-ui","fontSize":"13px"},
                    ),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Visuals", className="fw-semibold"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(dbc.Spinner(dcc.Graph(id="ranked",  style={"height":"360px"}, config={"displayModeBar": False}), size="sm"), label="Ranked by score", tab_id="tab-ranked"),
                        dbc.Tab(dbc.Spinner(dcc.Graph(id="radar",   style={"height":"360px"}, config={"displayModeBar": False}), size="sm"), label="Radar profile",  tab_id="tab-radar"),
                        dbc.Tab(dbc.Spinner(dcc.Graph(id="scatter", style={"height":"360px"}, config={"displayModeBar": False}), size="sm"), label="Price vs Vibe", tab_id="tab-scatter"),
                        dbc.Tab(dbc.Spinner(dcc.Graph(id="timeline",style={"height":"320px"}, config={"displayModeBar": False}), size="sm"), label="Timeline",      tab_id="tab-time"),
                    ], active_tab="tab-ranked")
                ])
            ]),
        ], md=8),
    ], className="g-3"),

    html.Footer("Tip: If “cozy” means you can open the fridge from bed, deduct 2 vibe points.",
                className="text-muted mt-4 mb-3"),
], fluid=True, className="py-3")

# ---------------------------- Helpers ----------------------------
def _weights_from_state(values: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Price":       float(values.get("w_price",   DEFAULT_WEIGHTS["Price"])),
        "Commute":     float(values.get("w_commute", DEFAULT_WEIGHTS["Commute"])),
        "Vibe":        float(values.get("w_vibe",    DEFAULT_WEIGHTS["Vibe"])),
        "Sunlight":    float(values.get("w_sun",     DEFAULT_WEIGHTS["Sunlight"])),
        "Safety":      float(values.get("w_safe",    DEFAULT_WEIGHTS["Safety"])),
        "Cleanliness": float(values.get("w_clean",   DEFAULT_WEIGHTS["Cleanliness"])),
        "Amenities":   float(values.get("w_amen",    DEFAULT_WEIGHTS["Amenities"])),
        "Noise":       float(values.get("w_noise",   DEFAULT_WEIGHTS["Noise"])),
    }

def _strip(s):
    return (str(s).strip() if s is not None else "")

def _parse_currency(val):
    if val is None or val == "":
        return None
    s = str(val)
    s = s.replace(",", "").replace("$", "").replace("USD", "").strip()
    try:
        return int(float(s))
    except Exception:
        return None

def _parse_int(val):
    try:
        return int(val) if val is not None and str(val) != "" else None
    except Exception:
        return None

# ---------------------------- Callbacks ----------------------------
@app.callback(
    Output("weight-sum","children"),
    Input("w_price","value"),Input("w_commute","value"),Input("w_vibe","value"),Input("w_sun","value"),
    Input("w_safe","value"),Input("w_clean","value"),Input("w_amen","value"),Input("w_noise","value")
)
def _sum_weights(*vals):
    total = float(np.nansum(vals)) if vals else 0.0
    return f"Total weight = {total:.2f} (no need to be exactly 1.00; we normalize features)"

@app.callback(
    # Store + Toast (is_open, children, icon, header) + Table data/columns
    Output("tour-store","data"),
    Output("form-msg","is_open"),
    Output("form-msg","children"),
    Output("form-msg","icon"),
    Output("form-msg","header"),
    Output("tour-table","data"),
    Output("tour-table","columns"),
    Input("btn-add","n_clicks"), Input("btn-update","n_clicks"), Input("btn-delete","n_clicks"),
    Input("btn-import","n_clicks"), Input("uploader","contents"), State("uploader","filename"),
    State("tour-store","data"),
    State("name","value"), State("date_toured","date"), State("unit","value"), State("neighborhood","value"),
    State("rent","value"), State("sqft","value"), State("commute_min","value"),
    State("vibe","value"), State("sunlight","value"), State("noise","value"), State("smell","value"),
    State("safety_seen","value"), State("cleanliness","value"), State("amenities","value"), State("notes","value"),
    State("tour-table","selected_rows")
)
def upsert_entry(n_add, n_update, n_delete, n_import, upload_contents, upload_name,
                 store, name, date_toured, unit, neighborhood, rent, sqft, commute_min,
                 vibe, sunlight, noise, smell, safety_seen, cleanliness, amenities, notes, selected_rows):

    store = store or []
    df = pd.DataFrame(store)
    trigger = ctx.triggered_id

    # Import CSV
    if trigger in ("btn-import","uploader") and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        df_new = pd.read_csv(io.BytesIO(decoded))
        df_new.columns = [c.strip().lower() for c in df_new.columns]
        df = pd.concat([df, df_new], ignore_index=True, sort=False)
        cols = [{"name":c,"id":c} for c in sorted(df.columns)]
        return df.to_dict("records"), True, f"Imported {upload_name} ({df_new.shape[0]} rows)", "info", "Status", df.to_dict("records"), cols

    # Delete selected
    if trigger == "btn-delete" and selected_rows:
        df = df.drop(df.index[selected_rows[0]]).reset_index(drop=True)
        cols = [{"name":c,"id":c} for c in sorted(df.columns)]
        return df.to_dict("records"), True, "Deleted selected entry", "danger", "Status", df.to_dict("records"), cols

    # Add/Update
    if trigger in ("btn-add","btn-update"):
        if not name or not date_toured:
            cols = [{"name":c,"id":c} for c in sorted(df.columns)]
            return store, True, "Please provide at least name and date.", "warning", "Status", df.to_dict("records"), cols

        entry = {
            "name": _strip(name),
            "date_toured": pd.to_datetime(date_toured).date().isoformat(),
            "unit": _strip(unit),
            "neighborhood": _strip(neighborhood),
            "rent": _parse_currency(rent),
            "sqft": _parse_int(sqft),
            "commute_min": _parse_int(commute_min),
            "vibe": vibe, "sunlight": sunlight, "noise": noise, "smell": smell,
            "safety_seen": safety_seen, "cleanliness": cleanliness,
            "notes": _strip(notes),
        }
        for a in AMENITIES:
            entry[a] = (a in (amenities or []))

        if trigger == "btn-add":
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            msg, icon = "Added entry", "success"
        else:
            if selected_rows:
                idx = selected_rows[0]
                for k,v in entry.items():
                    df.loc[idx, k] = v
                msg, icon = "Updated selected entry", "info"
            else:
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
                msg, icon = "No row selected; added as new entry", "secondary"

        cols = [{"name":c,"id":c} for c in sorted(df.columns)]
        return df.to_dict("records"), True, msg, icon, "Status", df.to_dict("records"), cols

    # Default: no change
    cols = [{"name":c,"id":c} for c in sorted(df.columns)]
    return df.to_dict("records"), no_update, no_update, no_update, no_update, df.to_dict("records"), cols

@app.callback(
    Output("ranked","figure"), Output("radar","figure"), Output("scatter","figure"), Output("timeline","figure"),
    Input("tour-store","data"),
    Input("tour-table","selected_rows"),
    Input("w_price","value"),Input("w_commute","value"),Input("w_vibe","value"),Input("w_sun","value"),
    Input("w_safe","value"),Input("w_clean","value"),Input("w_amen","value"),Input("w_noise","value"),
)
def update_viz(store, sel_rows, w_price,w_commute,w_vibe,w_sun,w_safe,w_clean,w_amen,w_noise):
    df = pd.DataFrame(store or [])
    if df.empty:
        def _placeholder(title, h=360):
            fig = go.Figure()
            fig.update_layout(title=title, height=h, margin=dict(l=60,r=20,t=50,b=40))
            fig.add_annotation(text="No tours yet — add one on the left.",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        return (_placeholder("Ranked by composite score"),
                _placeholder("Profile vs Weights", 360),
                _placeholder("Price vs Vibe"),
                _placeholder("Tour timeline", 320))

    weights = _weights_from_state({
        "w_price":w_price,"w_commute":w_commute,"w_vibe":w_vibe,"w_sun":w_sun,
        "w_safe":w_safe,"w_clean":w_clean,"w_amen":w_amen,"w_noise":w_noise,
    })
    scored = compute_scores(df, weights)

    # Ranked bar
    top = scored.sort_values("score", ascending=False).copy()
    fig_rank = go.Figure(go.Bar(x=top["score"].round(3), y=top["name"].astype(str), orientation="h"))
    fig_rank.update_layout(
        title="Ranked by composite score",
        xaxis_title="Score (0–1)",
        margin=dict(l=120,r=20,t=50,b=40),
        height=360
    )

    # Radar (selected or top[0]) vs weights
    row = scored.iloc[sel_rows[0]] if sel_rows else scored.iloc[0]
    radar_metrics = [
        ("Price","price_scaled"),("Commute","commute_scaled"),("Vibe","vibe_scaled"),
        ("Sunlight","sun_scaled"),("Safety","safety_scaled"),("Cleanliness","clean_scaled"),
        ("Amenities","amen_scaled"),("Noise","noise_scaled"),
    ]
    theta = [m[0] for m in radar_metrics]
    r_sel = [float(row[m[1]]) if pd.notna(row[m[1]]) else 0.5 for m in radar_metrics]
    r_wts = [float(weights[m[0]]) for m in radar_metrics]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=r_sel, theta=theta, fill='toself', name=row.get('name','(n/a)')))
    fig_radar.add_trace(go.Scatterpolar(r=r_wts, theta=theta, fill='toself', name='Your weights'))
    fig_radar.update_layout(
        title=f"Profile vs Weights — {row.get('name','(n/a)')}",
        polar=dict(radialaxis=dict(range=[0,1])),
        margin=dict(l=20,r=20,t=50,b=30),
        height=360
    )

    # Scatter: Price vs Vibe (bubble size = amenities)
    fig_scatter = go.Figure(go.Scatter(
        x=scored.get("rent", pd.Series([np.nan]*len(scored))),
        y=scored.get("vibe", pd.Series([np.nan]*len(scored))),
        mode="markers+text",
        text=scored.get("name",""),
        textposition="top center",
        marker=dict(size=10 + 20*scored.get("amenity_score",0).fillna(0))
    ))
    fig_scatter.update_layout(
        title="Price vs Vibe (bubble size = amenities)",
        xaxis_title="Rent ($/mo)",
        yaxis_title="Vibe (0–10)",
        margin=dict(l=60,r=20,t=50,b=50),
        height=360
    )

    # Timeline
    tdf = scored.copy()
    tdf["date_toured"] = pd.to_datetime(tdf.get("date_toured"), errors="coerce")
    tdf = tdf.dropna(subset=["date_toured"]) if not tdf.empty else tdf
    fig_time = go.Figure(go.Scatter(
        x=tdf.get("date_toured"),
        y=tdf.get("score"),
        mode="lines+markers+text",
        text=tdf.get("name"),
        textposition="top center"
    ))
    fig_time.update_layout(
        title="Tour timeline (score over time)",
        xaxis_title="Date",
        yaxis_title="Score",
        margin=dict(l=60,r=20,t=50,b=50),
        height=320
    )

    return fig_rank, fig_radar, fig_scatter, fig_time

@app.callback(
    Output("dl-csv","data"),
    Input("btn-export","n_clicks"),
    State("tour-store","data"),
    prevent_initial_call=True
)
def export_csv(n, data):
    if not data:
        return no_update
    df = pd.DataFrame(data)
    csv = df[sorted(df.columns)].to_csv(index=False)
    return dict(content=csv, filename="apartment_tours.csv")

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8050)),
        debug=True
    )
