"""Web Mercantile warehouse leasing optimizer.

Deploy this file to Streamlit Community Cloud from a GitHub repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog


st.set_page_config(
    page_title="Warehouse Leasing Optimizer",
    page_icon="🏭",
    layout="wide",
)


DEFAULT_REQUIREMENTS = [30_000, 20_000, 40_000, 10_000, 50_000]
DEFAULT_PERIOD_COSTS = [65, 100, 135, 160, 190]


def money(value: float) -> str:
    """Format a numeric amount as US dollars with no decimal cents."""
    return f"${value:,.0f}"


def build_lease_options(months: int, period_costs: list[float]) -> pd.DataFrame:
    """Create one decision-variable row for every feasible lease interval."""
    rows = []
    for start in range(months):
        for end in range(start, months):
            duration = end - start + 1
            rows.append(
                {
                    "Variable": f"x{len(rows) + 1}",
                    "Start Month": start + 1,
                    "End Month": end + 1,
                    "Lease Length": duration,
                    "Cost / Sq. Ft.": float(period_costs[duration - 1]),
                }
            )
    return pd.DataFrame(rows)


def solve_lease_problem(
    requirements: list[float], period_costs: list[float]
) -> tuple[float, pd.DataFrame, np.ndarray]:
    """Solve the continuous linear program for the least-cost lease plan."""
    months = len(requirements)
    options = build_lease_options(months, period_costs)
    coverage = np.zeros((months, len(options)))

    for column, option in options.iterrows():
        start = int(option["Start Month"]) - 1
        end = int(option["End Month"])
        coverage[start:end, column] = 1

    result = linprog(
        c=options["Cost / Sq. Ft."].to_numpy(dtype=float),
        A_ub=-coverage,
        b_ub=-np.asarray(requirements, dtype=float),
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    options["Square Feet"] = result.x
    options["Lease Cost"] = options["Square Feet"] * options["Cost / Sq. Ft."]
    options["Active Months"] = options.apply(
        lambda row: f"{int(row['Start Month'])}–{int(row['End Month'])}", axis=1
    )
    coverage_by_month = coverage @ result.x
    return float(result.fun), options, coverage_by_month


def make_capacity_chart(
    requirements: list[float], coverage_by_month: np.ndarray
) -> go.Figure:
    """Build a Cartesian chart of demand, leased capacity, and slack."""
    months = list(range(1, len(requirements) + 1))
    slack = coverage_by_month - np.asarray(requirements)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months,
            y=requirements,
            mode="lines+markers",
            name="Required space",
            line={"color": "#C0392B", "width": 3},
            marker={"size": 9},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=coverage_by_month,
            mode="lines+markers",
            name="Leased capacity",
            line={"color": "#1F77B4", "width": 3, "shape": "hv"},
            marker={"size": 9},
        )
    )
    fig.add_trace(
        go.Bar(
            x=months,
            y=slack,
            name="Unused capacity",
            marker_color="#8EC7A5",
            opacity=0.55,
        )
    )
    fig.update_layout(
        title="Cartesian view: monthly constraints and leased capacity",
        xaxis={"title": "Month", "dtick": 1, "tickmode": "linear"},
        yaxis={"title": "Square feet", "rangemode": "tozero"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 10, "r": 10, "t": 90, "b": 10},
        height=470,
        template="plotly_white",
    )
    return fig


st.title("🏭 Warehouse Leasing Optimizer")
st.write(
    "Find the least-cost combination of warehouse leases that satisfies every "
    "month's space requirement. A lease may start in any month and end in any "
    "later month."
)

with st.sidebar:
    st.header("Model inputs")
    st.caption("Edit the assumptions, then click **Solve model**.")

    months = st.number_input(
        "Planning horizon (months)", min_value=1, max_value=12, value=5, step=1
    )
    months = int(months)

    requirements_df = pd.DataFrame(
        {
            "Month": list(range(1, months + 1)),
            "Required Space (sq. ft.)": (
                DEFAULT_REQUIREMENTS[:months]
                if months <= len(DEFAULT_REQUIREMENTS)
                else [0] * months
            ),
        }
    )
    requirements_df = st.data_editor(
        requirements_df,
        hide_index=True,
        disabled=["Month"],
        column_config={
            "Required Space (sq. ft.)": st.column_config.NumberColumn(
                min_value=0, step=1_000, format="%d"
            )
        },
        key="requirements",
    )

    default_costs = DEFAULT_PERIOD_COSTS[:months]
    if months > len(DEFAULT_PERIOD_COSTS):
        default_costs += [DEFAULT_PERIOD_COSTS[-1]] * (
            months - len(DEFAULT_PERIOD_COSTS)
        )
    costs_df = pd.DataFrame(
        {
            "Lease Length (months)": list(range(1, months + 1)),
            "Cost / Sq. Ft.": default_costs,
        }
    )
    costs_df = st.data_editor(
        costs_df,
        hide_index=True,
        disabled=["Lease Length (months)"],
        column_config={
            "Cost / Sq. Ft.": st.column_config.NumberColumn(
                min_value=0, step=1, format="$%.2f"
            )
        },
        key="costs",
    )
    solve_clicked = st.button("Solve model", type="primary", use_container_width=True)

if "solution" not in st.session_state or solve_clicked:
    try:
        requirements = requirements_df["Required Space (sq. ft.)"].astype(float).tolist()
        period_costs = costs_df["Cost / Sq. Ft."].astype(float).tolist()
        if any(value < 0 for value in requirements + period_costs):
            raise ValueError("Requirements and costs must be nonnegative.")
        if any(cost == 0 for cost in period_costs):
            raise ValueError("Lease costs must be greater than zero.")
        solution = solve_lease_problem(requirements, period_costs)
        st.session_state.solution = solution
        st.session_state.inputs = (requirements, period_costs)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Unable to solve the model: {exc}")
        st.stop()

total_cost, options, coverage_by_month = st.session_state.solution
requirements, period_costs = st.session_state.inputs
active = options[options["Square Feet"] > 0.01].copy()
active["Square Feet"] = active["Square Feet"].round(2)
active["Lease Cost"] = active["Lease Cost"].round(2)

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Minimum total cost", money(total_cost))
metric_2.metric("Lease decisions used", f"{len(active)}")
metric_3.metric("Peak required space", f"{max(requirements):,.0f} sq. ft.")

st.subheader("Optimal lease plan")
st.dataframe(
    active[
        [
            "Variable",
            "Active Months",
            "Lease Length",
            "Square Feet",
            "Cost / Sq. Ft.",
            "Lease Cost",
        ]
    ].rename(
        columns={
            "Active Months": "Months Covered",
            "Lease Length": "Length (months)",
            "Square Feet": "Square Feet Leased",
            "Cost / Sq. Ft.": "Cost / Sq. Ft.",
            "Lease Cost": "Total Lease Cost",
        }
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Square Feet Leased": st.column_config.NumberColumn(format="%,.0f"),
        "Cost / Sq. Ft.": st.column_config.NumberColumn(format="$%.2f"),
        "Total Lease Cost": st.column_config.NumberColumn(format="$%,.0f"),
    },
)

st.plotly_chart(make_capacity_chart(requirements, coverage_by_month), use_container_width=True)

constraint_df = pd.DataFrame(
    {
        "Month": range(1, len(requirements) + 1),
        "Required Space": requirements,
        "Leased Capacity": coverage_by_month,
        "Unused Capacity": coverage_by_month - np.asarray(requirements),
    }
)
constraint_df["Feasible?"] = constraint_df["Leased Capacity"] >= constraint_df["Required Space"] - 1e-6

with st.expander("Show monthly constraint calculations"):
    st.dataframe(
        constraint_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Required Space": st.column_config.NumberColumn(format="%,.0f"),
            "Leased Capacity": st.column_config.NumberColumn(format="%,.0f"),
            "Unused Capacity": st.column_config.NumberColumn(format="%,.0f"),
        },
    )

st.subheader("How the model works")
st.latex(
    r"\min \sum_{i=1}^{T}\sum_{j=i}^{T} c_{j-i+1}x_{ij}"
)
st.markdown(
    "For every month **t**, the constraint is "
    r"$\sum_{i\leq t\leq j}x_{ij}\geq d_t$. "
    "Here, $x_{ij}$ is square footage leased from month $i$ through month "
    "$j$, $c$ is the cost per square foot for that lease length, and $d_t$ "
    "is the required space in month $t$."
)
st.caption("All lease quantities are continuous square-foot decisions; fractional square feet are rounded only for display.")
