import base64
import datetime
import logging
import os
import time
import webbrowser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import schedule

logging.basicConfig(
    filename="report.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

HEADLINES = {
    "schools": "Top-performing schools stand out clearly for field lab and park trip incentives",
    "engagement": "Schools with steadier outdoor and teacher participation completed more learning activities",
    "grades": "Lower and middle grades led activity uptake while upper grades need targeted follow-up",
}


def ensure_sample_education_data(csv_path: str = "data/conservation_education.csv") -> None:
    """Creates sample school-level conservation education data if the CSV does not exist yet."""
    if os.path.exists(csv_path):
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rng = np.random.default_rng(42)

    schools = [
        ("Kibwezi Primary", "Makueni"),
        ("Makindu Junior", "Makueni"),
        ("Tsavo View School", "Taita Taveta"),
        ("Voi Central Junior", "Taita Taveta"),
        ("Kajiado Model", "Kajiado"),
        ("Kitengela Plains", "Kajiado"),
    ]
    grades = ["Grade 5-6", "Grade 7-8", "Grade 9"]
    rows = []

    for school, county in schools:
        for grade in grades:
            for i in range(10):
                outdoor_sessions = int(rng.integers(2, 13))
                teacher_score = round(float(np.clip(55 + outdoor_sessions * 3.2 + rng.normal(0, 6), 40, 100)), 1)
                completion = round(
                    float(np.clip(25 + outdoor_sessions * 5.2 + (teacher_score - 60) * 0.35 + rng.normal(0, 7), 15, 100)),
                    1,
                )
                rows.append({
                    "school_name": school,
                    "county": county,
                    "grade_level": grade,
                    "outdoor_sessions_attended": outdoor_sessions,
                    "teacher_engagement_score": teacher_score,
                    "activity_completion_pct": completion,
                    "safety_compliance_ok": True,
                })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info(f"Created sample conservation education dataset at {csv_path}")


def generate_report(csv_path: str = "data/conservation_education.csv") -> dict:
    ensure_sample_education_data(csv_path)
    os.makedirs("charts", exist_ok=True)

    # 1. Load fresh data
    df = pd.read_csv(csv_path)

    # 2. Run analysis: rank schools for incentives, check engagement, compare grades
    school_rank = (
        df.groupby("school_name")["activity_completion_pct"]
        .mean()
        .sort_values(ascending=True)
    )
    top_schools = school_rank.tail(2).index.tolist()

    grade_order = ["Grade 5-6", "Grade 7-8", "Grade 9"]
    grade_county = (
        df.pivot_table(
            index="grade_level",
            columns="county",
            values="activity_completion_pct",
            aggfunc="mean",
        )
        .reindex(grade_order)
    )

    # 3. Generate three crisp charts as PNG files
    # Chart 1: School ranking for incentive shortlist (Horizontal bar chart)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    colors = ["#1F4E79" if s in top_schools else "#AED6F1" for s in school_rank.index]
    school_rank.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(HEADLINES["schools"], fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Average Activity Completion (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("charts/school_incentive_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Outdoor engagement vs. activity completion (Scatter + trend line)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    ax.scatter(
        df["outdoor_sessions_attended"],
        df["activity_completion_pct"],
        alpha=0.65,
        color="#2E86C1",
        edgecolors="none",
        label="School Cohorts",
    )
    slope, intercept = np.polyfit(df["outdoor_sessions_attended"], df["activity_completion_pct"], 1)
    x_vals = np.linspace(df["outdoor_sessions_attended"].min(), df["outdoor_sessions_attended"].max(), 100)
    ax.plot(x_vals, slope * x_vals + intercept, color="#1F4E79", linewidth=2.2, label="Overall trend")
    ax.set_title(HEADLINES["engagement"], fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Outdoor Learning Sessions Attended")
    ax.set_ylabel("Activity Completion (%)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("charts/engagement_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Chart 3: Grade & county delivery comparison (Grouped bar chart)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    grade_county.plot(kind="bar", ax=ax, color=["#1F4E79", "#2E86C1", "#85C1E9"])
    ax.set_title(HEADLINES["grades"], fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Average Activity Completion (%)")
    plt.xticks(rotation=0)
    ax.legend(title="County", frameon=False)
    plt.tight_layout()
    plt.savefig("charts/grade_county_delivery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "schools_count": df["school_name"].nunique(),
        "top_schools": ", ".join(reversed(top_schools)),
        "compliance_rate": round(df["safety_compliance_ok"].mean() * 100, 1),
    }


def encode_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html_report(data: dict, report_date_str: str) -> str:
    os.makedirs("reports", exist_ok=True)

    img_schools = encode_png("charts/school_incentive_ranking.png")
    img_engagement = encode_png("charts/engagement_outcomes.png")
    img_grades = encode_png("charts/grade_county_delivery.png")

    summary_paragraph = (
        f"Across {data['schools_count']} partner schools tracked ({data['compliance_rate']}% safety and activity compliance), "
        f"{HEADLINES['schools'].lower()}, {HEADLINES['engagement'].lower()}, and "
        f"{HEADLINES['grades'].lower()}—giving program leads a clear shortlist for field lab or park trip incentives "
        f"alongside plain-language evidence for internal learning and donor updates."
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Conservation Education Briefing — {report_date_str}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 860px;
      margin: 32px auto;
      padding: 0 20px;
      color: #1f2933;
      line-height: 1.5;
    }}
    h1 {{ margin-bottom: 6px; }}
    .meta {{ color: #52606d; font-size: 14px; margin-bottom: 20px; }}
    .summary {{
      background: #f0f4f8;
      border-left: 4px solid #1f4e79;
      padding: 14px 16px;
      margin-bottom: 28px;
    }}
    .finding {{ margin-bottom: 32px; }}
    .finding h2 {{ font-size: 17px; margin-bottom: 10px; color: #1f4e79; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #e4e7eb; }}
    @media print {{
      body {{ margin: 0; padding: 0; }}
    }}
  </style>
</head>
<body>
  <h1>Conservation Education Briefing — {report_date_str}</h1>
  <div class="meta">Internal Learning &amp; Donor Story Summary | Top Incentive Shortlist: {data['top_schools']}</div>

  <div class="summary">
    <p>{summary_paragraph}</p>
  </div>

  <div class="finding">
    <h2>{HEADLINES['schools']}</h2>
    <img src="data:image/png;base64,{img_schools}" alt="{HEADLINES['schools']}">
  </div>

  <div class="finding">
    <h2>{HEADLINES['engagement']}</h2>
    <img src="data:image/png;base64,{img_engagement}" alt="{HEADLINES['engagement']}">
  </div>

  <div class="finding">
    <h2>{HEADLINES['grades']}</h2>
    <img src="data:image/png;base64,{img_grades}" alt="{HEADLINES['grades']}">
  </div>
</body>
</html>"""

    output_path = os.path.join("reports", f"report_{report_date_str}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {output_path}")
    return output_path


def daily_job(custom_date: str | None = None) -> str:
    try:
        report_date_str = custom_date or datetime.date.today().strftime("%Y-%m-%d")
        data = generate_report()
        output_path = build_html_report(data, report_date_str)
        webbrowser.open("file://" + os.path.abspath(output_path))
        logging.info(f"Report generated and saved OK: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Report failed: {e}")
        raise


if __name__ == "__main__":
    daily_job()
    schedule.every().monday.at("07:00").do(daily_job)

    while True:
        schedule.run_pending()
        time.sleep(60)