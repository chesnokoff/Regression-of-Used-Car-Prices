from time import sleep
import os

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd


def submit_and_list():
    api = KaggleApi()
    api.authenticate()

    api.competition_submit(
        file_name="submission.csv",
        message="CI",
        competition="playground-series-s4e9",
    )
    print("Submission uploaded.")

    sleep(60)

    submissions = api.competition_submissions("playground-series-s4e9")

    rows = []
    for s in submissions:
        rows.append(
            {
                "date": s.date,
                "description": s.description,
                "public_score": s.public_score,
                "private_score": s.private_score,
                "status": str(s.status),
            }
        )

    df = pd.DataFrame(rows)

    print("::group::Kaggle submissions (table)")
    print(df.to_string(index=False))
    print("::endgroup::")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        df["status"] = df["status"].str.replace("SubmissionStatus.", "", regex=False)

        md_table = df.to_markdown(index=False)

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("## Kaggle submissions\n\n")
            f.write(md_table)
            f.write("\n\n")


if __name__ == "__main__":
    submit_and_list()
