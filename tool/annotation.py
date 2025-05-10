import argparse
import math
from pathlib import Path

import pandas as pd
import streamlit as st


def main():
    parser = argparse.ArgumentParser(
        description="Annotation tool for video classification"
    )
    parser.add_argument("-v", "--video", default=None)
    parser.add_argument("-l", "--label", default=None)
    args = parser.parse_args()

    root_dir_path = Path(__file__).parent.parent.resolve()

    if args.video:
        video_dir_path = Path(args.video).resolve()
    else:
        video_dir_path = root_dir_path / "data" / "video"
    st.write(f"{video_dir_path=}")

    if args.label:
        label_csv_path = Path(args.label).resolve()
    else:
        label_csv_path = root_dir_path / "data" / "label.csv"

    data_csv_path = root_dir_path / "data" / "csv" / "all.csv"

    if data_csv_path.exists():
        data_df = pd.read_csv(data_csv_path)
    else:
        data_csv_path.parent.mkdir(exist_ok=True, parents=True)
        video_list = sorted(video_dir_path.glob("*.mp4"))
        if len(video_list) == 0:
            st.write(f"There are no video files in {video_dir_path}")
            return
        video_names = [f.name for f in video_list]
        data_df = pd.DataFrame(video_names, columns=["video_name"])
        data_df["label"] = math.nan

    label_df = pd.read_csv(label_csv_path)

    data_df = st.data_editor(data_df, height=200)

    label_options = [f"{r.id}. {r.name}" for r in label_df.itertuples()]

    for row in data_df.itertuples():
        if math.isnan(row.label):
            break

    label = st.selectbox(
        label=f"Select the label of the video: {row.video_name} (index: {row.Index})",
        options=label_options,
        index=None,
        placeholder="Select label",
    )

    video_path = video_dir_path / row.video_name
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes, loop=True, autoplay=True)

    if label:
        id = int(label.split(".")[0])
        data_df.at[row.Index, "label"] = id

    st.button("OK", type="primary")

    data_df.to_csv(data_csv_path, index=False)


if __name__ == "__main__":
    main()
