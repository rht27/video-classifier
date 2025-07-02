import sys
from pathlib import Path
from typing import Any, OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    import cv2
else:
    from torchcodec.decoders import VideoDecoder

from model import Model


class VideoDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, dataset_root_path: Path, is_train: bool = True
    ):
        self.is_train = is_train
        self.dataset_root_path = dataset_root_path
        self.df = df.reset_index(drop=True)
        self.index_to_data = self.df.to_dict(orient="index")

        size = (224, 224)
        self.transformer = v2.Compose(
            [
                # v2.Resize(size),
                v2.ToDtype(torch.float32, scale=True)
            ]
        )

        # ROI
        self.roi_list = [
            (330, 600, 10, 80),  # roi1
            (1115, 1255, 10, 150),  # roi2
            (340, 940, 100, 700),  # roi3
        ]

    def __getitem__(self, index) -> Any:
        data = self.index_to_data[index]
        video_name, label = data.get("video_name"), data.get("label_id")

        label = torch.tensor(label - 1, dtype=torch.int64)

        video_path = self.dataset_root_path / video_name
        assert video_path.exists(), f"{video_path} does not exist"

        frames = self._get_video_frames(video_path)

        return frames, label

    def __len__(self):
        return len(self.df)

    def _get_video_frames(self, video_path: Path):
        raise NotImplementedError()


class CVVideoDataset(VideoDataset):
    def _get_video_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        extract_frames = np.linspace(0, frame_count, 10, endpoint=False, dtype=int)

        x0_1, x1_1, y0_1, y1_1 = self.roi_list[0]
        x0_2, x1_2, y0_2, y1_2 = self.roi_list[1]
        x0_3, x1_3, y0_3, y1_3 = self.roi_list[2]

        rois = {
            "roi1": [],
            "roi2": [],
            "roi3": [],
        }
        for i, extract_frame in enumerate(extract_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, extract_frame)
            ret, frame = cap.read()

            roi1 = frame[y0_1:y1_1, x0_1:x1_1]
            roi2 = frame[y0_2:y1_2, x0_2:x1_2]
            roi3 = frame[y0_3:y1_3, x0_3:x1_3]

            # debug
            # cv2.imwrite("a.png", frame)
            # cv2.imwrite("a.png", roi1)
            # cv2.imwrite("a.png", roi2)
            # cv2.imwrite("a.png", roi3)

            frame = frame.transpose(2, 0, 1)
            frame = torch.from_numpy(frame)
            frame = self.transformer(frame)

            roi1 = frame[:, y0_1:y1_1, x0_1:x1_1]
            roi2 = frame[:, y0_2:y1_2, x0_2:x1_2]
            roi3 = frame[:, y0_3:y1_3, x0_3:x1_3]

            # debug
            # roi1_d = roi1.to("cpu").detach().numpy().copy().transpose(1, 2, 0)
            # plt.imshow(roi1_d)
            # plt.show()
            # plt.close()

            rois["roi1"].append(roi1)
            rois["roi2"].append(roi2)
            rois["roi3"].append(roi3)

        rois["roi1"] = torch.stack(rois["roi1"])
        rois["roi2"] = torch.stack(rois["roi2"])
        rois["roi3"] = torch.stack(rois["roi3"])

        return rois


class TorchVideoDataset(VideoDataset):
    def _get_video_frames(self, video_path: Path):
        decoder = VideoDecoder(video_path)

        frame_count = len(decoder)

        extract_frames = np.linspace(0, frame_count, 10, endpoint=False, dtype=int)

        x0_1, x1_1, y0_1, y1_1 = self.roi_list[0]
        x0_2, x1_2, y0_2, y1_2 = self.roi_list[1]
        x0_3, x1_3, y0_3, y1_3 = self.roi_list[2]

        rois = {
            "roi1": [],
            "roi2": [],
            "roi3": [],
        }

        frames = decoder.get_frames_at(indices=extract_frames).data  # [N, C, H, W]
        frames = self.transformer(frames)

        rois["roi1"] = frames[:, :, y0_1:y1_1, x0_1:x1_1]
        rois["roi2"] = frames[:, :, y0_2:y1_2, x0_2:x1_2]
        rois["roi3"] = frames[:, :, y0_3:y1_3, x0_3:x1_3]

        return rois


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    with tqdm(dataloader) as train_pbar:
        for batch, (data, label) in enumerate(train_pbar):
            # data = data.to(device)
            # label = label.to(device)

            pred = model(data)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_pbar.set_postfix(
                OrderedDict(
                    Loss=loss.item(),
                )
            )


def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for data, label in dataloader:
            pred = model(data)
            valid_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    valid_loss /= num_batches
    correct /= size

    print(
        f"Valid Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n"
    )


def main():
    dataset_root_path = Path(__file__).parent / "data" / "video"
    dataset_csv_path = Path(__file__).parent / "data" / "csv" / "all.csv"

    seed = 42

    dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df = dataset_df.dropna().head(400).sample(100, random_state=seed)

    train_df, valid_df = train_test_split(dataset_df, test_size=0.2, random_state=seed)

    batch_size = 2
    epochs = 10

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    if sys.platform == "win32":
        train_dataset = CVVideoDataset(train_df, dataset_root_path)
        valid_dataset = CVVideoDataset(valid_df, dataset_root_path)
    else:
        train_dataset = TorchVideoDataset(train_df, dataset_root_path)
        valid_dataset = TorchVideoDataset(valid_df, dataset_root_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = Model(num_classes=146)
    model = model.to(device)
    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        valid(valid_dataloader, model, loss_fn)

    # save model
    save_model_path = Path(__file__).parent / "model.pth"
    torch.save(model.state_dict(), save_model_path)


if __name__ == "__main__":
    main()
