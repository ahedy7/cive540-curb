"""
Blurred example frames for slides, the report and the website (step 5, by hand for a few frames).

EgoBlur is not installed, and only a handful of frames are published, so the boxes are drawn by
hand after looking at each frame: licence plates, people, and company names on vehicles shown
double parking (a slide about violations should not name an operator). YOLO11 also blurs every
person it finds, as a safety net. Check every output by eye before publishing.

Output: figures/frames/<frame_id>.jpg and figures/examples.png (the frames side by side).

Usage:
    python pipeline/blur_examples.py
"""
import cv2
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

from common import DERIVED, ROOT
from detect import MODEL

matplotlib.use("Agg")
OUT = ROOT / "figures" / "frames"
PAD = 6   # px added around each box

# frame_id -> (caption, [(x0, y0, x1, y1), ...] in the frame's own pixels)
EXAMPLES = {
    "20261111_L_2_000631": ("Box truck double parked, De Maisonneuve (left camera, Nov 11)",
                            [(398, 100, 452, 134),      # truck plate
                             (388, 140, 722, 352),      # company name, phone and address, rear
                             (742, 100, 890, 480),      # company name, side
                             (1160, 470, 1205, 492)]),  # car plate
    "20261113_R_1_000505": ("Pickup, trailer and loader in the lane, Sainte-Catherine (right camera, Nov 13)",
                            [(905, 375, 1000, 470),     # company name on the loader
                             (1090, 410, 1145, 445),    # loader sticker
                             (40, 455, 125, 475)]),     # plate area, white van
    "20261111_F_2_000464": ("Box truck stopped in the lane ahead, Crescent (forward camera, Nov 11)",
                            [(1024, 788, 1088, 826),    # SUV plate
                             (180, 795, 215, 818),      # pickup plate
                             (360, 755, 420, 825),      # white SUV plate
                             (455, 778, 500, 806),      # box truck plate (far)
                             (1160, 715, 1210, 790)]),  # pedestrian
}


def blur(img, boxes):
    h, w = img.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0 - PAD), max(0, y0 - PAD), min(w, x1 + PAD), min(h, y1 + PAD)
        roi = img[y0:y1, x0:x1]
        k = max(31, (min(x1 - x0, y1 - y0) // 2) * 2 + 1)     # odd kernel, scaled to the box
        img[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.GaussianBlur(roi, (k, k), 0), (k, k), 0)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True).set_index("frame_id").path
    yolo = YOLO(str(MODEL))   # same weights as the pre-filter
    done = []
    for fid, (caption, boxes) in EXAMPLES.items():
        img = cv2.imread(str(ROOT / paths[fid]))
        people = [tuple(map(int, b)) for b in yolo(img, classes=[0], conf=0.25, verbose=False)[0].boxes.xyxy.tolist()]
        img = blur(img, boxes + people)
        cv2.imwrite(str(OUT / f"{fid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        done.append((fid, caption, img))
        print(f"-> figures/frames/{fid}.jpg  ({len(boxes)} hand boxes, {len(people)} people found)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), facecolor="#fcfcfb")
    for ax, (fid, caption, img) in zip(axes, done):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="equal")
        ax.set_title(caption, fontsize=9.5, color="#0b0b0b", loc="left", wrap=True)
        ax.axis("off")
    fig.text(0.01, 0.02, "Plates, people and company names blurred.", fontsize=8.5, color="#8a8984")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=0.03)
    fig.savefig(ROOT / "figures" / "examples.png", dpi=160)
    print("-> figures/examples.png")


if __name__ == "__main__":
    main()
