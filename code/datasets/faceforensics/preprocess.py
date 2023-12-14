import os
import torch
import numpy as np
import pickle
import logging
import argparse


from datetime import datetime
from PIL import Image
from decord import VideoReader
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from pathlib import Path


from ..utils import (
    _generate_connected_components,
    apply_bboxes,
    fixed_image_standardization,
    has_consecutive_frames,
    load_frames,
    preprocess_images,
    resize_images_if_larger_than,
)

def main(args):
    src_path = Path(args.src_path)
    dst_path = Path(__file__).parent
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = dst_path / f"preprocessing_{current_datetime}.log"
    logging.basicConfig(filename=log_filename.as_posix(), encoding="utf-8", level=logging.DEBUG)

    face_detector = MTCNN(
        keep_all=True,
        post_process=False,
        device=args.device,
        select_largest=False,
        min_face_size=args.min_face_size,
        selection_method="probability",
        thresholds=[0.65, 0.75, 0.95],
        margin=0,
    )
    face_recognition = InceptionResnetV1(pretrained="vggface2").to(args.device).eval()

    # load all videos and filter with quality
    videos = [path for path in src_path.rglob("*.mp4")]
    if args.filter:
        videos = list(filter(lambda v: v.parents[1].stem in args.quality, videos))
    if args.max_videos:
        videos = videos[: args.max_videos]

    metadata = {}

    for vid in tqdm(videos):
        target_video_path = dst_path / vid.relative_to(src_path)
        video_path = vid.as_posix()
        logging.info(f"Video path: {video_path}")
        logging.info(f"Target video path: {target_video_path.as_posix()}")

        # sample video at 1 fps
        try:
            video_reader = VideoReader(video_path)
        except Exception as e:
            logging.error(f"Could not read video {video_path}: {e}")
            continue
        fps = int(video_reader.get_avg_fps())
        frame_ids = np.arange(0, len(video_reader) - fps, fps)
        if len(frame_ids) == 0:
            err = f"Video file does not contain frames: {video_path}"
            logging.warn(err)
            continue
        frames, _cap_frame_ids = load_frames(video_path, frame_ids)
        if not frames:
            logging.warn(f"Could not load frames from {video_path}")
            continue
        frames, _scale = resize_images_if_larger_than(np.stack(frames), 1920)
        
        boxes, _, lmks = zip(*[face_detector.detect(frame, landmarks=True) for frame in frames])
        boxes, lmks = [box if box is not None else [] for box in boxes], [lmk if lmk is not None else [] for lmk in lmks]

        # sanity check
        for i, (box, lmk) in enumerate(zip(boxes, lmks)):
            if len(box) != len(lmk):
                logging.warn(i, box.shape[0], lmk.shape[0])

        # save metadata
        metadata[vid.relative_to(src_path).as_posix()] = {
            "frame_ids": frame_ids,
            "boxes": boxes,
            "lmks": lmks,
        }
        if args.only_metadata:
            continue

        # apply bounding boxes to frames
        faces_no_margin = apply_bboxes(frames, boxes, scale=1)
        faces = apply_bboxes(frames, boxes, scale=args.face_margin)
        # sanity check
        if len(faces) != len(frames):
            logging.warn(
                f"Length of face and frame lists does not match: {len(faces)} vs {len(frames)}"
            )

        # flatten and convert to torch arrays
        numpy_faces_no_margin_flat =  [face for frame_faces in faces_no_margin for face in frame_faces]
        torch_faces_no_margin_flat = [
            torch.from_numpy(face).permute(2, 0, 1)
            for face in numpy_faces_no_margin_flat
        ]
        numpy_faces_flat = [face for frame_faces in faces for face in frame_faces]

        if len(torch_faces_no_margin_flat) == 0:
            logging.warn(f"No faces found in {video_path}")
            logging.info("Continuing to next video...")
            continue

        torch_faces_no_margin_flat = [
            preprocess_images(face) for face in torch_faces_no_margin_flat
        ]
        torch_faces_no_margin_flat = np.stack(
            [np.uint8(face) for face in torch_faces_no_margin_flat]
        )
        torch_faces_no_margin_flat = torch.as_tensor(torch_faces_no_margin_flat.copy())
        torch_faces_no_margin_flat = torch_faces_no_margin_flat.permute(
            0, 3, 1, 2
        ).float()
        torch_faces_no_margin_flat = fixed_image_standardization(
            torch_faces_no_margin_flat
        )

        # create mapping from frame_id to face_id
        face_id = 0
        mapping = {}
        for frame_id, frame_faces in zip(frame_ids, faces):
            face_list = []
            for face in frame_faces:
                face_list.append(face_id)
                face_id += 1
            mapping[frame_id] = face_list

        embeddings = []
        for batch in torch_faces_no_margin_flat.split(args.batch_size, dim=0):
            batch = batch.to(args.device)
            batch_embeddings = face_recognition(batch).detach().cpu().numpy()
            embeddings.append(batch_embeddings)
        embeddings = np.concatenate(embeddings, axis=0)

        similarities = np.dot(np.array(embeddings), np.array(embeddings).T)
        components = _generate_connected_components(
            similarities, similarity_threshold=args.similarity_threshold
        )

        # check if components are valid
        if args.cluster_validation == "cons_frames":
            inv_mapping = {
                v: k for k, values in mapping.items() for v in values if len(values) > 0
            }

            def convert_to_frame_ids(comp):
                return [inv_mapping[face_id] for face_id in comp]

            frame_ids_components = list(map(convert_to_frame_ids, components))
            filter_mask = [
                has_consecutive_frames(comp, fps, threshold=args.num_seconds)
                for comp in frame_ids_components
            ]
            valid_components = np.array(components)[filter_mask].tolist()

        elif args.cluster_validation == "ratio":
            valid_cluster_size = int(len(frame_ids) * args.valid_cluster_size_ratio)
            valid_components = [c for c in components if len(c) > valid_cluster_size]
            logging.info(f"VALID CLUSTER SIZE: {valid_cluster_size}")

        elif args.cluster_validation == "ratio_faces":
            valid_cluster_size = int(
                len(numpy_faces_flat) * args.valid_cluster_size_ratio
            )
            valid_components = [c for c in components if len(c) > valid_cluster_size]
            logging.info(f"VALID CLUSTER SIZE: {valid_cluster_size}")

        else:
            raise ValueError("Cluster validation not supported")

        logging.info(f"valid_components: {list(map(len, valid_components))}")
        logging.info(f"valid_components: {valid_components}")

        if len(valid_components) == 0:
            logging.info(f"No valid components found in {video_path}")
            continue

        # save valid components to disk
        os.makedirs(target_video_path, exist_ok=True)
        for i, comp_face_ids in enumerate(valid_components):
            for j, comp_face_id in enumerate(comp_face_ids):
                img_path = os.path.join(
                    target_video_path, str(i).zfill(3) + "_" + str(j).zfill(4) + ".png"
                )
                img = Image.fromarray(np.uint8(numpy_faces_flat[comp_face_id]))
                img.save(img_path)

    
    # save metadata
    with open(dst_path / "metadata", "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # processing settings
    parser.add_argument("-dev", "--device", default="cuda:0", required=False, type=str)
    parser.add_argument("-b", "--batch-size", default=64, required=False, type=int)
    parser.add_argument(
        "--face_margin", "-fm", type=float, default=1.3, help="Face margin"
    )
    parser.add_argument(
        "--similarity_threshold",
        "-st",
        type=float,
        default=0.65,
        help="Similarity threshold",
    )
    parser.add_argument(
        "--min_face_size", "-mfs", type=int, default=100, help="Min face size"
    )
    parser.add_argument(
        "--cluster_validation",
        "-cv",
        type=str,
        default="ratio_faces",
        help="Cluster validation method [cons_frames, ratio]",
    )
    parser.add_argument(
        "--num_seconds",
        "-ns",
        type=int,
        default=3,
        help="Number of seconds for cluster validation in method 'cons_frames'",
    )
    parser.add_argument(
        "--valid_cluster_size_ratio",
        "-vcsr",
        type=float,
        default=0.1,
        help="Valid cluster size ratio for cluster validation in method 'ratio'",
    )

    ## FF settings
    parser.add_argument(
        "-srcp",
        "--src_path",
        required=True,
        default=None,
        type=str,
        help="Path to videos.",
    )

    parser.add_argument(
        "-q",
        "--quality",
        default=["c0", "c23", "c40"],
        choices=["c0", "c23", "c40"],
        nargs="+",
        help="Specifies video quaility, if None all qualities are used",
    )

    parser.add_argument(
        "-flt",
        "--filter",
        default=False,
        action="store_true",
        help="Filter videos by quality (only for FF).",
    )

    parser.add_argument(
        "-maxv",
        "--max_videos",
        type=int,
        default=None,
        help="Max number of videos to process",
    )

    parser.add_argument(
        "-onlymd",
        "--only_metadata",
        action="store_true",
        help="Only generate metadata",
        default=False,
    )

    # parse arguments
    args = parser.parse_args()
    main(args)