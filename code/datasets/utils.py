import cv2
import numpy as np
import networkx as nx

from typing import List
from torchvision.transforms import Resize, ToPILImage


def is_monotonically_increasing(frame_ids):
    for i in range(len(frame_ids) - 1):
        if frame_ids[i] >= frame_ids[i + 1]:
            return False
    return True


def load_frames(video_path, frame_ids):
    cap = cv2.VideoCapture(video_path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not is_monotonically_increasing(frame_ids):
        return None, Exception("frame_ids must be monotonically increasing")

    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ids[0])
    if not ok:
        cap.release()
        raise Exception("Cannot read frame")
    
    next_frame_id = frame_ids[0]
    frames = []
    cap_frame_ids = []
    for frame_id in frame_ids:
        if frame_id > num_frames:
            print(f"frame {frame_id} is out of range")
            break
        while frame_id != next_frame_id:
            cap.grab()
            next_frame_id += 1
            continue
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return Exception("Cannot read frame")
        next_frame_id += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        cap_frame_ids.append(frame_id)

    cap.release()
    return frames, cap_frame_ids


def batcher(iterable, n):
    """
    A function that accepts a list and returns a list of batches. The last batch
    may contain less than n elements.
    """
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]
    # if there are any elements left, yield them
    if iterable[i + 1 :]:
        yield iterable[i + 1 :]


def get_height_and_width(batch):
    if batch.ndim == 5:
        _, _, h, w, _ = batch.shape
    else:
        _, h, w, _ = batch.shape

    return h, w


def normalize_ids_for_batch(batch_ids, ids):
    join = zip(batch_ids, [x for x in range(len(batch_ids))])
    d = dict(join)
    return [d[i] for i in ids]


def preprocess_images(img, shape=[128, 128]):
    """
    Preprocess the images. Transforms them to PIL and resizes them
    Parameters
    ----------
    img : str the input image in numpy array format (h,w,c) shape : list the
        resulting shape (default is [128,128])
    """

    img = ToPILImage("RGB")(img)
    img = Resize(shape)(img)
    return img


def has_consecutive_frames(comp, fps, threshold=3):
    if len(comp) < threshold:
        return False

    # size of the list
    size = len(comp)

    # sort array comp
    comp = sorted(comp)

    # count the maximum consecutive pairs of frames
    max_consecutive = 0
    currect_consecutive = 0
    for i in range(size - 1):
        if (comp[i] + fps) == comp[i + 1]:
            currect_consecutive += 1
        else:
            currect_consecutive = 0
        max_consecutive = max(max_consecutive, currect_consecutive)

    # +1 since max_consecutive counts pairs of consecutive frames
    return max_consecutive + 1 >= threshold


def seconds_to_frames(seconds, fps):
    return int(seconds * fps)


def filter_components(frame_ids, mapping, components, fps, segment=None):
    if segment is None:
        return components

    selected_frames = frame_ids[
        (frame_ids > seconds_to_frames(segment.start_time, fps))
        & (frame_ids < seconds_to_frames(segment.end_time, fps))
    ]

    print(f"Selected frames for segment {segment}: {selected_frames}")

    selected_face_ids = []
    for frame_id in selected_frames:
        selected_face_ids.extend(mapping[frame_id])
    # filter componetns with selected face ids
    filtered_components = list(
        map(
            list,
            [filter(lambda x: x in selected_face_ids, comp) for comp in components],
        )
    )

    # log filtered components
    print(f"Filtered components for segment {segment}: {filtered_components}")

    return filtered_components


def resize_image_to_max_dim(image, max_dim, method="skimage"):
    height, width = image.shape[:2]

    # if both the width and height are None, then return the
    # original image

    if max(height, width) <= max_dim:
        return image

    if height > width:
        ratio = width / height
        scale = height / max_dim
        height = max_dim
        width = int(height * ratio)

    else:
        ratio = height / width
        scale = width / max_dim
        width = max_dim
        height = int(width * ratio)

    # resize the image
    if method == "skimage":
        import skimage.transform

        resized = skimage.transform.resize(
            image, (height, width), preserve_range=True
        ).astype(image.dtype)
    elif method == "cv2":
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # return the resized image and the scale
    return resized, scale


def resize_images_if_larger_than(images, max_dim):
    resized_batch = []
    _, height, width, _ = images.shape
    if max(height, width) <= max_dim:
        return images, None
    for image in images:
        image, scale = resize_image_to_max_dim(image, max_dim)
        resized_batch.append(resize_image_to_max_dim(image, max_dim))
    return np.stack(resized_batch), scale


def _generate_connected_components(similarities, similarity_threshold=0.80):
    # create graph
    graph = nx.Graph()
    for i in range(len(similarities)):
        for j in range(len(similarities)):
            if i != j and similarities[i, j] > similarity_threshold:
                graph.add_edge(i, j)

    components_list = []
    # for component in nx.strongly_connected_components(graph):
    for component in nx.connected_components(graph):
        components_list.append(list(component))
    graph.clear()
    graph = None

    return components_list


def scale_bbox(bbox, height, width, scale_factor):
    left, top, right, bottom = bbox
    size_bb = int(max(right - left, bottom - top) * scale_factor)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    # Check for out of bounds, x-y top left corner
    left = max(int(center_x - size_bb // 2), 0)
    top = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - left, size_bb)
    size_bb = min(height - top, size_bb)
    return left, top, left + size_bb, top + size_bb


def apply_bbox(image, bbox, scale_factor=None):
    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)
    bbox[bbox < 0] = 0
    bbox = np.around(bbox).astype(int)
    if scale_factor:
        bbox = scale_bbox(bbox, image.shape[0], image.shape[1], scale_factor)
    left, top, right, bottom = bbox
    face = image[top:bottom, left:right, :]
    return face


def apply_bboxes(frames, bboxes, scale=None) -> List[np.ndarray]:
    per_image_faces = []
    for i, frame_bboxes in enumerate(bboxes):
        faces = []
        if frame_bboxes is not None:
            for bbox in frame_bboxes:
                face = apply_bbox(frames[i], bbox, scale_factor=scale)
                faces.append(face)
        per_image_faces.append(faces)
    return per_image_faces


def fixed_image_standardization(image_tensor):
    return (image_tensor - 127.5) / 128.0
