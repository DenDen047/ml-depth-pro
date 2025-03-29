import argparse
import os
import sys

import cv2
import numpy as np
import rerun as rr
from loguru import logger
from tqdm import tqdm


def unproject_depth_to_3d(depth_map, f, cx, cy):
    """単眼深度推定モデルが想定する透視投影カメラパラメータ (f, cx, cy) を用いて
    深度マップから3次元点群 (X, Y, Z) を復元する関数.

    Parameters
    ----------
        depth_map: (H, W) のnumpy配列.
        f: モデルが想定する焦点距離.
        cx, cy: モデルが想定する主点の画素座標.

    Returns
    -------
        points_3d: (H, W, 3) の3次元点群.

    """
    H, W = depth_map.shape
    u = np.arange(W)
    v = np.arange(H)
    u_grid, v_grid = np.meshgrid(u, v)
    Z = depth_map
    X = (u_grid - cx) * Z / f
    Y = (v_grid - cy) * Z / f
    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d


def adjust_depth_for_narrow_fov(points_3d, assumed_focal_length, cx, cy):
    """多くの単眼深度推定モデルがFoVが広いPerspective camera modelを想定しているため、
    推定された深度をFoVが狭いもの（Zoomed image or Orthographic camera modelに近いもの）に補正する関数。

    Parameters
    ----------
        points_3d: (H, W, 3) または (N, 3) のnumpy配列.
        assumed_focal_length: 正射投影変換に使用する仮想的な焦点距離.
        cx: 画像の中心X座標.
        cy: 画像の中心Y座標.

    Returns
    -------
        points_3d_zoom: 補正後の3次元点群.

    """
    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2]

    H, W = X.shape
    u = np.arange(W)
    v = np.arange(H)
    u_grid, v_grid = np.meshgrid(u, v)

    d = np.sqrt((u_grid - cx) ** 2 + (v_grid - cy) ** 2)
    Z_zoom = Z * assumed_focal_length / np.sqrt(d**2 + assumed_focal_length**2)

    points_3d_zoom = np.stack((X, Y, Z_zoom), axis=-1)
    return points_3d_zoom


def visualize_rgb_depth(rgb_image, depth_map, focal_length=None, timestamp=None):
    """透視投影モデルでのRGB画像と深度マップの可視化（従来の処理）."""
    if timestamp is not None:
        rr.set_time_seconds("frame_time", timestamp)

    h, w = depth_map.shape
    focal_length = 0.7 * w if focal_length is None else focal_length
    rr.log(
        "world/camera/image",
        rr.Pinhole(
            resolution=[w, h],
            focal_length=focal_length,
        ),
    )
    rr.log("world/camera/image/rgb", rr.Image(rgb_image).compress(jpeg_quality=95))
    DEPTH_IMAGE_SCALING = 1.0
    rr.log("world/camera/image/depth", rr.DepthImage(depth_map, meter=DEPTH_IMAGE_SCALING))
    rr.reset_time()


def visualize_zoomed_depth(rgb_image, depth_map, f, cx, cy, timestamp=None):
    """Zoomed image or Orthographic camera modelでのRGB画像と深度マップから生成した3D点群の可視化.

    手順:
      1. モデル想定の透視投影カメラパラメータ (f, cx, cy) を利用して深度マップから
         3次元点群（透視）の復元.
      2. それをZoomed image or Orthographic camera model用に変換.
      3. Rerunを用いて点群を記録・可視化.

    Parameters
    ----------
        rgb_image: RGB画像 (H, W, 3)
        depth_map: 深度マップ (H, W)
        f: 焦点距離 (モデル想定)
        cx, cy: 主点座標 (モデル想定)
        timestamp: タイムライン用タイムスタンプ

    """
    if timestamp is not None:
        rr.set_time_seconds("frame_time", timestamp)
    assert f is not None, "focal_length is required"
    assert cx is not None, "cx is required"
    assert cy is not None, "cy is required"

    # 1. 透視投影カメラパラメータに基づく逆投影
    points_3d_persp = unproject_depth_to_3d(depth_map, f, cx, cy)

    # 2. Narrow FOVに補正
    points_3d_zoom = adjust_depth_for_narrow_fov(
        points_3d_persp, assumed_focal_length=3000, cx=cx, cy=cy
    )

    # 3. Filter points that are too far away
    flatten_points = points_3d_zoom.reshape(-1, 3)
    flatten_rgb = rgb_image.reshape(-1, 3)
    # mask = (10.0 < flatten_points[:, 2]) & (flatten_points[:, 2] < 20.0)
    # flatten_points = flatten_points[mask, :]
    # flatten_rgb = flatten_rgb[mask, :]

    # 4. Rerunを用いて点群を記録（ここでは仮に "world/points3d_ortho" に記録）
    rr.log(
        "world/points3d_zoom",
        rr.Points3D(
            flatten_points,
            colors=flatten_rgb,
        ),
    )
    rr.log(
        "debug_world/points3d_persp",
        rr.Points3D(points_3d_persp.reshape(-1, 3), colors=flatten_rgb),
    )
    # 参照用にRGB画像も記録
    rr.log("world/camera/image/rgb", rr.Image(rgb_image).compress(jpeg_quality=95))
    rr.log("world/camera/image/depth", rr.DepthImage(depth_map))
    rr.reset_time()


def main():
    parser = argparse.ArgumentParser(
        description="Rerunを用いて、深度マップから正射投影3D点群を生成・可視化するツール"
    )
    parser.add_argument(
        "--depth_file",
        type=str,
        required=True,
        help="run_video_depth_anythingで生成された深度マップ(NPYまたはNPZ)",
    )
    parser.add_argument(
        "--image_or_video", type=str, required=True, help="元の画像または動画ファイルのパス"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/rerun_visualize", help="可視化出力ディレクトリ"
    )
    parser.add_argument(
        "--invert_depth", action="store_true", help="深度マップが反転している場合に設定"
    )
    parser.add_argument(
        "--depth_scaling", type=float, default=1.0, help="深度マップのスケーリング係数"
    )
    parser.add_argument("--depth_shift", type=float, default=0.0, help="深度マップのシフト係数")
    parser.add_argument("--f", type=float, default=None, help="モデルが想定する焦点距離")
    parser.add_argument("--cx", type=float, default=None, help="モデルが想定する主点のx座標")
    parser.add_argument("--cy", type=float, default=None, help="モデルが想定する主点のy座標")
    parser.add_argument(
        "--assumed_focal_length",
        type=float,
        default=5000,
        help="The focal length the mono-depth model might assume.",
    )
    parser.add_argument(
        "--narrow_fov", action="store_true", help="Narrow FOVに対応して深度を補正する"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="インタラクティブモードでRerunを起動"
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    # Logger設定
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f"{args.output_dir}/log.txt", level="DEBUG", rotation="10 MB")
    os.makedirs(args.output_dir, exist_ok=True)

    recording_id = os.path.basename(args.depth_file).split(".")[0]
    rr.init(recording_id, spawn=args.interactive)

    # 深度マップのロード
    logger.info(f"Loading depth maps from {args.depth_file}")
    if args.depth_file.endswith(".npz"):
        with np.load(args.depth_file) as data:
            depth_maps = data["depth"]
    elif args.depth_file.endswith(".npy"):
        with open(args.depth_file, "rb") as f:
            depth_maps = np.load(f)
    else:
        logger.error("Unsupported depth file format. Use .npz or .npy")
        sys.exit(1)

    if args.invert_depth:
        depth_maps = 1.0 / depth_maps
    depth_maps = depth_maps * args.depth_scaling + args.depth_shift
    logger.info(f"Loaded depth maps with shape: {depth_maps.shape}")

    # 動画モード vs. 画像モードの判定
    if args.image_or_video.endswith(".mp4"):
        cap = cv2.VideoCapture(args.image_or_video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for i in tqdm(range(n_frames), desc="Visualizing video"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {i}")
                break
            rgb_image = frame
            depth_map = depth_maps[i]
            if args.narrow_fov:
                visualize_zoomed_depth(
                    rgb_image, depth_map, args.f, args.cx, args.cy, timestamp=i / fps
                )
            else:
                visualize_rgb_depth(rgb_image, depth_map, focal_length=args.f, timestamp=i / fps)
        cap.release()
    else:
        rgb_image = cv2.imread(args.image_or_video)
        if args.narrow_fov:
            visualize_zoomed_depth(rgb_image, depth_maps, args.f, args.cx, args.cy, timestamp=0)
        else:
            visualize_rgb_depth(rgb_image, depth_maps, focal_length=args.f, timestamp=0)

    if not args.interactive:
        recording_path = os.path.join(args.output_dir, "rerun_recording.rrd")
        logger.info(f"Saving Rerun recording to {recording_path}")
        rr.save(recording_path)
        logger.info("Visualization complete")
    else:
        logger.info("Interactive visualization running. Press Ctrl+C to exit.")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting interactive visualization")
    logger.info(f"All output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
