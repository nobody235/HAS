import collections
import json
from typing import Optional, List, Dict, Set

import numpy as np
import argparse
import os
import pickle

from sklearn.cluster import KMeans
from tqdm import tqdm



def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val == min_val:
        return np.full_like(scores, 0.5)
    return (scores - min_val) / (max_val - min_val)


def select_frames_cdpruner(
        embeddings: np.ndarray,
        relevance_scores: np.ndarray,
        frame_nums: List[int],
        num_core_frames: int,  # This now acts as the MAXIMUM number of core frames
        candidate_pool_multiplier: float,
        sigma: float,
        # --- NEW PARAMETERS ---
        min_contribution_threshold: float,
        min_core_frames: int
) -> List[int]:
    """
    [MODIFIED VERSION]
    Selects core frames based on a contribution threshold, not just a fixed number.
    It stops when a new frame's contribution is below a threshold, but ensures a minimum
    number of frames are selected.
    """
    num_total_frames = len(frame_nums)
    # If total frames are less than the minimum required, return all of them.
    if num_total_frames <= min_core_frames:
        return sorted(frame_nums)

    pool_size = int(candidate_pool_multiplier * num_core_frames)
    pool_size = min(pool_size, num_total_frames)
    pool_size = max(pool_size, num_core_frames, 1)

    candidate_pool_indices = np.argsort(-relevance_scores)[:pool_size]

    pool_embeddings = embeddings[candidate_pool_indices]
    pool_relevance_scores = relevance_scores[candidate_pool_indices]
    pool_frame_nums = np.array(frame_nums)[candidate_pool_indices]

    v = pool_embeddings / (np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-8)
    L = v @ v.T

    r_tilde = _normalize_scores(pool_relevance_scores)
    diag_r_tilde = np.diag(r_tilde)
    L_tilde = diag_r_tilde @ L @ diag_r_tilde

    num_pool_frames = len(candidate_pool_indices)
    max_to_select = min(num_core_frames, num_pool_frames)  # The absolute maximum we can select

    selected_indices_in_pool = []
    d_sq = np.diag(L_tilde).copy()
    c_vectors = [np.array([]) for _ in range(num_pool_frames)]

    # --- MODIFIED CORE LOOP ---
    while len(selected_indices_in_pool) < max_to_select:
        # Temporarily set scores of already selected frames to a very low value to ignore them
        if selected_indices_in_pool:
            d_sq[selected_indices_in_pool] = -1.0

        best_pool_idx = np.argmax(d_sq)
        contribution_score = d_sq[best_pool_idx]

        # --- NEW STOPPING CONDITION ---
        if contribution_score <= 0:
            break
        if len(selected_indices_in_pool) >= min_core_frames and contribution_score < min_contribution_threshold:
            break

        selected_indices_in_pool.append(best_pool_idx)
        last_selected_idx = best_pool_idx

        d_last_val = np.sqrt(contribution_score)
        c_last = c_vectors[last_selected_idx]

        for i in range(num_pool_frames):
            if d_sq[i] > -1.0:
                L_ji = L_tilde[last_selected_idx, i]
                c_i = c_vectors[i]
                e_i = (L_ji - np.dot(c_last, c_i)) / (d_last_val + 1e-8)
                c_vectors[i] = np.append(c_i, e_i)
                d_sq[i] -= e_i ** 2

    core_frames = [frame_nums[candidate_pool_indices[i]] for i in selected_indices_in_pool]
    return sorted(core_frames)


def find_context_by_threshold(core_idx: int, embeddings: np.ndarray, selected_frames_set: Set[int],
                              idx_to_frame: Dict[int, int], max_distance: int = 50,
                              sim_threshold: float = 0.8, direction: int = +1) -> Optional[int]:
    """ [MODIFIED] Finds a context frame based on similarity drop, but stops if it encounters an already selected frame. """
    v_core = embeddings[core_idx] / (np.linalg.norm(embeddings[core_idx]) + 1e-8)
    for d in range(1, max_distance + 1):
        idx = core_idx + direction * d
        if not (0 <= idx < embeddings.shape[0]):
            break

        # Check if the intermediate frame is already selected. If so, stop search in this direction.
        intermediate_frame_num = idx_to_frame.get(idx)
        if intermediate_frame_num in selected_frames_set:
            return None

        v_cur = embeddings[idx] / (np.linalg.norm(embeddings[idx]) + 1e-8)
        sim = np.dot(v_core, v_cur)
        if sim < sim_threshold:
            return idx  # Found the boundary
    return None


# ==============================================================================
#  (3) 补充帧采样模块 ([MODIFIED] to handle the new context logic)
# ==============================================================================
def analyze_frame_distribution(core_frames: List[int]) -> float:
    num_core_frames = len(core_frames)
    if num_core_frames < 2: return 0.0
    sorted_frames = sorted(core_frames)
    gaps = np.diff(sorted_frames)
    avg_gap = np.mean(gaps)
    std_dev_of_gaps = np.std(gaps)
    distribution_score = np.exp(- (std_dev_of_gaps / (avg_gap + 1e-8)))
    return distribution_score


def analyze_frame_distribution_se(
        core_frames: List[int],
        all_video_frames: List[int]
) -> float:
    """
    [CORRECTED] Analyzes the uniformity of frame distribution across the entire video timeline,
    including gaps at the start and end.

    Args:
        core_frames: The list of selected core frame numbers.
        all_video_frames: A sorted list of all possible frame numbers in the video.

    Returns:
        A score between 0 and 1, where 1 indicates a perfectly uniform distribution.
    """
    # Edge case: If there are no core frames or no video frames, distribution is undefined or poor.
    if not core_frames or not all_video_frames:
        return 0.0

    # Ensure core frames are sorted and unique for correct gap calculation.
    sorted_core = sorted(list(set(core_frames)))

    # Define the boundaries of the timeline. Assumes all_video_frames is sorted.
    video_start = all_video_frames[0]
    video_end = all_video_frames[-1]

    # Create a list of all points that define the gaps:
    # the video's start, all core frames, and the video's end.
    boundary_points = [video_start] + sorted_core + [video_end]

    # Using set() handles cases where a core frame might be the very first or last frame.
    boundary_points = sorted(list(set(boundary_points)))

    # Calculate the differences between consecutive boundary points to get the gap sizes.
    gaps = np.diff(boundary_points)

    # If there are no gaps (e.g., only one boundary point, or they are all the same),
    # the distribution can be considered perfect or undefined. 1.0 is a reasonable default.
    if gaps.size == 0:
        return 1.0

    # Calculate the average and standard deviation of these gaps.
    avg_gap = np.mean(gaps)

    # If the average gap is zero, it means all points are the same; distribution is perfect.
    if avg_gap < 1e-8:
        return 1.0

    std_dev_of_gaps = np.std(gaps)

    # The score is based on the coefficient of variation (std/mean).
    # We use an exponential function to map it to the range [0, 1].
    # A low std dev relative to the mean -> score close to 1 (good distribution).
    # A high std dev relative to the mean -> score close to 0 (poor distribution).
    distribution_score = np.exp(-(std_dev_of_gaps / avg_gap))

    return distribution_score

def supplement_frames(
        core_frames: List[int],
        video_data: Dict,
        args: argparse.Namespace
) -> List[int]:
    selected_frames = set(core_frames)
    total_budget = args.max_num_frames
    relevance_scores, frame_nums, embeddings = \
        video_data['relevance'], video_data['frames'], video_data['embeddings']
    total_frames_count = len(frame_nums)
    remaining_budget = total_budget - len(selected_frames)
    if remaining_budget <= 0:
        return sorted(list(selected_frames))

    frame_to_idx = {frame: i for i, frame in enumerate(frame_nums)}
    idx_to_frame = {i: frame for i, frame in enumerate(frame_nums)}
    frame_to_score = {frame: relevance_scores[i] for i, frame in enumerate(frame_nums)}

    distribution_score = analyze_frame_distribution(core_frames)
    dynamic_context_ratio = args.min_context_ratio + (
            args.max_context_ratio - args.min_context_ratio) * distribution_score
    context_budget = int(remaining_budget * dynamic_context_ratio)

    # --- [MODIFIED] 自适应上下文选择 (with collision check) ---
    if context_budget > 0 and core_frames:
        selected_context_count = 0

        # --- [A] 确定上下文查找函数 ---
        # [MODIFIED] Prepare common arguments for the find functions.
        find_args_common = {
            'selected_frames_set': selected_frames,
            'idx_to_frame': idx_to_frame
        }

        find_func = find_context_by_threshold
        find_args = {'sim_threshold': args.sim_threshold, **find_args_common}

        # --- [B] 对核心帧进行单轮遍历以寻找上下文 ---
        sorted_core_for_context = sorted(core_frames, key=lambda f: frame_to_score.get(f, 0), reverse=True)

        for core_f in sorted_core_for_context:
            if selected_context_count >= context_budget:
                break

            current_idx = frame_to_idx.get(core_f)
            if current_idx is None: continue

            # 1. 尝试向后扩展 (direction=+1)
            # [MODIFIED] The call now correctly passes all necessary arguments via **find_args
            idx_after = find_func(current_idx, embeddings, max_distance=args.max_context_distance, direction=+1,
                                  **find_args)

            if idx_after is not None:
                fnum_after = idx_to_frame.get(idx_after)
                # The check `fnum_after not in selected_frames` is now redundant due to the internal check, but harmless.
                if fnum_after is not None and fnum_after not in selected_frames:
                    selected_frames.add(fnum_after)
                    selected_context_count += 1
                    if selected_context_count >= context_budget: break

            # 2. 尝试向前扩展 (direction=-1)
            idx_before = find_func(current_idx, embeddings, max_distance=args.max_context_distance, direction=-1,
                                   **find_args)
            if idx_before is not None:
                fnum_before = idx_to_frame.get(idx_before)
                if fnum_before is not None and fnum_before not in selected_frames:
                    selected_frames.add(fnum_before)
                    selected_context_count += 1

    # --- 全局视野补充 (使用所有剩余预算) ---
    if len(selected_frames) < total_budget:
        global_budget = total_budget - len(selected_frames)
        available_frames = set(frame_nums) - selected_frames
        available_candidates = list(available_frames)
        if args.global_strategy == 'enhanced_uniform' and available_frames:
            for _ in range(global_budget):
                if not available_frames: break

                # 1. Identify all current gaps, including video start/end
                # Use the full list of original frame numbers to get the true boundaries.
                all_possible_frames = sorted(frame_nums)
                boundary_points = sorted(
                    list(set([all_possible_frames[0]] + list(selected_frames) + [all_possible_frames[-1]])))

                gaps = []  # Store as (gap_size, start_frame, end_frame)
                for j in range(len(boundary_points) - 1):
                    start, end = boundary_points[j], boundary_points[j + 1]
                    if end > start + 1:  # A gap exists only if there's at least one frame between them
                        gaps.append((end - start, start, end))

                if not gaps: break  # No more gaps to fill

                # 2. Find the largest gap that has available frames
                gaps.sort(key=lambda x: x[0], reverse=True)  # Sort by gap_size, descending

                frame_to_add = None
                for gap_size, start_gap, end_gap in gaps:
                    # Find available frames within this specific gap (exclusive of boundaries)
                    candidates_in_gap = [f for f in available_frames if start_gap < f < end_gap]
                    if not candidates_in_gap:
                        continue  # This gap has no available frames, try the next largest one

                    # 3. Select the frame closest to the middle of the gap
                    midpoint = start_gap + (end_gap - start_gap) / 2.0
                    frame_to_add = min(candidates_in_gap, key=lambda f: abs(f - midpoint))
                    break  # Found a frame, so exit the gap-finding loop

                # 4. Update selections
                if frame_to_add is not None:
                    selected_frames.add(frame_to_add)
                    available_frames.remove(frame_to_add)
                else:
                    # This can happen if all remaining available frames are contiguous with
                    # already selected ones, so no gaps can be filled.
                    break
        elif args.global_strategy == 'gap_fill_relevance' and available_frames:
            # Iteratively add frames until the budget is used up
            for _ in range(global_budget):
                # Stop if there are no more frames available to choose from
                if not available_frames:
                    break

                # 1. Identify boundaries (start, end, selected) and find all gaps
                all_possible_frames = sorted(frame_nums)
                boundary_points = sorted(
                    list(set([all_possible_frames[0]] + list(selected_frames) + [all_possible_frames[-1]])))

                gaps = []  # To store tuples of (gap_size, start_frame, end_frame)
                for j in range(len(boundary_points) - 1):
                    start_gap, end_gap = boundary_points[j], boundary_points[j + 1]
                    gap_size = end_gap - start_gap
                    if gap_size > 1:  # A gap must be able to contain at least one frame
                        gaps.append((gap_size, start_gap, end_gap))

                # If there are no more gaps to fill, stop.
                if not gaps:
                    break

                # 2. Find the largest gap that has available frames inside
                gaps.sort(key=lambda x: x[0], reverse=True)  # Sort by gap_size, descending

                frame_to_add = None
                for gap_size, start_gap, end_gap in gaps:
                    # Find all available frames that fall strictly within the current gap
                    candidates_in_gap = [f for f in available_frames if start_gap < f < end_gap]

                    # If this gap contains candidates, we've found our target gap
                    if candidates_in_gap:
                        # 3. Select the candidate with the maximum relevance score
                        frame_to_add = max(candidates_in_gap, key=lambda f: frame_to_score.get(f, 0.0))

                        # Once we've found a frame, break the loop over gaps
                        break

                # 4. Update selections for the next iteration
                if frame_to_add is not None:
                    selected_frames.add(frame_to_add)
                    available_frames.remove(frame_to_add)
                else:
                    # This can happen if no gaps have any available frames.
                    break
        # (Global strategies remain unchanged, as they operate on the remaining frames)
        elif args.global_strategy == 'uniform' and available_candidates:
            segment_boundaries = np.linspace(0, total_frames_count, global_budget + 1).astype(int)
            for i in range(global_budget):
                if not available_frames: break
                start_idx, end_idx = segment_boundaries[i], segment_boundaries[i + 1]
                segment_frames_set = {idx_to_frame.get(j) for j in range(start_idx, end_idx) if
                                      idx_to_frame.get(j) is not None}
                available_in_segment = list(segment_frames_set.intersection(available_frames))

                if available_in_segment:
                    uniform_pos = start_idx + (end_idx - start_idx) // 2
                    best_in_segment = min(available_in_segment, key=lambda f: abs(f - uniform_pos))
                    selected_frames.add(best_in_segment)
                    available_frames.remove(best_in_segment)

        elif args.global_strategy == 'max_relevance' and available_candidates:
            candidates_sorted_by_relevance = sorted(available_candidates, key=lambda f: frame_to_score.get(f, 0),
                                                    reverse=True)
            frames_to_add = candidates_sorted_by_relevance[:global_budget]
            selected_frames.update(frames_to_add)

        elif args.global_strategy == 'enhanced_cluster' and available_candidates:
            """
            Divides the video timeline into `global_budget` segments. For each segment,
            it finds all available (unselected) frames, clusters their embeddings,
            identifies the largest cluster, and selects the frame closest to that
            cluster's centroid.
            """
            if global_budget > 0:
                segment_boundaries = np.linspace(0, total_frames_count, global_budget + 1).astype(int)

                for i in range(global_budget):
                    start_idx, end_idx = segment_boundaries[i], segment_boundaries[i + 1]

                    # Find available candidate frames (and their indices) within this segment
                    candidates_in_segment_indices = []
                    for idx in range(start_idx, end_idx):
                        frame_num = idx_to_frame.get(idx)
                        if frame_num is not None and frame_num in available_frames:
                            candidates_in_segment_indices.append(idx)

                    if not candidates_in_segment_indices:
                        continue  # No available frames in this segment

                    # If only one candidate, select it directly
                    if len(candidates_in_segment_indices) == 1:
                        frame_to_add = idx_to_frame[candidates_in_segment_indices[0]]
                        selected_frames.add(frame_to_add)
                        available_frames.remove(frame_to_add)
                        continue

                    # Perform clustering
                    segment_embeddings = embeddings[candidates_in_segment_indices]
                    num_candidates = len(candidates_in_segment_indices)
                    # Number of clusters cannot exceed number of samples
                    k = min(args.num_clusters, num_candidates)

                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(segment_embeddings)
                    labels = kmeans.labels_

                    # Find the largest cluster
                    if k > 1:
                        counts = np.bincount(labels)
                        largest_cluster_id = np.argmax(counts)
                    else:  # Only one cluster
                        largest_cluster_id = 0

                    # Get the centroid of the largest cluster
                    target_centroid = kmeans.cluster_centers_[largest_cluster_id]

                    # Find indices of frames belonging to the largest cluster
                    # These are indices relative to `segment_embeddings`
                    indices_in_largest_cluster = np.where(labels == largest_cluster_id)[0]

                    # Get the embeddings of frames in the largest cluster
                    embeddings_in_largest_cluster = segment_embeddings[indices_in_largest_cluster]

                    # Calculate distance to the centroid for each frame in the cluster
                    distances = np.linalg.norm(embeddings_in_largest_cluster - target_centroid, axis=1)

                    # Find the frame closest to the centroid
                    closest_local_idx = np.argmin(distances)

                    # Map back to the original index in `candidates_in_segment_indices`
                    original_candidate_list_idx = indices_in_largest_cluster[closest_local_idx]

                    # Get the final frame index in the full video
                    final_frame_idx = candidates_in_segment_indices[original_candidate_list_idx]

                    # Get the frame number and add it
                    frame_to_add = idx_to_frame[final_frame_idx]
                    selected_frames.add(frame_to_add)
                    available_frames.remove(frame_to_add)
            pass  # Keep original implementation for cluster-based global supplement



    # --- 最终填充 (安全网) ---
    if len(selected_frames) < total_budget:
        available_frames = set(frame_nums) - selected_frames
        remaining_candidates = sorted(list(available_frames), key=lambda f: frame_to_score.get(f, 0), reverse=True)
        needed = total_budget - len(selected_frames)
        selected_frames.update(remaining_candidates[:needed])

    return sorted(list(selected_frames))[:total_budget], distribution_score


# ==============================================================================
#  (后续代码保持不变)
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Select Keyframes using adaptive CDPruner + adaptive or fixed context')
    parser.add_argument('--dataset_name', type=str, default='egoscheme')
    parser.add_argument('--extract_feature_model', type=str, default='clip')
    parser.add_argument('--input_dir', type=str, default='./outscores')
    parser.add_argument('--max_num_frames', type=int, default=16)
    parser.add_argument('--num_core_frames', type=int, default=32,
                        help='The MAXIMUM number of core frames to select.')
    parser.add_argument('--min_core_frames', type=int, default=32,
                        help='The MINIMUM number of core frames to always select.')
    parser.add_argument('--min_contribution_threshold', type=float, default=0.03,
                        help='Stop selecting frames if their marginal contribution score is below this threshold.')
    parser.add_argument('--candidate_pool_multiplier', type=float, default=6.0)
    parser.add_argument('--min_context_ratio', type=float, default=0)
    parser.add_argument('--max_context_ratio', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=1 / 128)
    parser.add_argument('--global_strategy', type=str, default='uniform',
                        choices=['uniform', 'enhanced_cluster', 'max_relevance'])
    parser.add_argument('--num_clusters', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='./selected_frames')

    parser.add_argument('--context_method', type=str, default='method1',
                        choices=['method1', 'method2', 'fixed'],
                        help="Choose context selection method: 'method1'=threshold, 'method2'=accumulated_change, 'fixed'=original fixed distance method")
    parser.add_argument('--sim_threshold', type=float, default=0.9, help='Similarity threshold for method1')

    return parser.parse_args()


def resample_at_higher_rate(original_frames: List[int], target_size: int) -> List[int]:
    if not original_frames: return []
    unique_sorted_frames = sorted(list(set(original_frames)))
    n = len(unique_sorted_frames)
    if n >= target_size: return unique_sorted_frames[:target_size]
    start_frame, end_frame = unique_sorted_frames[0], unique_sorted_frames[-1]
    resampled_float_frames = np.linspace(start_frame, end_frame, target_size)
    resampled_int_frames = np.round(resampled_float_frames).astype(int)
    return resampled_int_frames.tolist()


def main():
    args = parse_arguments()
    core_frame_counts = collections.defaultdict(int)

    if args.num_core_frames > args.max_num_frames:
        raise ValueError("`num_core_frames` cannot be greater than `max_num_frames`.")

    input_base_path = os.path.join(args.input_dir, args.dataset_name, args.extract_feature_model)

    # ========= 新的输入格式：scores.json / frames.json / embedding_files.json =========
    scores_path = os.path.join(input_base_path, 'scores.json')
    frames_path = os.path.join(input_base_path, 'frames.json')
    embedding_index_path = os.path.join(input_base_path, 'embedding_files.json')

    print("Loading index files...")
    try:
        with open(scores_path, 'r') as f:
            all_relevance_scores = json.load(f)   # list[list[float]]
        with open(frames_path, 'r') as f:
            all_frame_nums = json.load(f)         # list[list[int]]
        with open(embedding_index_path, 'r') as f:
            embedding_paths = json.load(f)        # list[str or None]
    except FileNotFoundError as e:
        print(f"Error: Required input file not found - {e}")
        return

    if not (len(all_relevance_scores) == len(all_frame_nums) == len(embedding_paths)):
        print("Error: Length mismatch between scores/frames/embedding_files.")
        print(f"len(scores)={len(all_relevance_scores)}, len(frames)={len(all_frame_nums)}, len(embedding_paths)={len(embedding_paths)}")
        return

    num_videos = len(all_frame_nums)
    print(f"Index files loaded. Num videos = {num_videos}")

    final_results = []
    all_distribution_scores = []
    all_core_frame_counts = []

    print("\nStarting frame selection...")
    for i in tqdm(range(num_videos), desc="Processing videos"):
        relevance_scores = np.array(all_relevance_scores[i], dtype=np.float32)
        frame_nums = all_frame_nums[i]
        emb_path = embedding_paths[i]

        # 处理帧列表为空的情况
        if not frame_nums or emb_path is None or (isinstance(emb_path, str) and not os.path.exists(emb_path)):
            final_results.append([])  # Empty frame selection
            all_distribution_scores.append(-1)
            all_core_frame_counts.append(0)
            continue

        # 每个视频单独读 pkl（新格式）
        try:
            with open(emb_path, 'rb') as f:
                video_emb_list = pickle.load(f)
        except Exception as e:
            print(f"\nWarning: Failed to load embedding for video {i} from {emb_path}: {e}")
            final_results.append([])  # Empty frame selection
            all_distribution_scores.append(-1)
            all_core_frame_counts.append(0)
            continue

        if video_emb_list is None or len(video_emb_list) == 0:
            final_results.append([])  # Empty frame selection
            all_distribution_scores.append(-1)
            all_core_frame_counts.append(0)
            continue

        video_embeddings = np.array(video_emb_list)
        video_embeddings = video_embeddings.astype(np.float32)

        if video_embeddings.ndim > 2:
            video_embeddings = video_embeddings.reshape(video_embeddings.shape[0], -1)
        if video_embeddings.ndim == 1:
            video_embeddings = video_embeddings[None, :]

        if video_embeddings.shape[0] != len(relevance_scores):
            print(f"\nWarning: Mismatch data length for video {i}. "
                  f"len(emb)={video_embeddings.shape[0]}, len(scores)={len(relevance_scores)}. Skipping.")
            final_results.append([])  # Empty frame selection
            all_distribution_scores.append(-1)
            all_core_frame_counts.append(0)
            continue

        # 帧数少于预算，直接全保留
        if len(frame_nums) <= args.max_num_frames:
            all_distribution_scores.append(-1)
            selected_frames = sorted(list(set(frame_nums)))
            final_results.append(selected_frames)
            all_core_frame_counts.append(len(selected_frames))
            continue

        # === 核心帧选择 + 补帧 ===
        core_frames = select_frames_cdpruner(
            video_embeddings,
            relevance_scores,
            frame_nums,
            args.num_core_frames,
            args.candidate_pool_multiplier,
            args.sigma,
            args.min_contribution_threshold,
            args.min_core_frames
        )

        core_frame_counts[len(core_frames)] += 1

        video_data = {
            'relevance': relevance_scores,
            'frames': frame_nums,
            'embeddings': video_embeddings
        }

        if len(core_frames) == args.max_num_frames:
            selected_frames = core_frames
            distribution_score = -1
        else:
            selected_frames, distribution_score = supplement_frames(core_frames, video_data, args)

        final_results.append(selected_frames)
        all_core_frame_counts.append(len(core_frames))
        all_distribution_scores.append(distribution_score)

    # 一些统计输出
    if all_core_frame_counts:
        average_core_frames = sum(all_core_frame_counts) / len(all_core_frame_counts)
        print("-" * 40)
        print(f"{'Average Core Frames':<25} | {average_core_frames:.2f}")

    strategy_name = f"HC/"
    output_path = os.path.join(args.output_dir, args.dataset_name, args.extract_feature_model, strategy_name)
    os.makedirs(output_path, exist_ok=True)
    output_filename = (
        f'frames_total_{args.max_num_frames}_context_{args.context_method}_'
        f'kf_{args.num_core_frames}_pool_{args.candidate_pool_multiplier}_'
        f'sim_{args.sim_threshold}_minthr_{args.min_contribution_threshold}_'
        f'gs_{args.global_strategy}_mcr_{args.max_context_ratio}_min_{args.min_core_frames}.json'
    )
    output_filepath = os.path.join(output_path, output_filename)

    final_results_py = to_python_type(final_results)
    with open(output_filepath, "w") as f:
        json.dump(final_results_py, f, indent=2)
    print("\nDone! Selected frame indices saved to", output_filepath)

    score_filename = f'score_origin_{args.dataset_name}.json'
    score_filepath = os.path.join('./', score_filename)
    with open(score_filepath, 'w') as f:
        scores_with_indices = {i: score for i, score in enumerate(all_distribution_scores)}
        json.dump(scores_with_indices, f, indent=2)

    print(f"Distribution scores saved to", score_filepath)

    print(f"{'Num of Core Frames':<25} | {'Num of Videos'}")
    print("-" * 40)
    for num_frames in sorted(core_frame_counts.keys()):
        count = core_frame_counts[num_frames]
        print(f"{num_frames:<25} | {count}")
    print("=" * 50)


if __name__ == '__main__':
    print("start")
    main()



if __name__ == '__main__':
    main()
