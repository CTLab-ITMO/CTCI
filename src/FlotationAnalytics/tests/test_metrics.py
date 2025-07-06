import pytest
import numpy as np
from app.main import TrackingQualityAnalyzer
from app.main import DeepSortTracker
from app.main import VideoTracker

@pytest.fixture
def analyzer():
    return TrackingQualityAnalyzer()

def test_initial_state(analyzer):
    assert len(analyzer.metrics['frame']) == 0
    assert analyzer.prev_tracks == {}
    assert analyzer.prev_frame is None

def test_calculate_iou():
    analyzer = TrackingQualityAnalyzer()
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert analyzer.calculate_iou(box1, box2) == 1.0

    box3 = [5, 5, 10, 10]
    assert analyzer.calculate_iou(box1, box3) == 0.14285714285714285
    
    box4 = [20, 20, 30, 30]
    assert analyzer.calculate_iou(box1, box4) == 0.0

def test_update_metrics_simple(analyzer):
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    current_tracks = [
        [1, 10, 10, 20, 20],  # track_id, x, y, w, h
        [2, 30, 30, 10, 10]
    ]
    analyzer.update_metrics(1, current_tracks, frame1)
    
    assert len(analyzer.metrics['displacement']) == 0
    assert len(analyzer.metrics['active_tracks']) == 0
    assert len(analyzer.metrics['track_lengths']) == 2
    
    current_tracks = [
        [1, 12, 12, 20, 20],
        [2, 31, 31, 10, 10]
    ]
    analyzer.update_metrics(2, current_tracks, frame2)
    
    assert len(analyzer.metrics['displacement']) == 1
    assert analyzer.metrics['active_tracks'] == [2]
    assert analyzer.metrics['track_lengths'] == {1: 2, 2: 2}
    assert analyzer.metrics['coverage'][0] == 1.0
    assert analyzer.metrics['temporal_consistency'][0] > 0

def test_tracking_score_calculation(analyzer):
    analyzer.metrics = {
        'displacement': [5.0, 6.0, 4.5],
        'coverage': [0.9, 0.8, 0.85],
        'temporal_consistency': [0.7, 0.75, 0.8],
        'optical_flow': [2.0, 2.5, 3.0],
        'track_lengths': {1: 10, 2: 15, 3: 8},
        'active_tracks': [2, 3, 2],
        'frame': [1, 2, 3]
    }
    
    normalized_score = analyzer.get_tracking_score()
    assert 0 <= normalized_score <= 2.0
    
    raw_score = analyzer.get_tracking_score(normalize=False)
    assert isinstance(raw_score, float)

    custom_weights = {
        'avg_displacement': -0.3,
        'avg_coverage': 0.4,
        'avg_temporal_consistency': 0.3,
        'avg_optical_flow': -0.2,
        'track_length_mean': 0.15,
        'max_active_tracks': 0.05
    }
    custom_score = analyzer.get_tracking_score(weights=custom_weights)
    assert 0 <= custom_score <= 2.0
    assert not np.isclose(custom_score, normalized_score)

def test_empty_metrics(analyzer):
    assert analyzer.get_final_metrics() == {}
    assert analyzer.get_tracking_score() == 0.0

def test_plot_generation(analyzer):
    for i in range(5):
        analyzer.metrics['frame'].append(i)
        analyzer.metrics['displacement'].append(i * 0.5)
        analyzer.metrics['coverage'].append(min(0.9, i * 0.2))
        analyzer.metrics['temporal_consistency'].append(min(0.8, i * 0.15))
        analyzer.metrics['optical_flow'].append(i * 0.3)
        analyzer.metrics['active_tracks'].append(i + 1)
        analyzer.metrics['track_lengths'][i] = i + 5
    
    fig = analyzer.generate_metrics_plots()
    assert fig is not None
    assert len(fig.get_axes()) == 6

def test_optical_flow_calculation(analyzer):
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    frame2[50:70, 50:70] = 255
    
    analyzer.update_metrics(1, [[1, 45, 45, 30, 30]], frame1)
    assert len(analyzer.metrics['optical_flow']) == 0
    
    analyzer.update_metrics(2, [[1, 50, 50, 30, 30]], frame2)
    assert len(analyzer.metrics['optical_flow']) == 1
    assert analyzer.metrics['optical_flow'][0] > 0
    
    analyzer.update_metrics(3, [[1, 55, 55, 30, 30]], frame3)
    assert len(analyzer.metrics['optical_flow']) == 2
    assert analyzer.metrics['optical_flow'][1] >= 0

def test_deepsort_tracker_initialization():
    tracker = DeepSortTracker(img_size=(640, 480))
    assert tracker.img_size == (640, 480)
    assert tracker.nms_max_overlap == 0.6
    assert tracker.iou_threshold == 0.3
    assert tracker.tracker is not None

    tracker = DeepSortTracker(
        img_size=(1280, 720),
        nms_max_overlap=0.5,
        max_cosine_distance=0.4,
        max_age=50,
        min_hits=5,
        iou_threshold=0.2
    )
    assert tracker.img_size == (1280, 720)
    assert tracker.nms_max_overlap == 0.5
    assert tracker.iou_threshold == 0.2

def test_prepare_detections():
    tracker = DeepSortTracker(img_size=(640, 480))
    detections = np.array([
        [10, 10, 50, 50, 0.9, 0],  # x1, y1, x2, y2, conf, class
        [20, 20, 60, 60, 0.8, 1]
    ])
    
    prepared = tracker.prepare_detections(detections)
    assert len(prepared) == 2
    assert prepared[0].tlwh.tolist() == [10, 10, 40, 40]  # x, y, w, h
    assert prepared[0].confidence == 0.9
    assert prepared[1].tlwh.tolist() == [20, 20, 40, 40]
    assert prepared[1].confidence == 0.8

    empty_detections = np.empty((0, 6))
    prepared = tracker.prepare_detections(empty_detections)
    assert len(prepared) == 0

def test_update_with_empty_detections():
    tracker = DeepSortTracker(img_size=(640, 480))
    results = tracker.update([])
    assert len(results) == 0


def test_update_with_multiple_detections():
    tracker = DeepSortTracker(img_size=(640, 480))
    detections = np.array([
        [10, 10, 50, 50, 0.9, 0],
        [20, 20, 60, 60, 0.8, 1],
        [30, 30, 70, 70, 0.7, 2]
    ])
    results = tracker.update(detections)
    assert len(results) <= len(detections)
    for result in results:
        assert len(result) == 5

@pytest.fixture
def tracker():
    return VideoTracker("model/FSC147.pth", "some_tracker_type")

def test_empty_input(tracker):
    points = np.array([])
    scores = np.array([])
    result = tracker.non_max_suppression(points, scores)
    assert len(result) == 0

def test_single_point(tracker):
    points = np.array([[10, 20]])
    scores = np.array([0.9])
    result = tracker.non_max_suppression(points, scores)
    assert np.array_equal(result, points)

def test_no_suppression_needed(tracker):
    points = np.array([[10, 20], [50, 60], [100, 120]])
    scores = np.array([0.9, 0.8, 0.7])
    result = tracker.non_max_suppression(points, scores)
    assert np.array_equal(result, points)

def test_basic_suppression(tracker):
    points = np.array([
        [10, 10],
        [12, 12],
        [50, 50],
        [52, 52]
    ])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    expected = np.array([[10, 10], [50, 50]])
    result = tracker.non_max_suppression(points, scores, radius=5)
    assert np.array_equal(result, expected)

def test_custom_radius(tracker):
    points = np.array([
        [10, 10],
        [15, 15],
        [30, 30]
    ])
    scores = np.array([0.9, 0.8, 0.7])
    result_small_radius = tracker.non_max_suppression(points, scores, radius=2)
    assert len(result_small_radius) == 3
    
    result_large_radius = tracker.non_max_suppression(points, scores, radius=10)
    assert len(result_large_radius) == 2
    assert np.array_equal(result_large_radius, np.array([[10, 10], [30, 30]]))

def test_scores_ordering(tracker):
    points = np.array([
        [10, 10],
        [11, 11],
        [50, 50]
    ])
    scores = np.array([0.8, 0.9, 0.7])
    result = tracker.non_max_suppression(points, scores, radius=5)
    assert np.array_equal(result, np.array([[11, 11], [50, 50]]))

def test_edge_case_radius_zero(tracker):
    points = np.array([
        [10, 10],
        [10, 10],
        [20, 20]
    ])
    scores = np.array([0.9, 0.8, 0.7])
    result = tracker.non_max_suppression(points, scores, radius=0)
    assert len(result) == 3

def test_multiple_close_points(tracker):
    points = np.array([
        [10, 10],
        [11, 11],
        [12, 12],
        [50, 50],
        [51, 51]
    ])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    result = tracker.non_max_suppression(points, scores, radius=5)
    assert np.array_equal(result, np.array([[10, 10], [50, 50]]))