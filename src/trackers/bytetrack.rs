use numpy::ndarray::prelude::*;
use numpy::pyo3::exceptions::PyValueError;
use numpy::pyo3::prelude::*;
use numpy::IntoPyArray;
use numpy::PyArray2;
use numpy::PyReadonlyArray2;

use crate::trackers::BaseTracker;
use crate::trackers::Sort;

/// Create a new ByteTrack bbox tracker
///
/// Parameters
/// ----------
/// max_age
///     maximum frames a tracklet is kept alive without matching detections
/// min_hits
///     minimum number of successive detections before a tracklet is set to alive
/// iou_threshold
///     minimum IOU to assign detection to tracklet
/// init_tracker_min_score
///     minimum score to create a new tracklet from unmatched detection box
/// high_score_threshold
///     boxes with higher scores than this will be used in the first round of association
/// low_score_threshold
///     boxes with score between low_score_threshold and high_score_threshold
///     will be used in the second round of association
/// measurement_noise
///     Diagonal of the measurement noise covariance matrix
///     i.e. uncertainties of (x, y, s, r) measurements
///     defaults should be reasonable in most cases
/// process_noise
///     Diagonal of the process noise covariance matrix
///     i.e. uncertainties of (x, y, s, r, dx, dy, ds) during each step
///     defaults should be reasonable in most cases
#[pyclass(
    extends=BaseTracker,
    text_signature = "(max_age=1, min_hits=3, iou_threshold=0.3, init_tracker_min_score=0.8, high_score_threshold=0.7, low_score_threshold=0.1, measurement_noise=[1., 1., 10., 10.], process_noise=[1., 1., 1., 1., 0.01, 0.01, 0.0001])"
)]
pub struct ByteTrack {
    #[pyo3(get, set)]
    pub high_score_threshold: f32,
    #[pyo3(get, set)]
    pub low_score_threshold: f32,

    sort_tracker: Sort,
}

impl ByteTrack {
    fn split_detections(&self, detection_boxes: CowArray<f32, Ix2>) -> (Array2<f32>, Array2<f32>) {
        let mut high_score_data = Vec::new();
        let mut low_score_data = Vec::new();

        for box_row in detection_boxes.outer_iter() {
            let score = box_row[4];
            if score < self.low_score_threshold {
                continue;
            };
            if score > self.high_score_threshold {
                high_score_data.extend(box_row);
            } else {
                low_score_data.extend(box_row);
            }
        }
        (
            Array2::from_shape_vec((high_score_data.len() / 5, 5), high_score_data).unwrap(),
            Array2::from_shape_vec((low_score_data.len() / 5, 5), low_score_data).unwrap(),
        )
    }

    pub fn update(
        &mut self,
        detection_boxes: CowArray<f32, Ix2>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        let tracklet_boxes = self.sort_tracker.predict_and_cleanup();

        let (high_score_detections, low_score_detections) = self.split_detections(detection_boxes);

        let unmatched_high_score_detections = self
            .sort_tracker
            .update_tracklets(high_score_detections.view(), tracklet_boxes.view())?;

        let unmatched_track_box_data: Vec<f32> = tracklet_boxes
            .outer_iter()
            .zip(
                self.sort_tracker
                    .tracklets
                    .iter()
                    .map(|(_, tracker)| tracker.steps_since_update == 0),
            )
            .filter_map(|(box_arr, matched)| if matched { None } else { Some(box_arr) })
            .flatten()
            .copied()
            .collect();
        let unmatched_track_boxes: Array2<f32> = Array2::from_shape_vec(
            (unmatched_track_box_data.len() / 5, 5),
            unmatched_track_box_data,
        )?;

        let unmatched_low_score_detections = self
            .sort_tracker
            .update_tracklets(low_score_detections.view(), unmatched_track_boxes.view())?;

        self.sort_tracker.remove_stale_tracklets();

        self.sort_tracker
            .create_tracklets(unmatched_high_score_detections);
        self.sort_tracker
            .create_tracklets(unmatched_low_score_detections);

        self.sort_tracker.n_steps += 1;
        Ok(self
            .sort_tracker
            .get_tracklet_boxes(return_all, return_indices))
    }
}

#[pymethods]
impl ByteTrack {
    #[new]
    #[args(
        max_age = "1",
        min_hits = "3",
        iou_threshold = "0.3",
        init_tracker_min_score = "0.8",
        high_score_threshold = "0.7",
        low_score_threshold = "0.1",
        measurement_noise = "[1., 1., 10., 10.]",
        process_noise = "[1., 1., 1., 1., 0.01, 0.01, 0.0001]"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_age: u32,
        min_hits: u32,
        iou_threshold: f32,
        init_tracker_min_score: f32,
        high_score_threshold: f32,
        low_score_threshold: f32,
        measurement_noise: [f32; 4],
        process_noise: [f32; 7],
    ) -> (Self, BaseTracker) {
        let sort_tracker = Sort::new(
            max_age,
            min_hits,
            iou_threshold,
            init_tracker_min_score,
            measurement_noise,
            process_noise,
        )
        .0;
        (
            ByteTrack {
                high_score_threshold,
                low_score_threshold,
                sort_tracker,
            },
            BaseTracker::new(),
        )
    }

    /// Update the tracker with new boxes and return position of current tracklets
    ///
    /// Parameters
    /// ----------
    /// boxes
    ///     array of boxes of shape (n_boxes, 5)
    ///     of the form [[xmin1, ymin1, xmax1, ymax1, score1], [xmin2,...],...]
    /// return_all
    ///     if true return all living trackers, including inactive (but not dead) ones
    ///     otherwise return only active trackers (those that got at least min_hits
    ///     matching boxes in a row)
    /// return_indices
    ///     if true return the indices of the boxes kept from the initial ones as the 6th dimension
    ///
    /// Returns
    /// -------
    ///    array of tracklet boxes with shape (n_tracks, 5) or (n_tracks, 6) if return_indices is true
    ///    of the form [[xmin1, ymin1, xmax1, ymax1, track_id1], [xmin2,...],...]
    #[args(boxes, return_all = "false", return_indices = "false")]
    #[pyo3(
        name = "update",
        text_signature = "(boxes, return_all = False, return_indices = False)"
    )]
    fn py_update<'py>(
        &mut self,
        _py: Python<'py>,
        boxes: &'py PyAny,
        return_all: bool,
        return_indices: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        // We allow 'boxes' to be either f32 (then we use it directly) or f64 (then we convert to f32)
        // TODO: find some way to extract this into a function...
        let boxes_py32_res: PyResult<PyReadonlyArray2<'py, f32>> = boxes.extract();
        let detection_boxes: CowArray<f32, Ix2> = match boxes_py32_res {
            Ok(ref arr) => arr.as_array().into(),
            Err(_) => boxes
                .extract::<PyReadonlyArray2<'py, f64>>()
                .map_err(|_| PyValueError::new_err("Argument 'boxes' needs to be an array of type f32/f64 and shape (n_boxes, 5)!",))?
                .as_array()
                .mapv(|x| x as f32)
                .into(),
        };
        if detection_boxes.shape()[1] != 5 {
            return Err(PyValueError::new_err(
                "Argument 'boxes' needs to have shape (n_boxes, 5)!",
            ));
        }

        return Ok(self
            .update(detection_boxes, return_all, return_indices)?
            .into_pyarray(_py));
    }

    /// Return current track boxes
    ///
    /// Parameters
    /// ----------
    /// return_all
    ///     if true return all living trackers, including inactive (but not dead) ones
    ///     otherwise return only active trackers (those that got at least min_hits
    ///     matching boxes in a row)
    /// return_indices
    ///     if true return the indices of the boxes kept from the initial ones as the 6th dimension
    ///
    /// Returns
    /// -------
    ///    array of tracklet boxes with shape (n_tracks, 5)
    ///    of the form [[xmin1, ymin1, xmax1, ymax1, track_id1], [xmin2,...],...]
    #[args(return_all = "false", return_indices = "false")]
    #[pyo3(
        name = "get_current_track_boxes",
        text_signature = "(return_all = False, return_indices = False)"
    )]
    pub fn get_current_track_boxes<'py>(
        &self,
        _py: Python<'py>,
        return_all: bool,
        return_indices: bool,
    ) -> &'py PyArray2<f32> {
        self.sort_tracker
            .get_tracklet_boxes(return_all, return_indices)
            .into_pyarray(_py)
    }

    /// Remove all current tracklets
    #[pyo3(name = "clear_trackers", text_signature = "()")]
    pub fn clear_trackers(&mut self) {
        self.sort_tracker.clear_trackers();
    }

    /// Remove the tracklet with the given track_id
    /// no effect if the track_id is not in use
    #[pyo3(name = "remove_tracker", text_signature = "(track_id)")]
    pub fn remove_tracker(&mut self, track_id: u32) {
        self.sort_tracker.remove_tracker(track_id);
    }

    #[getter]
    fn get_max_age(&self) -> u32 {
        self.sort_tracker.max_age
    }

    #[setter]
    fn set_max_age(&mut self, value: u32) {
        self.sort_tracker.max_age = value
    }

    #[getter]
    fn get_min_hits(&self) -> u32 {
        self.sort_tracker.min_hits
    }

    #[setter]
    fn set_min_hits(&mut self, value: u32) {
        self.sort_tracker.min_hits = value
    }

    #[getter]
    fn get_iou_threshold(&self) -> f32 {
        self.sort_tracker.iou_threshold
    }

    #[setter]
    fn set_iou_threshold(&mut self, value: f32) {
        self.sort_tracker.iou_threshold = value
    }

    #[getter]
    fn get_init_tracker_min_score(&self) -> f32 {
        self.sort_tracker.init_tracker_min_score
    }

    #[setter]
    fn set_init_tracker_min_score(&mut self, value: f32) {
        self.sort_tracker.init_tracker_min_score = value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_byetrack() {
        let boxes = array![
            [270.71, 1.6277, 374.85, 276.68, 0.85113],
            [376.79, 13.419, 464.71, 250.11, 0.79943],
            [198.16, 102.59, 243.19, 214.45, 0.71286],
            [105.97, 81.568, 120.27, 112.56, 0.59725],
            [184.77, 135.38, 209.81, 207.71, 0.5742],
            [158.85, 146.87, 175.76, 197.3, 0.5734],
            [444.9, 146.99, 485.94, 200.19, 0.52883],
            [112.16, 146.05, 129.28, 198.07, 0.52617],
            [128.5, 146.01, 146.11, 199.99, 0.51999],
            [62.405, 145.66, 81.003, 198.7, 0.46859],
            [92.517, 7.8918, 118.37, 63.64, 0.42798],
            [44.87, 78.817, 59.608, 124.28, 0.27232],
            [11.39, 48.35, 31.832, 106.23, 0.2623]
        ];

        let mut tracker = ByteTrack::new(
            5,
            2,
            0.3,
            0.25,
            0.7,
            0.1,
            [1., 1., 10., 10.],
            [1., 1., 1., 1., 0.01, 0.01, 0.0001],
        )
        .0;

        let tracks = tracker.update(boxes.view().into(), false, true).unwrap();

        println!("Tracks:");
        println!("{:?}", tracks);

        // Add assertions to verify the expected behavior
        assert_eq!(tracks.shape()[0], 13); // Ensure the number of tracks is correct
        assert_eq!(tracks.shape()[1], 6); // Ensure the shape of the tracks array is correct
    }
}
