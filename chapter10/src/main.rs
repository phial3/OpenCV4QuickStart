mod calibrate_camera;
mod chessboard;
mod concat;
mod homogeneous;
mod pnp_and_ransac;
mod project_points;
mod stereo_calibrate;
mod stereo_rectify;

fn main() {
    // 1. calibrate_camera
    // let _ = calibrate_camera::run();

    // 2. chessboard
    // let _ = chessboard::run();

    // 3. concat
    // let _ = concat::run();

    // 4. homogeneous
    // let _ = homogeneous::run();

    // 5. pnp_and_ransac
    // let _ = pnp_and_ransac::run();

    // 6. project_points
    // let _ = project_points::run();

    // 7. stereo_calibrate
    // let _ = stereo_calibrate::run();

    // 8. stereo_rectify
    let _ = stereo_rectify::run();
}
