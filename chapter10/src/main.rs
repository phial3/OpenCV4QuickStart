mod calibrate_camera;
mod chessboard;
mod concat;
mod homogeneous;
mod pnp_and_ransac;

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
    let _ = pnp_and_ransac::run();
}
