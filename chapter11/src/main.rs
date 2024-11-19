mod absdiff;
mod calc_optical_flow_farneback;
mod calc_optical_flow_pyr_lk;
mod cam_shift;
mod mean_shift;

fn main() {
    // 1. absdiff
    // let _ = absdiff::run();

    // 2. calc_optical_flow_farneback
    // let _ = calc_optical_flow_farneback::run();

    // 3. calc_optical_flow_pyr_lk
    // let _ = calc_optical_flow_pyr_lk::run();

    // 4. cam_shift
    // let _ = cam_shift::run();

    // 5. mean_shift
    let _ = mean_shift::run();
}
