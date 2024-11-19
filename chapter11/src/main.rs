mod absdiff;
mod calc_optical_flow_farneback;
mod calc_optical_flow_pyr_lk;


fn main() {
    // 1. absdiff
    // let _ = absdiff::run();

    // 2. calc_optical_flow_farneback
    // let _ = calc_optical_flow_farneback::run();

    // 3. calc_optical_flow_pyr_lk
    let _ = calc_optical_flow_pyr_lk::run();
}
