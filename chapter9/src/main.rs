mod corner_harris;
mod corner_sub_pix;
mod draw_keypoint;
mod good_features_to_track;
mod orb;
mod orb_match;
mod orb_match_flann;
mod orb_match_ransac;

fn main() {
    // 1. corner_harris::run()
    // let _ = corner_harris::run();

    // 2. corner_sub_pix::run()
    // let _ = corner_sub_pix::run();

    // 3. draw_keypoint::run()
    // let _ = draw_keypoint::run();

    // 4. good_features_to_track::run()
    // let _ = good_features_to_track::run();

    // 5. orb::run()
    // let _ = orb::run();

    // 6. orb_match::run()
    // let _ = orb_match::run();

    // 7. orb_match_flann::run()
    let _ = orb_match_flann::run();

    // 8. orb_match_ransac::run()
    // let _ = orb_match_ransac::run();
}
