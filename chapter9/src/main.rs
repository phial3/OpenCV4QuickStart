mod corner_harris;
mod corner_sub_pix;

fn main() {
    // 1. corner_harris::run()
    let _ = corner_harris::run();

    // 2. corner_sub_pix::run()
    let _ = corner_sub_pix::run();
}
