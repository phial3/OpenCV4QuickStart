mod bilateral_filter;
mod image_blur;
mod box_filter;
mod canny;

fn main() {
    // 1. bilateral filter
    // let _ = bilateral_filter::run();

    // 2. image blur
    // let _ = image_blur::run();

    // 3. box filter
    // let _ = box_filter::run();

    // 4. canny edge detector
    let _ = canny::run();
}
