mod bilateral_filter;
mod image_blur;
mod box_filter;
mod canny;
mod edge;
mod filter;
mod gauss_noise;
mod gaussian_blur;
mod get_deriv_kernels;
mod laplacian;

fn main() {
    // 1. bilateral filter
    // let _ = bilateral_filter::run();

    // 2. image blur
    // let _ = image_blur::run();

    // 3. box filter
    // let _ = box_filter::run();

    // 4. canny edge detector
    // let _ = canny::run();

    // 5. edge detection
    // let _ = edge::run();

    // 6. filter
    // let _ = filter::run();

    // 7. gaussian noise
    // let _ = gauss_noise::run();

    // 8. gaussian blur
    // let _ = gaussian_blur::run();

    // 9. get derivative kernels
    // let _ = get_deriv_kernels::run();

    // 10. laplacian
    let _ = laplacian::run();
}
