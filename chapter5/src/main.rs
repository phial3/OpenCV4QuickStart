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
mod median_blur;
mod salt_and_pepper;
mod scharr;
mod sobel;
mod self_filter;

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
    // let _ = laplacian::run();

    // 11. median blur
    // let _ = median_blur::run();

    // 12. salt and pepper
    // let _ = salt_and_pepper::run();

    // 13. scharr
    // let _ = scharr::run();

    // 14. sobel
    // let _ = sobel::run();

    // 15. self filter
    let _ = self_filter::run();
}
