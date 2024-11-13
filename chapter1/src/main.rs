use anyhow::Result;
use opencv::{prelude::*, core::Vector, imgproc, imgcodecs, highgui};

const BASE_PATH: &str = "../data/chapter1/";

fn main() -> Result<()> {
    let img = imgcodecs::imread(format!("{}{}", BASE_PATH, "lena.png").as_str(), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("Failed to load image");
    }

    // 打印下图片信息
    println!("Image size: {}x{}", img.cols(), img.rows());
    println!("Image channels: {}", img.channels());
    println!("Image depth: {}", img.depth());

    // 显示图片
    highgui::named_window("Lena", 0)?;
    highgui::imshow("Lena", &img)?;
    highgui::wait_key(0)?;

    // 对图像做一些处理（例如，转为灰度图像）
    let mut gray_image = Mat::default();
    imgproc::cvt_color(&img, &mut gray_image, imgproc::COLOR_BGR2GRAY, 0)?;

    // 保存图像
    let mut params = Vector::new();
    params.push(imgcodecs::IMWRITE_PNG_COMPRESSION);
    params.push(1);
    // 设置 params 方式
    imgcodecs::imwrite("lena_gray.png", &gray_image, &params)?;
    // 这是 def 方式
    // imgcodecs::imwrite_def("lena_gray.png", &gray_image)?;

    Ok(())
}
