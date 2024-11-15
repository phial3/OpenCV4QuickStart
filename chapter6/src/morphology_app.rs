
use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Size, Rect, Point, RNG, Vec3b, Scalar, Vector},
    imgcodecs,
    imgproc,
    highgui,
};

const BASE_PATH: &str = "../data/chapter6/";

pub(crate) fn run() -> Result<()> {
    // 用于验证形态学应用的二值化矩阵
    let mut src = Mat::zeros_size(Size::new(9, 12), opencv::core::CV_8U)?.to_mat()?;
    {
        let data = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
            0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        src.data_typed_mut::<u8>()?.copy_from_slice(&data);
    }
    highgui::named_window("src", highgui::WINDOW_NORMAL)?;
    highgui::imshow("src", &src)?;

    // 3×3矩形结构元素
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(3, 3), opencv::core::Point::new(-1, -1))?;
    // 对二值化矩阵进行形态学操作
    let mut open = Mat::default();
    let mut close = Mat::default();
    let mut gradient = Mat::default();
    let mut tophat = Mat::default();
    let mut blackhat = Mat::default();
    let mut hitmiss = Mat::default();
    // 开运算
    imgproc::morphology_ex(&src, &mut open, imgproc::MORPH_OPEN, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::named_window("open", highgui::WINDOW_NORMAL)?;
    highgui::imshow("open", &open)?;
    // 闭运算
    imgproc::morphology_ex(&src, &mut close, imgproc::MORPH_CLOSE, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::named_window("close", highgui::WINDOW_NORMAL)?;
    highgui::imshow("close", &close)?;
    // 梯度运算
    imgproc::morphology_ex(&src, &mut gradient, imgproc::MORPH_GRADIENT, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::named_window("gradient", highgui::WINDOW_NORMAL)?;
    highgui::imshow("gradient", &gradient)?;
    // 顶帽运算
    imgproc::morphology_ex(&src, &mut tophat, imgproc::MORPH_TOPHAT, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::named_window("tophat", highgui::WINDOW_NORMAL)?;
    highgui::imshow("tophat", &tophat)?;
    // 黑帽运算
    imgproc::morphology_ex(&src, &mut blackhat, imgproc::MORPH_BLACKHAT, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::named_window("blackhat", highgui::WINDOW_NORMAL)?;
    highgui::imshow("blackhat", &blackhat)?;
    // 击中击不中变换
    imgproc::morphology_ex(&src, &mut hitmiss, imgproc::MORPH_HITMISS, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::named_window("hitmiss", highgui::WINDOW_NORMAL)?;
    highgui::imshow("hitmiss", &hitmiss)?;
    // 用图像验证形态学操作效果
    let mut keys = imgcodecs::imread(&format!("{}keys.jpg", BASE_PATH), imgcodecs::IMREAD_GRAYSCALE)?;
    highgui::imshow("原图像", &keys)?;
    imgproc::threshold(&keys.clone(), &mut keys, 80.0, 255.0, imgproc::THRESH_BINARY)?;
    highgui::imshow("二值化后的keys", &keys)?;
    // 5×5矩形结构元素
    let kernel_keys = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(5, 5), Point::new(-1, -1))?;

    let mut open_keys = Mat::default();
    let mut close_keys = Mat::default();
    let mut gradient_keys = Mat::default();
    let mut tophat_keys = Mat::default();
    let mut blackhat_keys = Mat::default();
    let mut hitmiss_keys = Mat::default();
    // 开运算
    imgproc::morphology_ex(&keys, &mut open_keys, imgproc::MORPH_OPEN, &kernel_keys, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("open_keys", &open_keys)?;
    // 闭运算
    imgproc::morphology_ex(&keys, &mut close_keys, imgproc::MORPH_CLOSE, &kernel_keys, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("close_keys", &close_keys)?;
    // 梯度运算
    imgproc::morphology_ex(&keys, &mut gradient_keys, imgproc::MORPH_GRADIENT, &kernel_keys, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("gradient_keys", &gradient_keys)?;
    // 顶帽运算
    imgproc::morphology_ex(&keys, &mut tophat_keys, imgproc::MORPH_TOPHAT, &kernel_keys, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("tophat_keys", &tophat_keys)?;
    // 黑帽运算
    imgproc::morphology_ex(&keys, &mut blackhat_keys, imgproc::MORPH_BLACKHAT, &kernel_keys, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("blackhat_keys", &blackhat_keys)?;
    // 击中击不中变换
    imgproc::morphology_ex(&keys, &mut hitmiss_keys, imgproc::MORPH_HITMISS, &kernel_keys, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    highgui::imshow("hitmiss_keys", &hitmiss_keys)?;
    highgui::wait_key(0)?;

    Ok(())
}
