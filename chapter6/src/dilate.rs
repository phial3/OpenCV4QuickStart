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

    // 生成用于膨胀的原图像
    let mut src = Mat::zeros_size(Size::new(6, 6), opencv::core::CV_8U)?.to_mat()?;
    {
        let data = vec![
            0, 0, 0, 0, 255, 0,
            0, 255, 255, 255, 255, 255,
            0, 255, 255, 255, 255, 0,
            0, 255, 255, 255, 255, 0,
            0, 255, 255, 255, 255, 0,
            0, 0, 0, 0, 0, 0
        ];
        src.data_typed_mut::<u8>()?.copy_from_slice(&data);
    }

    // 创建结构元素
    let struct1 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        opencv::core::Point::new(-1, -1)
    )?;

    let struct2 = imgproc::get_structuring_element(
        imgproc::MORPH_CROSS,
        Size::new(3, 3),
        Point::new(-1, -1)
    )?;

    // 膨胀
    let mut dilate_src = Mat::default();
    imgproc::dilate(&src,
                    &mut dilate_src,
                    &struct2,
                    Point::new(-1, -1),
                    1,
                    opencv::core::BORDER_CONSTANT,
                    Scalar::all(0.0))?;

    // 显示原图和膨胀后的图像
    highgui::named_window("src", highgui::WINDOW_GUI_NORMAL)?;
    highgui::named_window("dilateSrc", highgui::WINDOW_GUI_NORMAL)?;
    highgui::imshow("src", &src)?;
    highgui::imshow("dilateSrc", &dilate_src)?;

    // 读取图像
    let learn_cv_black = imgcodecs::imread(&format!("{}{}", BASE_PATH, "LearnCV_black.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    let learn_cv_white = imgcodecs::imread(&format!("{}{}", BASE_PATH, "LearnCV_white.png"), imgcodecs::IMREAD_ANYCOLOR)?;
    if learn_cv_black.empty() || learn_cv_white.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 黑背景图像膨胀
    let mut dilate_black1 = Mat::default();
    let mut dilate_black2 = Mat::default();
    imgproc::dilate(&learn_cv_black, &mut dilate_black1, &struct1, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    imgproc::dilate(&learn_cv_black, &mut dilate_black2, &struct2, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;

    // 显示黑背景膨胀结果
    highgui::imshow("LearnCV_black", &learn_cv_black)?;
    highgui::imshow("dilate_black1", &dilate_black1)?;
    highgui::imshow("dilate_black2", &dilate_black2)?;

    // 白背景图像膨胀
    let mut dilate_write1 = Mat::default();
    let mut dilate_write2 = Mat::default();
    imgproc::dilate(&learn_cv_white, &mut dilate_write1, &struct1, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
    imgproc::dilate(&learn_cv_white, &mut dilate_write2, &struct2, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;

    // 显示白背景膨胀结果
    highgui::imshow("LearnCV_write", &learn_cv_white)?;
    highgui::imshow("dilate_write1", &dilate_write1)?;
    highgui::imshow("dilate_write2", &dilate_write2)?;

    // 腐蚀和按位运算
    let mut erode_black1 = Mat::default();
    let mut result_xor = Mat::default();
    let mut result_and = Mat::default();
    imgproc::erode(&learn_cv_black, &mut erode_black1, &struct1, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;

    opencv::core::bitwise_xor(&erode_black1, &dilate_write1, &mut result_xor, &opencv::core::no_array())?;
    opencv::core::bitwise_and(&erode_black1, &dilate_write1, &mut result_and, &opencv::core::no_array())?;

    // 显示按位运算结果
    highgui::imshow("resultXor", &result_xor)?;
    highgui::imshow("resultAnd", &result_and)?;
    highgui::wait_key(0)?;

    Ok(())
}
