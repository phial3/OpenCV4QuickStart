use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Point, Size, Vector},
    imgcodecs,
    imgproc,
    highgui,
    videoio,
    calib3d,
    features2d,
};

const BASE_PATH: &str = "../data/chapter11/";

pub(crate) fn run() -> Result<()> {
    // 加载视频文件
    let mut capture = videoio::VideoCapture::from_file(&format!("{}{}", BASE_PATH, "bike.avi"), videoio::CAP_ANY)?;
    // 检查视频是否成功打开
    if !capture.is_opened()? {
        panic!("请确认视频文件是否正确");
    }

    // 获取视频信息
    let fps = capture.get(videoio::CAP_PROP_FPS)? as i32;
    let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let num_of_frames = capture.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;

    println!("视频宽度：{} 视频高度：{} 视频帧率：{} 视频总帧数：{}", width, height, fps, num_of_frames);

    // 读取第一帧并进行灰度转换
    let mut pre_frame = Mat::default();
    let mut pre_gray = Mat::default();
    capture.read(&mut pre_frame)?;
    imgproc::cvt_color(&pre_frame, &mut pre_gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // 高斯模糊处理
    imgproc::gaussian_blur(
        &pre_gray.clone(),
        &mut pre_gray,
        Size::new(0, 0),
        15.0,
        15.0,
        opencv::core::BORDER_DEFAULT
    )?;

    let mut binary = Mat::default();
    let mut frame = Mat::default();
    let mut gray = Mat::default();

    // 创建形态学操作的结构元素
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(7, 7),
        Point::new(-1, -1)
    )?;

    loop {
        // 读取当前帧
        if !capture.read(&mut frame)? {
            break;
        }

        // 转换为灰度图并进行高斯模糊
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        imgproc::gaussian_blur(
            &gray.clone(),
            &mut gray,
            Size::new(0, 0),
            15.0,
            15.0,
            opencv::core::BORDER_DEFAULT
        )?;

        // 计算帧差
        opencv::core::absdiff(&gray, &pre_gray, &mut binary)?;

        // 二值化处理
        imgproc::threshold(
            &binary.clone(),
            &mut binary,
            10.0,
            255.0,
            imgproc::THRESH_BINARY | imgproc::THRESH_OTSU
        )?;

        // 开运算去噪
        imgproc::morphology_ex(
            &binary.clone(),
            &mut binary,
            imgproc::MORPH_OPEN,
            &kernel,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?
        )?;

        // 显示结果
        highgui::imshow("input", &frame)?;
        highgui::imshow("result", &binary)?;

        // 将当前帧复制为前一帧（注释掉此行为固定背景）
        // gray.copy_to(&mut pre_gray)?;

        // 检查是否按下 ESC 键
        let key = highgui::wait_key(5)?;
        if key == 27 {
            break;
        }
    }

    highgui::wait_key(0)?;

    Ok(())
}
