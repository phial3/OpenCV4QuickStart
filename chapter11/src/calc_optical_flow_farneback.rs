use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Scalar, Point, Point2f, Size, Vector},
    imgcodecs,
    imgproc,
    highgui,
    videoio,
    video,
    calib3d,
    features2d,
};

const BASE_PATH: &str = "../data/chapter11/";

pub(crate) fn run() -> Result<()> {
    // 打开视频文件
    let mut capture = videoio::VideoCapture::from_file(&format!("{}{}", BASE_PATH, "vtest.avi"), videoio::CAP_ANY)?;
    // 读取第一帧
    let mut prev_frame = Mat::default();
    if !capture.read(&mut prev_frame)? {
        println!("请确认视频文件名称是否正确");
        return Ok(());
    }

    // 转换为灰度图
    let mut prev_gray = Mat::default();
    imgproc::cvt_color(&prev_frame, &mut prev_gray, imgproc::COLOR_BGR2GRAY, 0)?;

    loop {
        let mut next_frame = Mat::default();
        // 读取下一帧
        if !capture.read(&mut next_frame)? {
            break;
        }
        highgui::imshow("视频图像", &next_frame)?;

        // 计算稠密光流
        let mut next_gray = Mat::default();
        imgproc::cvt_color(&next_frame, &mut next_gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut flow = Mat::default();

        // 计算 Farneback 光流
        video::calc_optical_flow_farneback(
            &prev_gray,
            &next_gray,
            &mut flow,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )?;

        // 创建x和y方向的速度矩阵
        let mut x_v = Mat::zeros_size(prev_frame.size()?, opencv::core::CV_32FC1)?.to_mat()?;
        let mut y_v = Mat::zeros_size(prev_frame.size()?, opencv::core::CV_32FC1)?.to_mat()?;

        // 提取两个方向的速度
        for row in 0..flow.rows() {
            for col in 0..flow.cols() {
                let flow_xy = *flow.at_2d::<Point2f>(row, col)?;
                *x_v.at_2d_mut::<f32>(row, col)? = flow_xy.x;
                *y_v.at_2d_mut::<f32>(row, col)? = flow_xy.y;
            }
        }

        // 计算向量角度和幅值
        let mut magnitude = Mat::default();
        let mut angle = Mat::default();
        opencv::core::cart_to_polar(&x_v, &y_v, &mut magnitude, &mut angle, true)?;

        // 将角度转换为角度制
        opencv::core::multiply(&angle.clone(),
                               &Scalar::new(180.0 / opencv::core::CV_PI / 2.0, 0.0, 0.0, 0.0),
                               &mut angle, 1.0, -1)?;

        // 归一化幅值到0-255
        opencv::core::normalize(&magnitude.clone(), &mut magnitude, 0.0, 255.0,
                                opencv::core::NORM_MINMAX, -1, &opencv::core::no_array())?;

        // 转换为8位无符号整型
        let mut magnitude_abs = Mat::default();
        let mut angle_abs = Mat::default();
        opencv::core::convert_scale_abs(&magnitude, &mut magnitude_abs, 1.0, 0.0)?;
        opencv::core::convert_scale_abs(&angle, &mut angle_abs, 1.0, 0.0)?;

        // 创建HSV图像和三个通道
        let mut h_channel = Mat::default();
        let mut s_channel = Mat::zeros_size(prev_frame.size()?, opencv::core::CV_8UC1)?.to_mat()?;
        let mut v_channel = Mat::default();

        // 设置各个通道的值
        angle_abs.copy_to(&mut h_channel)?;  // H通道 - 色调
        s_channel.set_to(&Scalar::all(255.0), &opencv::core::no_array())?;  // S通道 - 饱和度
        magnitude_abs.copy_to(&mut v_channel)?;  // V通道 - 亮度

        // 创建通道向量
        let mut channels = Vector::<Mat>::new();
        channels.push(h_channel);
        channels.push(s_channel);
        channels.push(v_channel);

        // 合并通道创建HSV图像
        let mut hsv = Mat::default();
        opencv::core::merge(&channels, &mut hsv)?;

        // 转换到BGR颜色空间
        let mut rgb_img = Mat::default();
        imgproc::cvt_color(&hsv, &mut rgb_img, imgproc::COLOR_HSV2BGR, 0)?;

        // 显示结果
        highgui::imshow("运动检测结果", &rgb_img)?;

        let key = highgui::wait_key(5)?;
        if key == 27 { // 按下ESC退出
            break;
        }

        // 更新前一帧
        next_gray.copy_to(&mut prev_gray)?;
    }

    highgui::wait_key(0)?;

    Ok(())
}
