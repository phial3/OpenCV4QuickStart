
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
    // 读取图像
    let img = imgcodecs::imread(&format!("{}{}", BASE_PATH, "rice.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }
    // 显示原图
    highgui::imshow("原图", &img)?;
    let mut rice = Mat::default();
    let mut rice_bw = Mat::default();

    // 将图像转为灰度图像
    imgproc::cvt_color(&img, &mut rice, imgproc::COLOR_BGR2GRAY, 0)?;
    // 二值化处理
    imgproc::threshold(&rice, &mut rice_bw, 50.0, 255.0, imgproc::THRESH_BINARY)?;
    // 生成随机颜色
    let mut rng = RNG::new(10086)?;
    let mut out = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    // 统计连通域
    let number = imgproc::connected_components_with_stats(&rice_bw, &mut out, &mut stats, &mut centroids, 8, opencv::core::CV_16U)?;
    // 生成随机颜色
    let mut colors = Vec::with_capacity(number as usize);
    for _ in 0..number {
        let vec3 = Vec3b::from_array([
            rng.uniform(0, 256)? as u8,
            rng.uniform(0, 256)? as u8,
            rng.uniform(0, 256)? as u8,
        ]);
        colors.push(vec3);
    }
    // 创建结果图像
    let mut result = img.clone();
    for i in 1..number {
        // 中心位置
        let center_x = centroids.at_2d::<f64>(i, 0)?;
        let center_y = centroids.at_2d::<f64>(i, 1)?;
        // 矩形边框
        let x = stats.at_2d::<i32>(i, 0)?;
        let y = stats.at_2d::<i32>(i, 1)?;
        let w = stats.at_2d::<i32>(i, 2)?;
        let h = stats.at_2d::<i32>(i, 3)?;
        let area = stats.at_2d::<i32>(i, 4)?;

        // 中心位置绘制
        imgproc::circle(&mut result,
                        Point::new(*center_x as i32, *center_y as i32),
                        2,
                        Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        imgproc::LINE_8,
                        0)?;

        // 外接矩形
        let rect = Rect::new(*x, *y, *w, *h);
        imgproc::rectangle(&mut result,
                           rect,
                           Scalar::new(colors[i as usize][0] as f64, colors[i as usize][1] as f64, colors[i as usize][2] as f64, 0.0),
                           1,
                           imgproc::LINE_8,
                           0)?;

        // 绘制文本
        imgproc::put_text(&mut result,
                          &format!("{}", i), Point::new(*center_x as i32, *center_y as i32),
                          imgproc::FONT_HERSHEY_SIMPLEX,
                          0.5,
                          Scalar::new(0.0, 0.0, 255.0, 0.0),
                          1,
                          imgproc::LINE_8,
                          false)?;

        println!("number: {}, area: {}", i, area);
    }

    // 显示结果
    highgui::imshow("标记后的图像", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}